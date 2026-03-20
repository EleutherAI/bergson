import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torchopt
from scipy.stats import describe, spearmanr
from simple_parsing import ArgumentParser, field
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from torchopt.typing import Numeric
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DataConfig, DistributedConfig
from ..data import load_data_string
from ..distributed import grad_tree, launch_distributed_run, simple_fsdp
from ..utils.math import weighted_causal_lm_ce
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .trainer import BackwardState, Trainer, TrainerState


@dataclass
class MagicConfig:
    run_path: str = field(positional=True)
    """Directory to save checkpoints and results."""

    overwrite: bool = False
    """Whether to overwrite the run directory if it already exists."""

    model: str = "EleutherAI/pythia-160m"
    """HuggingFace model name."""

    revision: str | None = None
    """Model revision (branch, tag, or commit hash)."""

    data: DataConfig = field(default_factory=DataConfig)
    """Training dataset."""

    query: DataConfig = field(default_factory=DataConfig)
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: Literal["mean", "sum"] = "mean"
    """Method for reducing query gradients across batches."""

    query_batches: int = 1
    """Number of query batches to use for computing eval gradients."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""

    lr: float = 1e-5
    """Base learning rate after warmup."""

    warmup_steps: int = 10
    """Number of warmup steps before applying base lr."""

    batch_size: int = 8
    """Per-device batch size."""

    num_steps: int = 100
    """Number of training steps."""

    max_length: int = 256
    """Maximum token sequence length."""

    num_subsets: int = 100
    """Number of leave-one-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""

    beta1: float = 0.95
    """Beta1 for AdamW optimizer."""

    beta2: float = 0.975
    """Beta2 for AdamW optimizer."""

    eps_root: float = 1e-8
    """Epsilon root for AdamW optimizer. Use 1e-2 for better stability
    with small models."""


def compute_query_gradients(
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_stream: DataStream,
    method: str = "mean",
) -> dict[str, torch.Tensor]:
    """Compute reduced query gradients over the query dataset.

    Iterates over the query stream, computing per-batch parameter gradients
    and reducing them (mean or sum) into a single gradient dict.
    """
    grad_accum: dict[str, torch.Tensor] | None = None
    n_batches = 0

    with fwd_state.activate(model) as params:
        for batch in query_stream:
            del batch["example_weight"]
            loss = model(**batch).loss
            grads = grad_tree(loss, params)

            if grad_accum is None:
                grad_accum = {k: g.detach().clone() for k, g in grads.items()}
            else:
                for k, g in grads.items():
                    grad_accum[k] += g.detach()
            n_batches += 1

    assert grad_accum is not None, "Query stream was empty"

    if method == "mean":
        for k in grad_accum:
            grad_accum[k] /= n_batches

    return grad_accum


def worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset,
    query_dataset,
    run_cfg: MagicConfig,
):
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model,
        revision=run_cfg.revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(f"cuda:{rank}")  # type: ignore[reportArgumentType]

    if run_cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
        )

    processor = AutoTokenizer.from_pretrained(run_cfg.model)
    processor.pad_token = processor.eos_token

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

        apply_dtensor_patch()
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            model = simple_fsdp(model)

    def schedule(step: Numeric) -> Numeric:
        if step < run_cfg.warmup_steps:
            return 0.0
        return run_cfg.lr

    opt = torchopt.adamw(
        schedule,
        betas=(run_cfg.beta1, run_cfg.beta2),
        eps_root=run_cfg.eps_root,
    )
    trainer, fwd_state = Trainer.initialize(model, opt)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    path0 = os.path.join(ckpts_path, "state0.pt")
    save_fut = fwd_state.save(path0)

    stream = DataStream(
        train_dataset,
        processor,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.num_steps,
        device=f"cuda:{rank}",
        max_length=run_cfg.max_length,
        input_key=run_cfg.data.prompt_column,
    )
    fwd_state = trainer.train(
        fwd_state,
        stream,
        inplace=True,
        save_dir=ckpts_path,
    )

    # Compute query gradients
    query_stream = DataStream(
        query_dataset,
        processor,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.query_batches,
        device=f"cuda:{rank}",
        max_length=run_cfg.max_length,
        input_key=run_cfg.query.prompt_column,
    )

    query_grads = compute_query_gradients(
        fwd_state, model, query_stream, run_cfg.query_method
    )

    stream.requires_grad = True
    opt_grads = [
        torch.zeros_like(buf)
        for buf in tree_iter(fwd_state.opt_state)
        if isinstance(buf, torch.Tensor) and buf.is_floating_point()
    ]
    bwd_state = BackwardState(query_grads, opt_grads, torch.zeros_like(stream.weights))

    # Compute baseline eval loss for validation
    with fwd_state.activate(model):
        baseline = torch.tensor(0.0, device=stream.weights.device)
        for batch in query_stream:
            del batch["example_weight"]

            baseline += model(**batch).loss.detach() / len(query_stream)

    if world_size > 1:
        dist.all_reduce(baseline, op=dist.ReduceOp.AVG)

    bwd_state = trainer.backward(
        ckpts_path,
        stream,
        bwd_state,
        fwd_state,
        inplace=True,
    )
    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

    baseline = baseline.item()
    scores = bwd_state.weight_grads.cpu()
    if global_rank == 0:
        print(f"Baseline loss: {baseline}")

        summ = describe(scores)
        print(f"Score summary: {summ}")

        score_path = os.path.join(run_cfg.run_path, "scores.pt")
        torch.save(scores, score_path)
        print(f"Saved attribution scores to {score_path}")

    stream.requires_grad = False

    # Validate attribution scores via leave-subset-out retraining
    diffs = []
    score_sums = []

    gen = torch.Generator().manual_seed(run_cfg.seed)
    perm = torch.randperm(len(stream.weights), generator=gen)
    subsets = perm.chunk(run_cfg.num_subsets)

    pbar = tqdm(subsets, desc="Validating", disable=global_rank != 0)
    save_fut.result()  # ensure state0 is saved before loading in loop

    for subset in pbar:
        fwd_state.load(path0)

        stream.weights.fill_(1.0)
        stream.weights[subset] = 0.0

        for x in stream:
            fwd_state = trainer.step(fwd_state, x)

        with fwd_state.activate(model):
            loss = torch.tensor(0.0, device=stream.weights.device)
            for batch in query_stream:
                del batch["example_weight"]

                loss += model(**batch).loss.detach() / len(query_stream)

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        diffs.append(baseline - loss.item())
        score_sums.append(scores[subset].sum().item())

        corr = spearmanr(diffs, score_sums)
        if global_rank == 0:
            pbar.set_postfix({"rho": corr.statistic})

    if global_rank == 0:
        corr = spearmanr(diffs, score_sums)
        print(f"Final Spearman correlation: {corr.statistic:.4f} (p={corr.pvalue:.2e})")


def run_magic(run_cfg: MagicConfig, dist_cfg: DistributedConfig):
    run_path = Path(run_cfg.run_path)
    if run_path.exists():
        if run_cfg.overwrite:
            shutil.rmtree(run_path)
        else:
            raise FileExistsError(
                f"Run path {run_path} already exists. Use --overwrite to overwrite it."
            )

    run_path.mkdir(parents=True)
    with (run_path / "run_config.json").open("w") as f:
        json.dump(asdict(run_cfg), f, indent=2)
    with (run_path / "dist_config.json").open("w") as f:
        json.dump(asdict(dist_cfg), f, indent=2)

    train_ds = load_data_string(
        run_cfg.data.dataset,
        run_cfg.data.split,
        run_cfg.data.subset,
        run_cfg.data.data_args,
    )

    query_ds = load_data_string(
        run_cfg.query.dataset,
        run_cfg.query.split,
        run_cfg.query.subset,
        run_cfg.query.data_args,
    )

    launch_distributed_run("run_magic", worker, [train_ds, query_ds, run_cfg], dist_cfg)


def main():
    parser = ArgumentParser()
    parser.add_arguments(MagicConfig, dest="run_cfg")
    parser.add_arguments(DistributedConfig, dest="dist_cfg")
    args = parser.parse_args()

    run_cfg: MagicConfig = args.run_cfg
    dist_cfg: DistributedConfig = args.dist_cfg

    run_magic(run_cfg, dist_cfg)


if __name__ == "__main__":
    main()
