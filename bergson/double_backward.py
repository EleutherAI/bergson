import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torchopt
from scipy.stats import spearmanr
from simple_parsing import ArgumentParser, field
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from torchopt.typing import Numeric
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.config import DataConfig, DistributedConfig
from bergson.data import load_data_string
from bergson.distributed import grad_tree, launch_distributed_run, simple_fsdp
from bergson.magic_patch import apply_dtensor_patch
from bergson.trainer import BackwardState, DataStream, Trainer, TrainerState
from bergson.utils.math import weighted_causal_lm_ce


@dataclass
class DoubleBackwardConfig:
    run_path: str = field(positional=True)
    """Directory to save checkpoints and results."""

    model: str = "EleutherAI/pythia-160m"
    """HuggingFace model name."""

    revision: str | None = None
    """Model revision (branch, tag, or commit hash)."""

    data: DataConfig = field(default_factory=DataConfig)
    """Training dataset."""

    query: DataConfig = field(default_factory=lambda: DataConfig())
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: Literal["mean", "sum"] = "mean"
    """Method for reducing query gradients across batches."""

    query_batches: int = 1
    """Number of query batches to use for computing eval gradients."""

    fsdp: bool = False
    """Whether to use FSDP for multi-GPU training."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""

    lr: float = 1e-5
    """Base learning rate after warmup."""

    warmup_steps: int = 10
    """Number of warmup steps before applying base lr."""

    batch_size: int = 8
    """Per-device batch size."""

    num_batches: int = 25
    """Number of training batches."""

    max_length: int = 256
    """Maximum token sequence length."""

    num_subsets: int = 100
    """Number of leave-one-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""


def compute_query_gradients(
    trainer: Trainer,
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_stream: DataStream,
    method: str = "mean",
) -> dict[str, torch.Tensor]:
    """Compute reduced query gradients over the query dataset.

    Iterates over the query stream, computing per-batch parameter gradients
    and reducing them (mean or sum) into a single gradient dict.

    When ``query_stream`` has padding (batch_size rounded up to world_size),
    padded examples have their labels set to ``ignore_index`` so they
    contribute zero loss.  The caller must apply a correction factor of
    ``batch_size / logical_batch_size`` after the all-reduce to account for
    the inflated denominator.
    """
    grad_accum: dict[str, torch.Tensor] | None = None
    n_batches = 0
    has_padding = query_stream._pad_per_batch > 0

    with fwd_state.activate(model) as params:
        for batch in query_stream:
            del batch["example_weight"]

            # Mask padded examples so they contribute zero loss.
            if has_padding:
                w = query_stream.weights[
                    n_batches * query_stream.batch_size
                    + query_stream.rank : (n_batches + 1)
                    * query_stream.batch_size : query_stream.world_size
                ]
                pad_mask = w == 0
                batch["labels"] = batch["labels"].clone()
                batch["labels"][pad_mask] = -100

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
    run_cfg: DoubleBackwardConfig,
):
    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model,
        revision=run_cfg.revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(f"cuda:{rank}")
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

    if run_cfg.fsdp and world_size > 1:
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
        betas=(0.95, 0.975),
        eps_root=1e-8,
    )
    trainer, fwd_state = Trainer.initialize(model, opt)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    path0 = os.path.join(ckpts_path, "state0.pt")
    save_fut = fwd_state.save(path0)

    if run_cfg.batch_size % world_size != 0:
        raise ValueError(
            f"Training batch_size ({run_cfg.batch_size}) must be divisible by "
            f"world_size ({world_size}). Padding would change training dynamics."
        )

    stream = DataStream(
        train_dataset,
        processor,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.num_batches,
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

    # Compute query gradients — batch size is rounded up to world_size if
    # needed.  Padded examples are label-masked in compute_query_gradients;
    # the correction factor below fixes the denominator after all-reduce.
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
        trainer, fwd_state, model, query_stream, run_cfg.query_method
    )

    if world_size > 1:
        reduce_op = (
            dist.ReduceOp.AVG if run_cfg.query_method == "mean" else dist.ReduceOp.SUM
        )
        for v in query_grads.values():
            dist.all_reduce(v, op=reduce_op)

        # Correct for padded denominator: .mean() divides by padded_bs but
        # we want to divide by logical_batch_size.
        if query_stream._pad_per_batch > 0:
            correction = query_stream.batch_size / query_stream._logical_batch_size
            for v in query_grads.values():
                v *= correction

    scores_path = Path(run_cfg.run_path) / "scores.npy"
    baseline_path = Path(run_cfg.run_path) / "baseline.npy"
    num_examples = len(stream.weights)

    if scores_path.exists() and baseline_path.exists():
        # Resume: load previously computed scores
        scores = torch.from_numpy(np.load(str(scores_path))).to(f"cuda:{rank}")
        baseline = float(np.load(str(baseline_path)))
        if global_rank == 0:
            print(f"Resumed scores from {scores_path}")
            print(f"Scores: {scores.tolist()}")
            print(f"Baseline: {baseline}")
            print(f"Grad sum: {scores.sum()}")
    else:
        stream.requires_grad = True
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(fwd_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        bwd_state = BackwardState(
            query_grads, opt_grads, torch.zeros_like(stream.weights)
        )

        # Compute baseline eval loss for validation
        with fwd_state.activate(model):
            baseline_batch = query_stream[0]
            del baseline_batch["example_weight"]
            baseline_loss = model(**baseline_batch).loss

        if world_size > 1:
            dist.all_reduce(baseline_loss, op=dist.ReduceOp.AVG)

        bwd_state = trainer.backward(
            ckpts_path,
            stream,
            bwd_state,
            fwd_state,
            inplace=True,
        )
        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.AVG)

        scores = bwd_state.weight_grads
        baseline = baseline_loss.item()

        # Save scores and baseline to disk
        if global_rank == 0:
            np.save(str(scores_path), scores.cpu().numpy())
            np.save(str(baseline_path), np.array(baseline))
            print(f"Saved scores to {scores_path}")
            print(f"Scores: {scores.tolist()}")
            print(f"Baseline: {baseline}")
            print(f"Grad sum: {scores.sum()}")

    stream.requires_grad = False

    # Validate attribution scores via leave-subset-out retraining
    validation_path = Path(run_cfg.run_path) / "validation.npy"
    gen = torch.Generator().manual_seed(run_cfg.seed)
    perm = torch.randperm(num_examples, generator=gen)
    subsets = perm.chunk(run_cfg.num_subsets)

    # Resume validation: load existing results and skip completed subsets
    start_subset = 0
    if validation_path.exists():
        saved = np.load(str(validation_path))
        start_subset = len(saved)
        diffs = saved[:, 0].tolist()
        score_sums = saved[:, 1].tolist()
        if global_rank == 0:
            print(f"Resumed validation from subset {start_subset}/{len(subsets)}")
    else:
        diffs = []
        score_sums = []

    if start_subset < len(subsets):
        save_fut.result()  # ensure state0 is saved before loading in loop
        fwd_state.load(path0)

        for i, subset in enumerate(subsets):
            if i < start_subset:
                continue

            stream.weights.fill_(1.0)
            stream.weights[subset] = 0.0

            for x in stream:
                fwd_state = trainer.step(fwd_state, x)

            with fwd_state.activate(model):
                eval_batch = query_stream[0]
                del eval_batch["example_weight"]
                loss = model(**eval_batch).loss

            if world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

            diffs.append(baseline - loss.item())
            score_sums.append(scores[subset].sum().item())

            # Save validation progress to disk
            if global_rank == 0:
                val_arr = np.column_stack([diffs, score_sums])
                np.save(str(validation_path), val_arr)

            corr = spearmanr(diffs, score_sums)
            if global_rank == 0:
                print(f"Loss diff: {diffs[-1]}")
                print(f"Score: {score_sums[-1]}")
                print(f"Spearman correlation: {corr}")

            fwd_state.load(path0)
    else:
        if global_rank == 0:
            corr = spearmanr(diffs, score_sums)
            print(f"Validation already complete. Spearman correlation: {corr}")


def double_backward(run_cfg: DoubleBackwardConfig, dist_cfg: DistributedConfig):
    run_path = Path(run_cfg.run_path)
    run_path.mkdir(parents=True, exist_ok=True)
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

    launch_distributed_run(
        "double_backward", worker, [train_ds, query_ds, run_cfg], dist_cfg
    )


def main():
    parser = ArgumentParser()
    parser.add_arguments(DoubleBackwardConfig, dest="run_cfg")
    parser.add_arguments(DistributedConfig, dest="dist_cfg")
    args = parser.parse_args()

    run_cfg: DoubleBackwardConfig = args.run_cfg
    dist_cfg: DistributedConfig = args.dist_cfg

    double_backward(run_cfg, dist_cfg)


if __name__ == "__main__":
    main()
