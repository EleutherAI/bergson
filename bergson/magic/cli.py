import math
import os
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset
from scipy.stats import describe, spearmanr
from simple_parsing import ArgumentParser, field
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from torchopt.typing import Numeric
from tqdm import tqdm
from transformers import PreTrainedModel

from ..config import AttributionConfig, DataConfig, DistributedConfig, TrainingConfig
from ..distributed import grad_tree, launch_distributed_run, simple_fsdp
from ..utils import assert_type
from ..utils.math import weighted_causal_lm_ce
from ..utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .trainer import BackwardState, Trainer, TrainerState, sorted_checkpoints


@dataclass
class TrainingConfig:
    """Optimizer and learning rate schedule configuration."""

    lr: float = 1e-5
    """Peak learning rate."""

    lr_start: float = 1e-6
    """Initial learning rate at the beginning of warmup."""

    lr_end_ratio: float = 0.1
    """Final learning rate as a fraction of the peak learning rate."""

    warmup_ratio: float = 0.25
    """Fraction of total training steps used for linear warmup."""

    batch_size: int = 8
    """Per-device batch size."""

    num_steps: int = 100
    """Number of training steps. Overridden by num_epochs if set."""

    num_epochs: int = 0
    """Number of full passes over the training data. If set (> 0),
    overrides num_steps."""

    max_length: int = 256
    """Maximum token sequence length."""

    beta1: float = 0.95
    """Beta1 for AdamW optimizer."""

    beta2: float = 0.975
    """Beta2 for AdamW optimizer."""

    eps_root: float = 1e-8
    """Epsilon root for AdamW optimizer. Use 1e-2 for better stability
    with small models."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""

    def schedule(self, num_steps: int) -> "Callable[[Numeric], Numeric]":
        """Return a one-cycle linear schedule: warmup from lr_start to lr,
        then decay to lr * lr_end_ratio."""
        warmup_steps = int(num_steps * self.warmup_ratio)
        lr_start = self.lr_start
        lr_peak = self.lr
        lr_end = self.lr * self.lr_end_ratio

        def _schedule(step: Numeric) -> Numeric:
            if step < warmup_steps:
                # Linear warmup
                return lr_start + (lr_peak - lr_start) * step / max(warmup_steps, 1)
            else:
                # Linear decay
                decay_steps = num_steps - warmup_steps
                progress = (step - warmup_steps) / max(decay_steps, 1)
                return lr_peak - (lr_peak - lr_end) * progress

        return _schedule


@dataclass
class MagicConfig(AttributionConfig):
    """Special config for MAGIC attribution."""

    query: DataConfig = field(
        default_factory=lambda: DataConfig(split="train"),
    )
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: Literal["mean", "sum"] = "mean"
    """Method for reducing query gradients across batches."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training hyperparameters and learning rate schedule."""

    num_subsets: int = 100
    """Number of leave-one-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""

    resume: bool = False
    """Resume a previously interrupted run from the last checkpoint."""

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    def __post_init__(self):
        assert not self.fsdp, "PyTorch FSDP is not currently supported for MAGIC."


def compute_query_gradients(
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_stream: DataStream,
    method: str = "mean",
) -> tuple[dict[str, torch.Tensor], float]:
    """Compute reduced query gradients over the query dataset.

    Iterates over the query stream, computing per-batch parameter gradients
    and reducing them (mean or sum) into a single gradient dict.
    """
    grad_accum: dict[str, torch.Tensor] | None = None
    loss_accum = 0.0
    n_batches = 0

    with fwd_state.activate(model) as params:
        for batch in tqdm(query_stream, desc="Query"):
            del batch["example_weight"]
            loss = model(**batch).loss
            grads = grad_tree(loss, params)

            if grad_accum is None:
                grad_accum = {k: g.detach().clone() for k, g in grads.items()}
            else:
                for k, g in grads.items():
                    grad_accum[k] += g.detach()

            loss_accum += loss.detach() / len(query_stream)
            n_batches += 1

    assert grad_accum is not None, "Query stream was empty"

    if method == "mean":
        for k in grad_accum:
            grad_accum[k] /= n_batches

    if dist.is_initialized():
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    return grad_accum, float(loss_accum)


def prepare_trainer(
    cfg: "MagicConfig", rank: int, world_size: int, num_steps: int
):
    """Prepare the model, optimizer, and trainer for training."""
    torch.cuda.set_device(rank)

    model, target_modules = setup_model_and_peft(
        cfg,
        attn_implementation="eager",
    )
    model.to(f"cuda:{rank}")  # type: ignore[reportArgumentType]

    if target_modules:
        # We need to access the base model to set the loss function
        base = assert_type(PreTrainedModel, model.base_model)
        base.loss_function = weighted_causal_lm_ce

        # Only train the PEFT adapter parameters
        model.requires_grad_(False)
        for name in target_modules:
            module = model.get_submodule(name)
            module.requires_grad_(True)
    else:
        model.loss_function = weighted_causal_lm_ce
        model.requires_grad_(True)

    if cfg.training.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
        )

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

    train_cfg = cfg.training
    schedule = train_cfg.schedule(num_steps)
    opt = torchopt.adamw(
        schedule,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps_root=train_cfg.eps_root,
    )
    trainer, fwd_state = Trainer.initialize(model, opt)
    return trainer, fwd_state, model


def worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset: Dataset,
    query_dataset: Dataset,
    num_train_docs: int,
    num_query_docs: int,
    run_cfg: MagicConfig,
):
    train_cfg = run_cfg.training
    if train_cfg.num_epochs > 0:
        batches_per_epoch = len(train_dataset) // train_cfg.batch_size
        num_steps = train_cfg.num_epochs * batches_per_epoch
        wrap = True
    else:
        num_steps = train_cfg.num_steps
        wrap = False

    trainer, fwd_state, model = prepare_trainer(run_cfg, rank, world_size, num_steps)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    path0 = os.path.join(ckpts_path, "state0.pt")

    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.data.prompt_column,
        num_docs=num_train_docs,
        num_batches=num_steps,
        wrap=wrap,
    )

    if run_cfg.resume:
        # Replay forward from last valid checkpoint to regenerate deleted ones
        ckpt_list = sorted_checkpoints(ckpts_path)

        # Filter out incomplete checkpoints (missing .metadata from interrupted saves)
        # and clean them up from disk
        valid_ckpts = []
        for idx, path in ckpt_list:
            metadata = os.path.join(path, ".metadata")
            if os.path.exists(metadata):
                valid_ckpts.append((idx, path))
            else:
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        last_idx, last_path = valid_ckpts[-1]
        fwd_state.batch_index = last_idx
        fwd_state.load(last_path)
        fwd_state.detach_()

        chunk_size = math.isqrt(num_steps)
        last_start = num_steps - chunk_size

        main = not dist.is_initialized() or dist.get_rank() == 0
        pending_fut = None
        pbar = tqdm(
            range(last_idx, num_steps), desc="Replaying forward", disable=not main
        )
        for i in pbar:
            if i % chunk_size == 0 or i >= last_start:
                p = os.path.join(ckpts_path, f"step_{i}.ckpt")
                if not os.path.exists(p):
                    if pending_fut is not None:
                        pending_fut.result()
                    pending_fut = fwd_state.save(p)

            fwd_state = trainer.step(fwd_state, stream[i], inplace=True)

        if pending_fut is not None:
            pending_fut.result()
    else:
        save_fut = fwd_state.save(path0)
        fwd_state = trainer.train(
            fwd_state,
            stream,
            inplace=True,
            save_dir=ckpts_path,
        )
        save_fut.result()  # ensure state0 is saved before validation loads it

    # Compute query gradients
    query_stream = DataStream(
        query_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.query.prompt_column,
        num_docs=num_query_docs,
    )
    query_grads, baseline = compute_query_gradients(
        fwd_state, model, query_stream, run_cfg.query_method
    )

    stream.requires_grad = True
    opt_grads = [
        torch.zeros_like(buf)
        for buf in tree_iter(fwd_state.opt_state)
        if isinstance(buf, torch.Tensor) and buf.is_floating_point()
    ]
    bwd_state = BackwardState(query_grads, opt_grads, torch.zeros_like(stream.weights))

    bwd_state = trainer.backward(
        ckpts_path,
        stream,
        bwd_state,
        fwd_state,
        inplace=True,
    )
    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

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
    if run_cfg.resume:
        if not run_path.exists():
            raise FileNotFoundError(
                f"Run path {run_path} does not exist. Cannot resume."
            )
    else:
        if run_path.exists():
            if run_cfg.overwrite:
                shutil.rmtree(run_path)
            else:
                raise FileExistsError(
                    f"Run path {run_path} already exists. "
                    f"Use --overwrite to overwrite it."
                )

    run_path.mkdir(parents=True)
    run_cfg.save_json(run_path / "run_config.json")
    dist_cfg.save_json(run_path / "dist_config.json")

    train_ds, train_n = setup_data_pipeline(run_cfg)
    query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)

    launch_distributed_run(
        "run_magic",
        worker,
        [train_ds, query_ds, train_n, query_n, run_cfg],
        dist_cfg,
    )


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
