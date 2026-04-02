import csv
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset
from scipy.stats import describe, pearsonr, spearmanr
from simple_parsing import ArgumentParser, field
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from tqdm import tqdm

from ..config import AttributionConfig, DataConfig, TrainingConfig
from ..distributed import grad_tree, launch_distributed_run, simple_fsdp
from ..utils.logging import wandb_log_fn
from ..utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .trainer import BackwardState, Trainer, TrainerState


@dataclass
class MagicConfig(AttributionConfig, TrainingConfig):
    """Special config for MAGIC attribution."""

    query: DataConfig = field(
        default_factory=lambda: DataConfig(split="train"),
    )
    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    query_method: Literal["mean", "sum"] = "mean"
    """Method for reducing query gradients across batches."""

    save_mode: Literal["all", "sqrt"] = "sqrt"
    """Checkpoint saving mode. 'all' saves every checkpoint, 'sqrt' saves every
    sqrt(N) steps, and rematerializes checkpoints when needed."""

    num_subsets: int = 100
    """Number of leave-one-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""

    wandb_project: str = ""
    """Weights & Biases project name. If set, logs training loss to W&B."""

    resume: bool = False
    """Resume a previously interrupted run from the last checkpoint."""

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    def __post_init__(self):
        assert not self.fsdp, "PyTorch FSDP is not currently supported for MAGIC."


def compute_query_gradients(
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_stream: DataStream,
    method: str = "mean",
    fsdp: bool = False,
) -> tuple[dict[str, torch.Tensor], float]:
    """Compute reduced query gradients over the query dataset.

    Iterates over the query stream, computing per-batch parameter gradients
    and reducing them (mean or sum) into a single gradient dict.
    """
    grad_accum: dict[str, torch.Tensor] | None = None
    loss_accum = 0.0
    n_batches = len(query_stream)

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

            loss_accum += loss.detach()

    assert grad_accum is not None, "Query stream was empty"

    if method == "mean":
        for k in grad_accum:
            grad_accum[k] /= n_batches

        loss_accum /= n_batches

    if dist.is_initialized():
        if not fsdp:
            for k in grad_accum:
                differentiable_all_reduce(grad_accum[k], op=dist.ReduceOp.SUM)
        # Loss is never a DTensor
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)

    return grad_accum, float(loss_accum)


def get_schedule(lr_cfg, num_steps: int):
    """Return a learning rate schedule function: step → lr.

    Supports HF-compatible scheduler types and an optional non-zero warmup
    start (``lr_start``).
    """
    if lr_cfg.warmup_steps >= 1:
        warmup_steps = int(lr_cfg.warmup_steps)
    else:
        warmup_steps = math.ceil(num_steps * lr_cfg.warmup_steps)

    lr = lr_cfg.lr
    lr_start = lr_cfg.lr_start
    decay_steps = max(num_steps - warmup_steps, 1)

    def _warmup(step):
        """Linear warmup from lr_start to lr."""
        progress = step / max(warmup_steps, 1)
        return lr_start + (lr - lr_start) * progress

    scheduler_type = lr_cfg.lr_scheduler_type

    if scheduler_type == "constant":

        def _schedule(step):
            return lr

    elif scheduler_type == "constant_with_warmup":

        def _schedule(step):
            if step < warmup_steps:
                return _warmup(step)
            return lr

    elif scheduler_type == "linear":

        def _schedule(step):
            if step < warmup_steps:
                return _warmup(step)
            progress = (step - warmup_steps) / decay_steps
            return lr * (1 - progress)

    elif scheduler_type == "cosine":

        def _schedule(step):
            if step < warmup_steps:
                return _warmup(step)
            progress = (step - warmup_steps) / decay_steps
            return (
                lr * 0.5 * (1 + math.cos(math.pi * lr_cfg.num_cycles * 2.0 * progress))
            )

    elif scheduler_type == "cosine_with_restarts":

        def _schedule(step):
            if step < warmup_steps:
                return _warmup(step)
            progress = (step - warmup_steps) / decay_steps
            return (
                lr
                * 0.5
                * (1 + math.cos(math.pi * ((lr_cfg.num_cycles * progress) % 1.0) * 2.0))
            )

    elif scheduler_type == "polynomial":

        def _schedule(step):
            if step < warmup_steps:
                return _warmup(step)
            progress = (step - warmup_steps) / decay_steps
            return lr_cfg.lr_end + (lr - lr_cfg.lr_end) * (1 - progress) ** lr_cfg.power

    else:
        raise ValueError(f"Unknown lr_scheduler_type: {scheduler_type!r}")

    return _schedule


class CSVWriter:
    """CSV writer that no-ops when disabled."""

    def __init__(self, path: str, columns: list[str], enabled: bool = True):
        self.path = path
        if enabled:
            self._file = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(columns)
        else:
            self._file = None
            self._writer = None

    def writerow(self, *args):
        if self._writer is None or self._file is None:
            return
        self._writer.writerow([*args])
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()


def prepare_trainer(
    cfg: TrainingConfig,
    rank: int,
    schedule: Callable,
):
    """Prepare the model, optimizer, and trainer for training."""
    model, target_modules = setup_model_and_peft(
        cfg,
        attn_implementation="eager",
        apply_fsdp=False,
    )
    model.to(f"cuda:{rank}")  # type: ignore[reportArgumentType]

    if target_modules:
        # Only train the PEFT adapter parameters
        model.requires_grad_(False)
        for name in target_modules:
            module = model.get_submodule(name)
            module.requires_grad_(True)
    else:
        model.requires_grad_(True)

    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(  # type: ignore[attr-defined]
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
        )

    if cfg.fsdp and dist.is_initialized():
        apply_dtensor_patch()
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        with mesh:
            model = simple_fsdp(model)

    opt = torchopt.adamw(
        schedule,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps_root=cfg.eps_root,
    )
    trainer, fwd_state = Trainer.initialize(model, opt)
    return trainer, fwd_state, model


def pad_dataset_to_batch_size(
    dataset: Dataset,
    batch_size: int,
    num_docs: int,
    label: str,
    global_rank: int,
) -> tuple[Dataset, int, int]:
    """Pad dataset to be divisible by batch_size by repeating the last example.

    Returns (padded_dataset, num_docs, pad_count). num_docs is updated only when
    the dataset has no "doc_ids" column (i.e. each row is its own document).
    pad_count is 0 if no padding was needed.
    """
    remainder = len(dataset) % batch_size
    if not remainder:
        return dataset, num_docs, 0

    pad_count = batch_size - remainder
    total = len(dataset)
    pad_indices = list(range(total)) + [total - 1] * pad_count
    dataset = dataset.select(pad_indices)
    if "doc_ids" not in dataset.column_names:
        num_docs = len(dataset)
    if global_rank == 0:
        print(
            f"{label}: padded {pad_count}/{total} examples "
            f"(weight=0) to fill last batch"
        )
    return dataset, num_docs, pad_count


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
    torch.cuda.set_device(rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            world_size=world_size,
        )

    if run_cfg.num_epochs > 1:
        train_dataset = train_dataset.repeat(run_cfg.num_epochs)

    # Ensure total effective batch size is divisible by world size
    assert run_cfg.batch_size % world_size == 0

    # Pad train dataset to be divisible by batch_size (weight=0 for padding)
    train_dataset, num_train_docs, pad_count = pad_dataset_to_batch_size(
        train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
    )

    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.data.prompt_column,
        num_docs=num_train_docs,
    )
    if pad_count:
        stream.weights.data[-pad_count:] = 0.0

    log_fn = None
    if run_cfg.wandb_project and global_rank == 0:
        log_fn = wandb_log_fn(run_cfg.wandb_project, config=asdict(run_cfg))

    schedule = get_schedule(run_cfg.lr_schedule, len(stream))
    trainer, fwd_state, model = prepare_trainer(
        run_cfg,
        rank,
        schedule,
    )

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    path0 = os.path.join(ckpts_path, "state0.pt")

    resume = run_cfg.resume and os.path.exists(path0)

    save_fut = None
    if not resume:
        save_fut = fwd_state.save(path0)

    fwd_state = trainer.train(
        fwd_state,
        stream,
        debug=run_cfg.debug,
        inplace=True,
        save_dir=ckpts_path,
        save_mode=run_cfg.save_mode,
        log_fn=log_fn,
        resume=resume,
        fsdp=run_cfg.fsdp,
    )

    if save_fut is not None:
        save_fut.result()  # ensure state0 is saved before validation loads it

    # Pad query dataset to be divisible by batch_size (weight=0 for padding)
    query_dataset, num_query_docs, query_pad_count = pad_dataset_to_batch_size(
        query_dataset, run_cfg.batch_size, num_query_docs, "Query", global_rank
    )
    if len(query_dataset) < run_cfg.batch_size:
        raise ValueError(
            f"Query dataset has {len(query_dataset)} examples, fewer than "
            f"batch_size={run_cfg.batch_size}. Use a larger query split or "
            f"smaller batch_size."
        )

    # Compute query gradients
    query_stream = DataStream(
        query_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.query.prompt_column,
        num_docs=num_query_docs,
    )
    if query_pad_count:
        query_stream.weights.data[-query_pad_count:] = 0.0

    query_grads, baseline = compute_query_gradients(
        fwd_state, model, query_stream, run_cfg.query_method, run_cfg.fsdp
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
        debug=run_cfg.debug,
        inplace=True,
        fsdp=run_cfg.fsdp,
        resume=run_cfg.resume,
        save_every=run_cfg.backward_save_every,
    )
    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

    scores = bwd_state.weight_grads.cpu()
    if pad_count:
        scores = scores[:-pad_count]
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
    num_real = len(stream.weights) - pad_count
    perm = torch.randperm(num_real, generator=gen)
    subsets = perm.chunk(run_cfg.num_subsets)

    csv_path = os.path.join(run_cfg.run_path, "validation.csv")
    val_csv_writer = CSVWriter(
        csv_path,
        columns=["subset", "diff", "score_sum"],
        enabled=global_rank == 0,
    )

    pbar = tqdm(subsets, desc="Validating", disable=global_rank != 0)
    for i, subset in enumerate(pbar):
        fwd_state.load(path0)
        fwd_state.detach_()

        stream.weights.fill_(1.0)
        if pad_count:
            stream.weights.data[-pad_count:] = 0.0
        stream.weights[subset] = 0.0

        for x in stream:
            fwd_state = trainer.step(fwd_state, x, inplace=True, fsdp=run_cfg.fsdp)

        with fwd_state.activate(model), torch.no_grad():
            loss = torch.tensor(0.0, device=stream.weights.device)
            for batch in query_stream:
                del batch["example_weight"]

                loss += model(**batch).loss.detach() / len(query_stream)

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        diff = baseline - loss.item()
        score_sum = scores[subset].sum().item()
        val_csv_writer.writerow(i, diff, score_sum)

        if global_rank == 0:
            diffs.append(diff)
            score_sums.append(score_sum)

            if len(diffs) >= 2:
                sp = spearmanr(diffs, score_sums)
                pe = pearsonr(diffs, score_sums)
                pbar.set_postfix({"rho": sp.statistic, "r": pe.statistic})
            else:
                pbar.set_postfix({"rho": "n/a", "r": "n/a"})

    val_csv_writer.close()
    if global_rank == 0:
        sp = spearmanr(diffs, score_sums)
        pe = pearsonr(diffs, score_sums)
        print(f"Final Spearman correlation: {sp.statistic:.4f} (p={sp.pvalue:.2e})")
        print(f"Final Pearson correlation:  {pe.statistic:.4f} (p={pe.pvalue:.2e})")
        print(f"Saved validation data to {csv_path}")

        summary_csv_writer = CSVWriter(
            os.path.join(run_cfg.run_path, "summary.csv"),
            columns=[
                "spearman_corr",
                "spearman_p",
                "pearson_corr",
                "pearson_p",
                "N",
                "baseline_loss",
            ],
        )
        summary_csv_writer.writerow(
            sp.statistic, sp.pvalue, pe.statistic, pe.pvalue, len(subsets), baseline
        )


def run_magic(run_cfg: MagicConfig):
    run_path = Path(run_cfg.run_path)
    is_main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    multi_node = run_cfg.distributed.nnode > 1

    if is_main_node:
        if run_path.exists() and not run_cfg.resume:
            if run_cfg.overwrite:
                shutil.rmtree(run_path)
            else:
                raise FileExistsError(
                    f"Run path {run_path} already exists. "
                    f"Use --overwrite to overwrite it."
                )

        run_path.mkdir(parents=True, exist_ok=True)
        run_cfg.save_yaml(run_path / "run_config.yaml")

    # HF datasets caches are not safe for concurrent writers, so the main node
    # must finish populating the cache before others read from it.
    barrier = run_path / ".preprocess_done" if multi_node else None
    if barrier is not None and not is_main_node:
        run_path.mkdir(parents=True, exist_ok=True)
        while not barrier.exists():
            time.sleep(0.5)

    train_ds, train_n = setup_data_pipeline(run_cfg)

    # Shuffle the train_ds with the seed.
    train_ds = train_ds.shuffle(seed=run_cfg.seed)

    query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)

    if barrier is not None and is_main_node:
        barrier.touch()

    launch_distributed_run(
        "run_magic",
        worker,
        [train_ds, query_ds, train_n, query_n, run_cfg],
        run_cfg.distributed,
    )


def main():
    parser = ArgumentParser()
    parser.add_arguments(MagicConfig, dest="run_cfg")
    args = parser.parse_args()

    run_cfg: MagicConfig = args.run_cfg
    run_magic(run_cfg)


if __name__ == "__main__":
    main()
