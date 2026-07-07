import csv
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset
from scipy.stats import describe, pearsonr, spearmanr
from simple_parsing import ArgumentParser
from torch.distributed._functional_collectives import (
    all_reduce as differentiable_all_reduce,
)
from torch.distributed._functional_collectives import (
    wait_tensor,
)
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from tqdm import tqdm
from transformers.utils.logging import (
    disable_progress_bar as hf_disable_pbar,
)
from transformers.utils.logging import (
    set_verbosity_error as hf_set_verbosity_error,
)

from ..config.config import ScoreConfig, TrainingConfig, ValidationConfig
from ..config.config_io import load_subconfig, save_run_config
from ..data import load_scores
from ..distributed import grad_tree, launch_distributed_run
from ..utils.logging import wandb_log_fn
from ..utils.utils import get_device, get_device_index
from ..utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)
from .config import MagicConfig
from .data_stream import DataStream
from .dtensor_patch import apply_dtensor_patch
from .fsdp import simple_fsdp
from .optim import muon
from .trainer import BackwardState, Trainer, TrainerState


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
    denom = len(query_stream)
    grad_accum: dict[str, torch.Tensor] | None = None
    loss_accum = 0.0

    if dist.is_initialized():
        denom *= dist.get_world_size()
        main = dist.get_rank() == 0
    else:
        main = True

    with fwd_state.activate(model) as params:
        for batch in tqdm(query_stream, desc="Query", disable=not main):
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
            grad_accum[k] /= denom

        loss_accum /= denom

    if dist.is_initialized():
        if not fsdp:
            grad_accum = {
                k: wait_tensor(
                    differentiable_all_reduce(
                        g,
                        "sum",
                        dist.distributed_c10d._get_default_group(),
                    )
                )
                for k, g in grad_accum.items()
            }

        # Loss is never a DTensor
        dist.all_reduce(loss_accum)

    return grad_accum, float(loss_accum)


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


def prepare_trainer(cfg: TrainingConfig, rank: int, schedule: Callable):
    """Prepare the model, optimizer, and trainer for training."""
    model, target_modules = setup_model_and_peft(
        cfg,
        attn_implementation="eager",
        apply_fsdp=False,
    )
    model.to(get_device(rank))  # type: ignore[reportArgumentType]

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

    match cfg.optimizer:
        case "adamw":
            opt = torchopt.adamw(
                schedule,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps_root=cfg.eps_root,
                weight_decay=cfg.weight_decay,
            )
        case "muon":
            opt = muon(
                schedule,
                momentum=cfg.adam_beta1,
                adamw_betas=(cfg.adam_beta1, cfg.adam_beta2),
                adamw_eps_root=cfg.eps_root,
                weight_decay=cfg.weight_decay,
            )
        case "sgd":
            opt = torchopt.sgd(
                schedule,
                momentum=cfg.adam_beta1,
                weight_decay=cfg.weight_decay,
            )
        case other:
            raise ValueError(f"Unsupported optimizer: {other}")

    trainer, fwd_state = Trainer.initialize(model, opt)
    return trainer, fwd_state, model


def attach_doc_ids_if_missing(dataset: Dataset) -> Dataset:
    """Ensure the dataset has a ``doc_ids`` column.

    ``doc_ids`` is a per-row list of length ``max_seq_len`` giving the
    document id of every token position in that row. Chunked/packed
    datasets already have it (multiple docs may share a chunk). For
    one-doc-per-row datasets the column is synthesized as
    ``[row_index] * max_seq_len`` so the two cases look identical to
    downstream code (DataStream indexing, per-doc aggregation via
    ``scatter_add(doc_ids, scores)``).

    No-op if ``doc_ids`` is already present.
    """
    if "doc_ids" in dataset.column_names:
        return dataset
    if "length" in dataset.column_names:
        seq_len = max(dataset["length"])
    else:
        seq_len = max(len(row) for row in dataset["input_ids"])
    return dataset.map(
        lambda _, idx: {"doc_ids": [idx] * seq_len},
        with_indices=True,
        desc="Attaching doc_ids",
    )


def pad_dataset_to_batch_size(
    dataset: Dataset,
    batch_size: int,
    num_docs: int,
    label: str,
    global_rank: int,
) -> tuple[Dataset, int, int, int]:
    """Pad dataset to be divisible by batch_size by repeating the last example.

    Returns (padded_dataset, num_docs, pad_count, weight_pad_count).

    `pad_count` is the number of rows appended to the dataset (0 if unchanged).
    `weight_pad_count` is the number of trailing entries of a *1D* per-doc
    weight tensor that should be zeroed to silence the pad rows' training
    contribution.

    - If the dataset has a "doc_ids" column, `.select(total - 1, ...)` copies
      the last doc's doc_ids into every pad row. Zeroing the last `pad_count`
      entries of a weights-indexed-by-doc_id tensor would silence real docs,
      so we instead route pad rows to a fresh synthetic doc id (=num_docs),
      bump num_docs by 1, and set `weight_pad_count = 1`.
    - Otherwise rows are self-identified docs: num_docs becomes the padded
      length and `weight_pad_count = pad_count` zeros the pad rows directly.

    In per-token (2D) mode callers should zero `weights[-pad_count:]` instead
    — `weight_pad_count` applies only to 1D per-doc weights.
    """
    remainder = len(dataset) % batch_size
    if not remainder:
        return dataset, num_docs, 0, 0

    pad_count = batch_size - remainder
    total = len(dataset)
    pad_indices = list(range(total)) + [total - 1] * pad_count
    dataset = dataset.select(pad_indices)

    if "doc_ids" in dataset.column_names:
        synthetic_doc_id = num_docs
        new_doc_ids = [
            row if i < total else [synthetic_doc_id] * len(row)
            for i, row in enumerate(dataset["doc_ids"])
        ]
        dataset = dataset.remove_columns("doc_ids").add_column("doc_ids", new_doc_ids)
        num_docs += 1
        weight_pad_count = 1
    else:
        num_docs = len(dataset)
        weight_pad_count = pad_count

    if global_rank == 0:
        print(
            f"{label}: padded {pad_count}/{total} examples "
            f"(weight=0) to fill last batch"
        )
    return dataset, num_docs, pad_count, weight_pad_count


def worker(
    global_rank: int,  # global
    rank: int,  # local
    world_size: int,
    train_dataset: Dataset,
    query_dataset: Dataset | None,
    num_train_docs: int,
    num_query_docs: int,
    run_cfg: TrainingConfig,
    score_path: str = "",
):
    torch.cuda.set_device(get_device_index(rank))

    # For each non-main local rank, suppress HF info and warning messages
    if rank != 0:
        hf_disable_pbar()
        hf_set_verbosity_error()

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(rank)),
            rank=global_rank,
            world_size=world_size,
        )

    if run_cfg.num_epochs > 1:
        train_dataset = train_dataset.repeat(run_cfg.num_epochs)

    # Ensure total effective batch size is divisible by world size
    assert run_cfg.batch_size % world_size == 0

    # Pad train dataset to be divisible by batch_size (weight=0 for padding)
    train_dataset, num_train_docs, pad_count, weight_pad_count = (
        pad_dataset_to_batch_size(
            train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
        )
    )

    if getattr(run_cfg, "per_token", False):
        seq_len = run_cfg.data.chunk_length
        if seq_len <= 0:
            seq_len = max(train_dataset["length"])
            print(f"Using max sequence length {seq_len} for per-token attribution")

        w_shape = (len(train_dataset), seq_len)
    else:
        w_shape = (num_train_docs,)

    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=get_device(rank),
        input_key=run_cfg.data.prompt_column,
        weight_shape=w_shape,
    )
    if pad_count:
        if stream.weights.ndim == 1:
            stream.weights.data[-weight_pad_count:] = 0.0
        else:
            stream.weights.data[-pad_count:] = 0.0

    log_fn = None
    if run_cfg.wandb_project and global_rank == 0:
        log_fn = wandb_log_fn(run_cfg.wandb_project, config=asdict(run_cfg))

    if dist.is_initialized():
        dist.barrier()

    schedule = run_cfg.lr_schedule.get_schedule(len(stream))
    trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    resume = run_cfg.resume

    fwd_state = trainer.train(
        fwd_state,
        stream,
        debug=run_cfg.debug,
        inplace=True,
        save_dir=ckpts_path,
        save_mode=getattr(run_cfg, "save_mode", "sqrt"),
        log_fn=log_fn,
        resume=resume,
        fsdp=run_cfg.fsdp,
    )

    # If no query dataset is provided, skip backward and validation entirely
    if query_dataset is None:
        return
    elif not isinstance(run_cfg, ValidationConfig):
        raise RuntimeError(
            "run_cfg must be a ValidationConfig if query_dataset is provided"
        )

    # Pad query dataset to be divisible by batch_size (weight=0 for padding)
    query_dataset, num_query_docs, query_pad_count, query_weight_pad_count = (
        pad_dataset_to_batch_size(
            query_dataset, run_cfg.batch_size, num_query_docs, "Query", global_rank
        )
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
        device=get_device(rank),
        input_key=run_cfg.query.prompt_column,
        weight_shape=(num_query_docs,),
    )
    if query_pad_count:
        # query_stream.weights is always 1D (weight_shape=(num_query_docs,))
        query_stream.weights.data[-query_weight_pad_count:] = 0.0

    query_grads, baseline = compute_query_gradients(
        fwd_state, model, query_stream, run_cfg.query_method, run_cfg.fsdp
    )

    if not score_path:
        # Sanity check
        if not isinstance(run_cfg, MagicConfig):
            raise RuntimeError("run_cfg must be a MagicConfig to compute scores")

        stream.requires_grad = True
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(fwd_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        bwd_state = BackwardState(
            query_grads,
            opt_grads,
            torch.zeros_like(stream.weights),
        )

        bwd_state = trainer.backward(
            ckpts_path,
            stream,
            bwd_state,
            fwd_state,
            cleanup=run_cfg.cleanup_ckpts,
            debug=run_cfg.debug,
            inplace=True,
            fsdp=run_cfg.fsdp,
            resume=run_cfg.resume,
            save_every=run_cfg.backward_save_every,
            save_mode=run_cfg.save_mode,
        )
        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

        scores = bwd_state.weight_grads.cpu()
        if pad_count:
            if scores.ndim == 1:
                scores = scores[:-weight_pad_count]
            else:
                scores = scores[:-pad_count]

        if global_rank == 0:
            print(f"Baseline loss: {baseline}")

            summ = describe(scores.flatten())
            print(f"Score summary: {summ}")

            score_path = os.path.join(run_cfg.run_path, "scores.pt")
            torch.save(scores, score_path)
            print(f"Saved attribution scores to {score_path}")
    elif os.path.isdir(score_path):
        scores = torch.from_numpy(load_scores(Path(score_path))[:])
        score_cfg = load_subconfig(score_path, "score_cfg", ScoreConfig)
        if score_cfg is not None and score_cfg.higher_is_better:
            scores = -scores
    else:
        scores = torch.load(score_path, map_location="cpu")

        # Per-token scores are indexed by (shuffled_chunk_idx, token_idx).
        # Save doc_ids alongside so downstream can aggregate per-doc with
        # one scatter_add and no reference to the raw dataset or seed.
        if scores.ndim == 2:
            doc_ids = torch.tensor(train_dataset["doc_ids"])
            if pad_count:
                doc_ids = doc_ids[:-pad_count]
            doc_ids_path = os.path.join(run_cfg.run_path, "doc_ids.pt")
            torch.save(doc_ids, doc_ids_path)
            print(f"Saved doc_ids to {doc_ids_path}")

    stream.requires_grad = False

    # Validate attribution scores via leave-subset-out retraining
    diffs = []
    score_sums = []

    assert scores.ndim == 1 or scores.shape[1] == 1
    scores = scores.flatten()

    if run_cfg.exclude_zero_scores:
        valid_indices = torch.nonzero(scores != 0, as_tuple=True)[0]
    else:
        valid_indices = torch.arange(len(scores))

    if run_cfg.subset_strategy == "random":
        rng = torch.Generator().manual_seed(run_cfg.seed)
        perm = valid_indices[torch.randperm(len(valid_indices), generator=rng)]
    elif run_cfg.subset_strategy == "sorted":

        perm = valid_indices[scores[valid_indices].argsort()]
    else:
        raise ValueError(f"Unsupported subset strategy: {run_cfg.subset_strategy}")

    # Shuffle the order of the subsets so that the estimate of correlation on the
    # progress bar is unbiased. This does not change the final correlation since all
    # subsets are eventually evaluated, but prevents the early subsets from being
    # biased towards higher or lower scores.
    subsets = list(perm.chunk(run_cfg.num_subsets))
    rng = random.Random(run_cfg.seed)
    rng.shuffle(subsets)

    csv_path = os.path.join(run_cfg.run_path, "validation.csv")
    val_csv_writer = CSVWriter(
        csv_path,
        columns=["subset", "diff", "score_sum"],
        enabled=global_rank == 0,
    )

    # Disable annoying repetitive model loading messages, even on rank 0
    hf_disable_pbar()
    hf_set_verbosity_error()

    pbar = tqdm(subsets, desc="Validating", disable=global_rank != 0)
    for i, subset in enumerate(pbar):
        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)
        fwd_state.detach_()

        stream.weights.fill_(1.0)
        if pad_count:
            if stream.weights.ndim == 1:
                stream.weights.data[-weight_pad_count:] = 0.0
            else:
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


def prepare_run_dir(run_cfg: TrainingConfig) -> None:
    """Validate/clear the run path and persist the run config for reproducibility.

    The MAGIC analogue of ``validate_run_path`` + ``save_run_config`` used by the
    index commands (sharing ``clear_run_path`` for the overwrite policy): clear or
    keep the run path (resume-aware), then write ``config.yaml`` before any
    compute starts so the run can be replayed with
    ``bergson <run_path>/config.yaml``. A no-op off the main node
    (``SLURM_PROCID != 0``) so only one writer touches the shared run directory.
    """
    if int(os.environ.get("SLURM_PROCID", 0)) != 0:
        return

    run_path = Path(run_cfg.run_path)
    if not run_cfg.resume:
        clear_run_path(run_path, run_cfg.overwrite)

    run_path.mkdir(parents=True, exist_ok=True)
    save_run_config(run_cfg, run_path)


def run_magic(run_cfg: TrainingConfig, *, score_path: str = ""):
    run_path = Path(run_cfg.run_path)
    is_main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    multi_node = run_cfg.distributed.nnode > 1

    # HF datasets caches are not safe for concurrent writers, so the main node
    # must finish populating the cache before others read from it.
    barrier = run_path / ".preprocess_done" if multi_node else None
    if barrier is not None and not is_main_node:
        run_path.mkdir(parents=True, exist_ok=True)
        while not barrier.exists():
            time.sleep(0.5)

    train_ds, train_n = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)

    # Shuffle the train_ds with the seed.
    train_ds = train_ds.shuffle(seed=run_cfg.seed)

    if isinstance(run_cfg, ValidationConfig):
        query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)
    else:
        query_ds, query_n = None, 0

    if barrier is not None and is_main_node:
        barrier.touch()

    launch_distributed_run(
        "run_magic",
        worker,
        [train_ds, query_ds, train_n, query_n, run_cfg, score_path],
        run_cfg.distributed,
    )


def main():
    parser = ArgumentParser()
    parser.add_arguments(MagicConfig, dest="run_cfg")
    args = parser.parse_args()

    run_cfg: MagicConfig = args.run_cfg
    prepare_run_dir(run_cfg)
    run_magic(run_cfg)


if __name__ == "__main__":
    main()
