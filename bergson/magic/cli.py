import gc
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from scipy.stats import describe
from simple_parsing import ArgumentParser
from torch.distributed._functional_collectives import (
    all_reduce as differentiable_all_reduce,
)
from torch.distributed._functional_collectives import (
    wait_tensor,
)
from torchopt.pytree import tree_iter
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils.logging import (
    disable_progress_bar as hf_disable_pbar,
)
from transformers.utils.logging import (
    set_verbosity_error as hf_set_verbosity_error,
)

from ..config.config import TrainingConfig, ValidationConfig
from ..config.config_io import save_run_config
from ..data import compute_num_token_grads, load_scores_loss_signed
from ..distributed import launch_distributed_run
from ..score.score_writer import save_sequence_scores, save_token_scores
from ..utils.load_from_optimizer import (
    save_second_moments_as_optimizer_pt,
)
from ..utils.logging import wandb_log_fn
from ..utils.utils import (
    current_device,
    dist_backend,
    dist_device_id,
    get_device,
    get_device_index,
)
from ..utils.worker_utils import setup_data_pipeline
from ..validate import validate_scores
from .config import MagicConfig
from .data_stream import DataStream, mask_padded_rows, pad_dataset_to_batch_size
from .grad_accum import accumulate_grads
from .score_trajectory import has_matplotlib, plot_score_trajectory
from .trainer import BackwardState, TrainerState, prepare_trainer, write_lr_history


def compute_query_gradients(
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_stream: DataStream,
    method: str = "mean",
    fsdp: bool = False,
    grad_accum_steps: int = 1,
) -> tuple[dict[str, torch.Tensor], float]:
    """Compute reduced query gradients over the query dataset.

    Iterates over the query stream, computing per-batch parameter gradients
    and reducing them (mean or sum) into a single gradient dict.
    """
    grad_accum: dict[str, torch.Tensor] | None = None
    loss_accum = 0.0
    denom = 0

    main = not dist.is_initialized() or dist.get_rank() == 0

    with fwd_state.activate(model) as params:
        for batch in tqdm(query_stream, desc="Query", disable=not main):
            batch, n_tokens = mask_padded_rows(batch)
            denom += n_tokens
            grads, loss = accumulate_grads(
                model,
                params,
                batch,
                grad_accum_steps,
                create_graph=False,
                loss_scale=n_tokens,
            )

            if grad_accum is None:
                grad_accum = {k: g.detach().clone() for k, g in grads.items()}
            else:
                for k, g in grads.items():
                    grad_accum[k] += g.detach()

            loss_accum += loss * n_tokens

    assert grad_accum is not None, "Query stream was empty"

    if dist.is_initialized():
        denom_t = torch.tensor(denom, device=current_device())
        dist.all_reduce(denom_t)
        denom = int(denom_t.item())

    if method != "sum":
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
        loss_tensor = torch.tensor(loss_accum, device=current_device())
        dist.all_reduce(loss_tensor)
        loss_accum = loss_tensor.item()

    return grad_accum, float(loss_accum)


def compute_per_query_magic_scores(
    trainer,
    ckpts_path: str,
    stream: DataStream,
    fwd_state: TrainerState,
    model: torch.nn.Module,
    query_dataset: Dataset,
    num_query_docs: int,
    run_cfg: "MagicConfig",
    world_size: int,
    global_rank: int,
    pad_count: int,
    weight_pad_count: int,
) -> torch.Tensor:
    """Per-query MAGIC scores: one backward per query, sharing the forward.

    ``run_magic`` reduces the queries to one gradient before the backward, so it
    yields a single aggregate-query score. This scores each query separately:
    for each query document it takes that document's gradient at the final model
    as the backward cotangent and runs ``Trainer.backward`` over the saved
    trajectory, producing ``[num_train_docs, num_query_docs]`` — or
    ``[num_train_docs, seq_len, num_query_docs]`` when attributing tokens.

    The backward is linear in the cotangent, so this is exact; the forward runs
    once and every query reuses its checkpoints (``cleanup=False``). Per-query
    scores are written incrementally to ``<run_path>/per_query/q{i}.pt`` so a
    crash or preemption only loses the in-flight query (resume redoes the
    forward but skips finished queries), and the final state is restored before
    each query since the backward walks it back down the trajectory.
    """
    main = global_rank == 0
    device = stream.weights.device
    scores_dir = os.path.join(run_cfg.run_path, "per_query")
    if main:
        os.makedirs(scores_dir, exist_ok=True)

    # Snapshot the final state (CPU) to restore before each query's backward, and
    # the forward's checkpoint files so per-query temp checkpoints can be cleaned.
    final_state = fwd_state.to("cpu").detach_()
    orig_ckpts = set(os.listdir(ckpts_path)) if os.path.isdir(ckpts_path) else set()
    opt_grads_zero = [
        torch.zeros_like(buf)
        for buf in tree_iter(fwd_state.opt_state)
        if isinstance(buf, torch.Tensor) and buf.is_floating_point()
    ]

    per_query = []
    for qi in range(num_query_docs):
        qpath = os.path.join(scores_dir, f"q{qi}.pt")
        if os.path.exists(qpath):  # resume: already scored
            per_query.append(torch.load(qpath, map_location="cpu"))
            continue

        # Restore the final trained state (the backward walks it back down the
        # trajectory). detach_ first: the previous iteration left params
        # requiring grad, and copy_ is an in-place write a leaf-requiring-grad
        # forbids. Free the GPU copy so states don't accumulate across queries.
        fwd_state.detach_()
        restored = final_state.to(device)
        fwd_state.copy_(restored)
        del restored

        # A one-document stream indexes its weights by row, not by doc id.
        # One row per rank; more rows are zero-weight padding.
        one = query_dataset.select([qi])
        if "doc_ids" in one.column_names:
            one = one.remove_columns("doc_ids")
        one, n_one, one_pad, one_wpad = pad_dataset_to_batch_size(
            one, world_size, 1, f"Query {qi}", global_rank
        )
        qstream = DataStream(
            one,
            world_size,
            device=device,
            input_key=run_cfg.query.prompt_column,
            weight_shape=(n_one,),
        )
        if one_pad:
            qstream.weights.data[-one_wpad:] = 0.0
        qgrads, _ = compute_query_gradients(
            fwd_state, model, qstream, "mean", run_cfg.fsdp, run_cfg.grad_accum_steps
        )

        fwd_state.detach_()  # clear requires_grad set by the activation above
        stream.requires_grad = True
        stream.weights.grad = None
        bwd_state = BackwardState(
            qgrads,
            [g.clone() for g in opt_grads_zero],
            torch.zeros_like(stream.weights),
        )
        bwd_state = trainer.backward(
            ckpts_path,
            stream,
            bwd_state,
            fwd_state,
            cleanup=False,  # reuse the forward's checkpoints for every query
            debug=run_cfg.debug,
            inplace=True,
            fsdp=run_cfg.fsdp,
            save_mode=run_cfg.save_mode,
            max_grad_norm=run_cfg.max_grad_norm,
            grad_accum_steps=run_cfg.grad_accum_steps,
            double_backward_batch_size=run_cfg.double_backward_batch_size,
        )
        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

        s = bwd_state.weight_grads.detach().cpu()
        if pad_count:
            s = s[:-weight_pad_count] if s.ndim == 1 else s[:-pad_count]
        if main:
            torch.save(s, qpath)
        per_query.append(s)

        # Free per-query state and any temp checkpoints the backward wrote.
        del bwd_state, qgrads, qstream, one
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if main and os.path.isdir(ckpts_path):
            for name in set(os.listdir(ckpts_path)) - orig_ckpts:
                fp = os.path.join(ckpts_path, name)
                shutil.rmtree(fp) if os.path.isdir(fp) else os.remove(fp)
        if main:
            print(f"[per-query MAGIC] scored query {qi + 1}/{num_query_docs}")

    # The last backward walked fwd_state down the trajectory; validate_scores
    # needs the trained state for the multi-query baseline.
    fwd_state.detach_()
    restored = final_state.to(device)
    fwd_state.copy_(restored)
    del restored

    return torch.stack(per_query, dim=-1)


def scores_are_per_token(score_path: str) -> bool:
    info_path = os.path.join(score_path, "info.json")
    if not os.path.isfile(info_path):
        return False
    with open(info_path) as f:
        return bool(json.load(f).get("attribute_tokens", False))


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


def emit_trajectory_plot(run_path: str, scores: torch.Tensor, batch_size: int):
    """Plot the per-step score level beside the scores, if matplotlib is present."""
    out = str(Path(run_path) / "score_vs_step.png")
    try:
        if plot_score_trajectory(scores.float().numpy(), batch_size, out):
            print(f"Saved score trajectory plot to {out}")
    except Exception as e:
        # The scores are already saved; a plotting failure must not end the run.
        print(f"Could not plot the score trajectory: {e}")


def save_magic_scores(
    run_path: str,
    scores: torch.Tensor,
    train_dataset: Dataset,
    pad_count: int,
    per_token: bool,
    batch_size: int,
) -> str:
    """Write MAGIC scores as a score directory under ``<run_path>/scores``.

    Columns past a row's ``length - 1`` never receive gradient, so the dense
    per-token grid packs into the ragged store without loss.
    """
    path = Path(run_path) / "scores"

    if not per_token:
        arr = scores.float().numpy()
        save_sequence_scores(path, arr if arr.ndim > 1 else arr[:, None])
        print(f"Saved attribution scores to {path}")
        emit_trajectory_plot(run_path, scores, batch_size)
        return str(path)

    num_token_grads = compute_num_token_grads(train_dataset)
    doc_ids = np.asarray(train_dataset["doc_ids"], dtype=np.int64)
    if pad_count:
        num_token_grads = num_token_grads[:-pad_count]
        doc_ids = doc_ids[:-pad_count]

    offsets = np.zeros(len(num_token_grads) + 1, dtype=np.int64)
    np.cumsum(num_token_grads, out=offsets[1:])

    grid = scores.float().numpy()
    if grid.ndim == 2:
        grid = grid[:, :, None]
    flat = np.concatenate(
        [grid[r, : num_token_grads[r]] for r in range(len(num_token_grads))]
    )

    save_token_scores(path, flat, offsets)
    np.save(path / "doc_ids.npy", doc_ids)
    print(f"Saved attribution scores to {path}")
    emit_trajectory_plot(run_path, scores, batch_size)
    return str(path)


def shuffled_epochs(dataset: Dataset, seed: int, num_epochs: int) -> Dataset:
    """Concatenate `num_epochs` independently shuffled copies of `dataset`.

    Each epoch is seeded off `seed`, so the full sequence is reproducible:
    MAGIC's backward pass replays these exact steps.
    """
    assert num_epochs >= 1, f"num_epochs must be >= 1, got {num_epochs}"
    return concatenate_datasets(
        [dataset.shuffle(seed=seed + epoch) for epoch in range(num_epochs)]
    )


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
    validate: bool = False,
    baseline_model: str = "",
):
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(rank))

    # For each non-main local rank, suppress HF info and warning messages
    if rank != 0:
        hf_disable_pbar()
        hf_set_verbosity_error()

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            dist_backend(mixed=True),
            init_method=f"tcp://{addr}:{port}",
            device_id=dist_device_id(rank),
            rank=global_rank,
            world_size=world_size,
        )

    # Ensure total effective batch size is divisible by world size
    assert run_cfg.batch_size % world_size == 0

    # Pad train dataset to be divisible by batch_size (weight=0 for padding)
    train_dataset, num_train_docs, pad_count, weight_pad_count = (
        pad_dataset_to_batch_size(
            train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
        )
    )

    # Plain magic runs enter with score_path="" (scores are computed below).
    per_token = (isinstance(run_cfg, MagicConfig) and run_cfg.attribute_tokens) or (
        score_path and scores_are_per_token(score_path)
    )
    if per_token:
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
    torch.manual_seed(run_cfg.seed)
    torch.cuda.manual_seed_all(run_cfg.seed)
    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")

    if baseline_model:
        # The leave-k-out retrains still start from run_cfg.model.
        trainer, fwd_state, model = prepare_trainer(
            replace(run_cfg, model=baseline_model), rank, schedule
        )
    else:
        trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)

        if global_rank == 0:
            write_lr_history(ckpts_path, schedule, len(stream))

        fwd_state = trainer.train(
            fwd_state,
            stream,
            debug=run_cfg.debug,
            inplace=True,
            save_dir=ckpts_path,
            save_mode=run_cfg.save_mode,
            log_fn=log_fn,
            resume=run_cfg.resume,
            fsdp=run_cfg.fsdp,
            max_grad_norm=run_cfg.max_grad_norm,
            grad_accum_steps=run_cfg.grad_accum_steps,
            optimizer_cfg=(
                dict(
                    betas=(run_cfg.adam_beta1, run_cfg.adam_beta2),
                    eps=run_cfg.adam_eps,
                    eps_root=run_cfg.eps_root,
                )
                if run_cfg.save_optimizer_state == "all"
                else None
            ),
            save_interval=run_cfg.save_interval,
        )
        # Called on every rank: FSDP moments are DTensors whose gather is a
        # collective; rank 0 writes inside.
        if run_cfg.save_optimizer_state != "none":
            save_second_moments_as_optimizer_pt(
                model,  # type: ignore[reportArgumentType]
                fwd_state.opt_state,
                os.path.join(run_cfg.run_path, "optimizer.pt"),
            )

    # A sliced run leaves the shared baseline to the process that starts at 0.
    trails_slice = isinstance(run_cfg, ValidationConfig) and run_cfg.subset_start != 0
    if run_cfg.save_models and global_rank == 0 and not trails_slice:
        # For the leave-k-out family the trained model is the query baseline
        # that evaluate_retrained reads from retrained/base.
        if isinstance(run_cfg, ValidationConfig):
            base_dir = os.path.join(run_cfg.run_path, "retrained", "base")
        else:
            base_dir = os.path.join(run_cfg.run_path, "model")
        os.makedirs(base_dir, exist_ok=True)
        with fwd_state.activate(model), torch.no_grad():
            model.save_pretrained(base_dir, safe_serialization=True)
        base_tokenizer = AutoTokenizer.from_pretrained(
            run_cfg.tokenizer or run_cfg.model
        )
        base_tokenizer.save_pretrained(base_dir)

    # If no query dataset is provided, skip backward and validation entirely
    if query_dataset is None:
        return
    elif not isinstance(run_cfg, ValidationConfig):
        raise RuntimeError(
            "run_cfg must be a ValidationConfig if query_dataset is provided"
        )

    # Pad query dataset to be divisible by batch_size (weight=0 for padding)
    num_real_query_docs = num_query_docs
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
        fwd_state,
        model,
        query_stream,
        run_cfg.query_method,
        run_cfg.fsdp,
        run_cfg.grad_accum_steps,
    )

    multi_query = False
    if not score_path and run_cfg.query_method == "none":
        # Per-query MAGIC: one backward per query, sharing the forward. Yields
        # the unit for a per-query LDS.
        if not isinstance(run_cfg, MagicConfig):
            raise RuntimeError("run_cfg must be a MagicConfig to compute scores")
        assert query_dataset is not None

        scores = compute_per_query_magic_scores(
            trainer,
            ckpts_path,
            stream,
            fwd_state,
            model,
            query_dataset,
            num_real_query_docs,
            run_cfg,
            world_size,
            global_rank,
            pad_count,
            weight_pad_count,
        )
        multi_query = True
        if global_rank == 0:
            print(f"Baseline loss: {baseline}")
            print(f"Score summary: {describe(scores.flatten())}")
            score_path = save_magic_scores(
                run_cfg.run_path,
                scores,
                train_dataset,
                pad_count,
                bool(per_token),
                run_cfg.batch_size,
            )
    elif not score_path:
        # Sanity check
        if not isinstance(run_cfg, MagicConfig):
            raise RuntimeError("run_cfg must be a MagicConfig to compute scores")
        if run_cfg.save_mode == "interval":
            raise ValueError("save_mode='interval' not supported for MAGIC attribution")

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
            max_grad_norm=run_cfg.max_grad_norm,
            grad_accum_steps=run_cfg.grad_accum_steps,
            double_backward_batch_size=run_cfg.double_backward_batch_size,
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

            score_path = save_magic_scores(
                run_cfg.run_path,
                scores,
                train_dataset,
                pad_count,
                bool(per_token),
                run_cfg.batch_size,
            )
    else:
        scores, multi_query = load_scores_loss_signed(score_path)

    stream.requires_grad = False

    if not validate:
        return

    validate_scores(
        run_cfg,
        scores,
        multi_query,
        global_rank=global_rank,
        rank=rank,
        schedule=schedule,
        stream=stream,
        query_stream=query_stream,
        fwd_state=fwd_state,
        model=model,
        baseline=baseline,
        num_query_docs=num_query_docs,
        query_weight_pad_count=query_weight_pad_count,
        pad_count=pad_count,
        weight_pad_count=weight_pad_count,
    )


def run_magic(
    run_cfg: TrainingConfig,
    *,
    score_path: str = "",
    validate: bool = False,
    baseline_model: str = "",
):
    """Train ``run_cfg``, score the query set, and validate those scores."""
    if validate and not isinstance(run_cfg, ValidationConfig):
        raise TypeError("validation requires a ValidationConfig")

    run_path = Path(run_cfg.run_path)
    is_main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    multi_node = run_cfg.distributed.nnode > 1

    if is_main_node:
        if not has_matplotlib():
            print(
                "Hint: run `pip install matplotlib` for an optional "
                "trajectory plot of scores"
            )

        if run_path.exists() and not run_cfg.resume:
            if run_cfg.overwrite:
                shutil.rmtree(run_path)
            else:
                raise FileExistsError(
                    f"Run path {run_path} already exists. "
                    f"Use --overwrite to overwrite it."
                )

        run_path.mkdir(parents=True, exist_ok=True)
        save_run_config(run_cfg, run_path)

    # HF datasets caches are not safe for concurrent writers, so the main node
    # must finish populating the cache before others read from it.
    barrier = run_path / ".preprocess_done" if multi_node else None
    if barrier is not None and not is_main_node:
        run_path.mkdir(parents=True, exist_ok=True)
        while not barrier.exists():
            time.sleep(0.5)

    train_ds, train_n = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)

    train_ds = shuffled_epochs(train_ds, run_cfg.seed, max(1, run_cfg.num_epochs))

    if isinstance(run_cfg, ValidationConfig):
        query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)
    else:
        query_ds, query_n = None, 0

    if barrier is not None and is_main_node:
        barrier.touch()

    launch_distributed_run(
        "run_magic",
        worker,
        [
            train_ds,
            query_ds,
            train_n,
            query_n,
            run_cfg,
            score_path,
            validate,
            baseline_model,
        ],
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
