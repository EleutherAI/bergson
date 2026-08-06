import gc
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

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
from ..config.config_io import read_first_step_config, save_run_config
from ..distributed import launch_distributed_run
from ..utils.load_from_optimizer import (
    save_second_moments_as_optimizer_pt,
)
from ..utils.logging import wandb_log_fn
from ..utils.utils import get_device, get_device_index
from ..utils.worker_utils import setup_data_pipeline
from ..validate import load_attribution_scores, validate_scores
from .config import MagicConfig
from .data_stream import DataStream, pad_dataset_to_batch_size
from .grad_accum import accumulate_grads
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
            grads, loss = accumulate_grads(
                model, params, batch, grad_accum_steps, create_graph=False
            )

            if grad_accum is None:
                grad_accum = {k: g.detach().clone() for k, g in grads.items()}
            else:
                for k, g in grads.items():
                    grad_accum[k] += g.detach()

            loss_accum += loss

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
        loss_tensor = torch.tensor(loss_accum, device=torch.cuda.current_device())
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

        one = query_dataset.select([qi])
        one, n_one, one_pad, one_wpad = pad_dataset_to_batch_size(
            one, run_cfg.batch_size, 1, f"Query {qi}", global_rank
        )
        qstream = DataStream(
            one,
            run_cfg.batch_size,
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
    if os.path.isdir(score_path):
        info_path = os.path.join(score_path, "info.json")
        if not os.path.isfile(info_path):
            return False
        with open(info_path) as f:
            return bool(json.load(f).get("attribute_tokens", False))
    step_cfg = read_first_step_config(score_path)
    if step_cfg is not None:
        return bool(step_cfg.get("attribute_tokens") or step_cfg.get("per_token"))

    scores = torch.load(score_path, map_location="cpu")
    return isinstance(scores, torch.Tensor) and (
        scores.ndim == 3 or (scores.ndim == 2 and scores.shape[1] > 1)
    )


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


def save_doc_ids(run_path: str, train_dataset: Dataset, pad_count: int) -> str:
    """Write ``doc_ids.pt`` beside a per-token ``scores.pt``."""
    doc_ids = torch.tensor(train_dataset["doc_ids"])
    if pad_count:
        doc_ids = doc_ids[:-pad_count]

    doc_ids_path = os.path.join(run_path, "doc_ids.pt")
    torch.save(doc_ids, doc_ids_path)
    print(f"Saved doc_ids to {doc_ids_path}")
    return doc_ids_path


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
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(rank)),
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
    trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    resume = run_cfg.resume

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
        resume=resume,
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

    if run_cfg.save_models and global_rank == 0:
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
            score_path = os.path.join(run_cfg.run_path, "scores.pt")
            torch.save(scores, score_path)
            print(f"Saved per-query attribution scores to {score_path}")
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

            score_path = os.path.join(run_cfg.run_path, "scores.pt")
            torch.save(scores, score_path)
            print(f"Saved attribution scores to {score_path}")
    elif os.path.isdir(score_path):
        scores, multi_query = load_attribution_scores(score_path)
    else:
        scores = torch.load(score_path, map_location="cpu")

    if per_token and global_rank == 0:
        save_doc_ids(run_cfg.run_path, train_dataset, pad_count)

    stream.requires_grad = False

    if isinstance(run_cfg, MagicConfig) and run_cfg.skip_validation:
        return

    validate_scores(
        run_cfg,
        scores,
        multi_query,
        global_rank=global_rank,
        rank=rank,
        world_size=world_size,
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


def run_magic(run_cfg: TrainingConfig, *, score_path: str = ""):
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
        [train_ds, query_ds, train_n, query_n, run_cfg, score_path],
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
