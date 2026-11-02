"""The ``gnh-unroll`` command: unrolled Gauss-Newton scores of a query set
against the documents of an AdamW training run."""

import os
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from scipy.stats import describe
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils.logging import (
    disable_progress_bar as hf_disable_pbar,
)
from transformers.utils.logging import (
    set_verbosity_error as hf_set_verbosity_error,
)

from ..config.config_io import save_run_config
from ..data import sorted_checkpoints
from ..distributed import DIST_TIMEOUT, launch_distributed_run
from ..magic.cli import (
    attach_doc_ids_if_missing,
    attach_example_ids,
    compute_query_gradients,
    save_magic_scores,
    shuffled_epochs,
)
from ..magic.data_stream import DataStream, Padding, pad_dataset_to_batch_size
from ..magic.fsdp import full_state_dict
from ..magic.score_plot import plot_score_trajectory
from ..magic.trainer import Trainer, TrainerState, prepare_trainer, write_lr_history
from ..utils.logging import wandb_log_fn
from ..utils.utils import dist_backend, dist_device_id, get_device, get_device_index
from ..utils.worker_utils import setup_data_pipeline
from ..validate import build_contrast_stream
from .config import GnhUnrollConfig
from .unroll import unroll_scores, unrolled_names


def replay_to_end(
    trainer: Trainer,
    state: TrainerState,
    stream: DataStream,
    ckpts_path: str,
    grad_accum_steps: int,
) -> TrainerState:
    """Load the last complete checkpoint in ``ckpts_path`` into ``state`` and
    step it to the end of ``stream`` without writing anything."""
    complete = [
        (i, p)
        for i, p in sorted_checkpoints(ckpts_path)
        if os.path.exists(os.path.join(p, ".metadata"))
    ]
    if not complete:
        raise FileNotFoundError(f"No complete checkpoints in {ckpts_path}")
    last, path = complete[-1]
    if last > len(stream):
        raise ValueError(
            f"{path} is past the end of this run ({len(stream)} steps); the "
            "training options must match the run that wrote the checkpoints."
        )
    state.load(path)
    state.detach_()
    state.batch_index = last
    main = not dist.is_initialized() or dist.get_rank() == 0
    for t in tqdm(range(last, len(stream)), desc="Replaying", disable=not main):
        state = trainer.step(
            state, stream[t], inplace=True, grad_accum_steps=grad_accum_steps
        )
    return state


def score_queries(
    trainer: Trainer,
    ckpts_path: str,
    stream: DataStream,
    fwd_state: TrainerState,
    query_grad: Callable[[int], dict[str, torch.Tensor]],
    num_queries: int,
    run_cfg: GnhUnrollConfig,
    world_size: int,
    global_rank: int,
    padding: Padding,
    schedule: Callable[[int], float],
) -> torch.Tensor:
    """Scores of ``num_queries`` queries against every training document,
    ``[num_train_docs, num_queries]``.

    ``query_grad(i)`` is a collective that returns query ``i``'s gradient on
    every rank. Queries are dealt round-robin to the ranks and each rank scores
    ``query_batch_size`` of its own per pass back through the run. Each pass
    writes ``<run_path>/per_query/q{i}.pt``, and a pass on disk is skipped on
    resume.
    """
    device = stream.weights.device
    scores_dir = os.path.join(run_cfg.run_path, "per_query")
    if global_rank == 0:
        os.makedirs(scores_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    def query_path(qi: int) -> str:
        return os.path.join(scores_dir, f"q{qi}.pt")

    names = unrolled_names(trainer.model, fwd_state.params)
    mine = list(range(global_rank, num_queries, world_size))
    size = run_cfg.query_batch_size
    passes = -(-max(1, -(-num_queries // world_size)) // size)
    for p in range(passes):
        group = mine[p * size : (p + 1) * size]
        todo = [qi for qi in group if not os.path.exists(query_path(qi))]
        done = torch.tensor([len(todo) == 0], device=device)
        if world_size > 1:
            dist.all_reduce(done, op=dist.ReduceOp.MIN)
        if bool(done.item()):
            continue

        queries = []
        for qi in range(num_queries):
            # Every rank takes part in each gradient, then keeps its own.
            if (qi // world_size) // size != p:
                continue
            grads = query_grad(qi)
            if qi in group:
                queries.append({k: grads[k] for k in names})
            del grads

        scores = unroll_scores(
            trainer,
            ckpts_path,
            stream,
            fwd_state,
            queries,
            lr=schedule,
            betas=(run_cfg.adam_beta1, run_cfg.adam_beta2),
            eps=run_cfg.adam_eps,
            eps_root=run_cfg.eps_root,
            weight_decay=run_cfg.weight_decay,
            grad_accum_steps=run_cfg.grad_accum_steps,
            double_backward_batch_size=run_cfg.double_backward_batch_size,
            log_dir=run_cfg.run_path if run_cfg.debug else "",
        )
        del queries
        for j, qi in enumerate(group):
            s = padding.trim(scores[..., j].cpu())
            torch.save(s, query_path(qi) + ".tmp")
            os.replace(query_path(qi) + ".tmp", query_path(qi))
        if world_size > 1:
            dist.barrier()
        if global_rank == 0:
            print(f"[gnh-unroll] pass {p + 1}/{passes} scored")

    return torch.stack([torch.load(query_path(qi)) for qi in range(num_queries)], -1)


def worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset: Dataset,
    query_dataset: Dataset,
    num_train_docs: int,
    num_query_docs: int,
    run_cfg: GnhUnrollConfig,
):
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(rank))

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
            timeout=DIST_TIMEOUT,
            world_size=world_size,
        )

    assert run_cfg.batch_size % world_size == 0
    device = get_device(rank)

    train_dataset, num_train_docs, padding = pad_dataset_to_batch_size(
        train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
    )
    if run_cfg.attribute_tokens:
        seq_len = run_cfg.data.chunk_length
        if seq_len <= 0:
            seq_len = max(train_dataset["length"])
        # One weight row per example; the pad rows share one synthetic row.
        w_shape = (max(train_dataset["example_ids"]) + 1, seq_len)
    else:
        w_shape = (num_train_docs,)
    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=device,
        input_key=run_cfg.data.prompt_column,
        weight_shape=w_shape,
    )
    padding.zero_weights(stream.weights.data)

    log_fn = None
    if run_cfg.wandb_project and global_rank == 0:
        log_fn = wandb_log_fn(run_cfg.wandb_project, config=asdict(run_cfg))

    if dist.is_initialized():
        dist.barrier()

    schedule = run_cfg.lr_schedule.get_schedule(len(stream))
    torch.manual_seed(run_cfg.seed)
    torch.cuda.manual_seed_all(run_cfg.seed)
    trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)

    if run_cfg.checkpoints:
        ckpts_path = run_cfg.checkpoints
        fwd_state = replay_to_end(
            trainer, fwd_state, stream, ckpts_path, run_cfg.grad_accum_steps
        )
    else:
        ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
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
            grad_accum_steps=run_cfg.grad_accum_steps,
            save_interval=run_cfg.save_interval,
        )

    if run_cfg.save_models:
        # Gather on every rank, save on rank 0. The leave-k-out family reads
        # the trained model from retrained/base as the query baseline.
        with fwd_state.activate(model), torch.no_grad():
            state = full_state_dict(model)
        if global_rank == 0:
            base_dir = os.path.join(run_cfg.run_path, "retrained", "base")
            os.makedirs(base_dir, exist_ok=True)
            model.save_pretrained(base_dir, state_dict=state, safe_serialization=True)
            base_tokenizer = AutoTokenizer.from_pretrained(
                run_cfg.tokenizer or run_cfg.model
            )
            base_tokenizer.save_pretrained(base_dir)
        del state

    num_real_query_docs = num_query_docs
    query_dataset, num_query_docs, query_padding = pad_dataset_to_batch_size(
        query_dataset, run_cfg.batch_size, num_query_docs, "Query", global_rank
    )
    if len(query_dataset) < run_cfg.batch_size:
        raise ValueError(
            f"Query dataset has {len(query_dataset)} examples, fewer than "
            f"batch_size={run_cfg.batch_size}. Use a larger query split or "
            f"smaller batch_size."
        )
    query_stream = DataStream(
        query_dataset,
        run_cfg.batch_size,
        device=device,
        input_key=run_cfg.query.prompt_column,
        weight_shape=(num_query_docs,),
    )
    query_padding.zero_weights(query_stream.weights.data)

    query_grads, baseline = compute_query_gradients(
        fwd_state,
        model,
        query_stream,
        run_cfg.query.aggregation,
        False,
        run_cfg.grad_accum_steps,
        ckpts_path,
        run_cfg.ckpt_avg_k,
    )
    fwd_state.detach_()

    # A contrast set turns the query into ``loss(query) - loss(contrast)``.
    contrast_stream = build_contrast_stream(run_cfg, run_cfg.query, device, global_rank)
    if contrast_stream is not None:
        contrast_grads, contrast_baseline = compute_query_gradients(
            fwd_state,
            model,
            contrast_stream,
            run_cfg.query.aggregation,
            False,
            run_cfg.grad_accum_steps,
            ckpts_path,
            run_cfg.ckpt_avg_k,
        )
        fwd_state.detach_()
        query_grads = {k: g - contrast_grads[k] for k, g in query_grads.items()}
        baseline = baseline - contrast_baseline

    def one_query_grad(qi: int) -> dict[str, torch.Tensor]:
        # A one-document stream indexes its weights by row, not by doc id.
        # One row per rank; more rows are zero-weight padding.
        one = query_dataset.select([qi])
        if "doc_ids" in one.column_names:
            one = one.remove_columns("doc_ids")
        one, n_one, one_padding = pad_dataset_to_batch_size(
            one, world_size, 1, f"Query {qi}", global_rank
        )
        qstream = DataStream(
            one,
            world_size,
            device=device,
            input_key=run_cfg.query.prompt_column,
            weight_shape=(n_one,),
        )
        one_padding.zero_weights(qstream.weights.data)
        grads, _ = compute_query_gradients(
            fwd_state, model, qstream, "mean", False, run_cfg.grad_accum_steps
        )
        fwd_state.detach_()
        return grads

    if run_cfg.query.aggregation == "none":
        query_grad, num_queries = one_query_grad, num_real_query_docs
    else:
        query_grad, num_queries = lambda qi: query_grads, 1

    scores = score_queries(
        trainer,
        ckpts_path,
        stream,
        fwd_state,
        query_grad,
        num_queries,
        run_cfg,
        world_size,
        global_rank,
        padding,
        schedule,
    )

    if global_rank == 0:
        print(f"Baseline loss: {baseline}")
        print(f"Score summary: {describe(scores.flatten())}")
        save_magic_scores(
            run_cfg.run_path,
            scores,
            train_dataset,
            padding,
            run_cfg.attribute_tokens,
        )
        if not run_cfg.attribute_tokens:
            plot_score_trajectory(
                scores.float().numpy(),
                run_cfg.batch_size,
                Path(run_cfg.run_path) / "score_vs_step.png",
            )


def run_gnh_unroll(run_cfg: GnhUnrollConfig):
    """Score ``run_cfg.query`` against the documents of ``run_cfg``'s training
    run with unrolled Gauss-Newton."""
    run_path = Path(run_cfg.run_path)
    if run_path.exists() and not run_cfg.resume:
        if run_cfg.overwrite:
            shutil.rmtree(run_path)
        else:
            raise FileExistsError(
                f"Run path {run_path} already exists. Use --overwrite to overwrite it."
            )
    run_path.mkdir(parents=True, exist_ok=True)
    save_run_config(run_cfg, run_path)

    train_ds, train_n = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)
    train_ds = attach_example_ids(train_ds)
    train_ds = shuffled_epochs(train_ds, run_cfg.seed, max(1, run_cfg.num_epochs))
    query_ds, query_n = setup_data_pipeline(run_cfg, run_cfg.query)

    launch_distributed_run(
        "run_gnh_unroll",
        worker,
        [train_ds, query_ds, train_n, query_n, run_cfg],
        run_cfg.distributed,
    )
