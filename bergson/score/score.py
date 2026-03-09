import json
import os
import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

from bergson.collection import collect_gradients
from bergson.config import IndexConfig, ScoreConfig
from bergson.data import (
    allocate_batches,
    load_gradients,
)
from bergson.distributed import launch_distributed_run
from bergson.gradients import GradientProcessor
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    MemmapTokenScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.math import compute_damped_inverse
from bergson.utils.utils import (
    assert_type,
    convert_precision_to_torch,
    get_gradient_dtype,
)
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def create_scorer(
    path: Path,
    data: Dataset,
    query_grads: dict[str, torch.Tensor],
    score_cfg: ScoreConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    attribute_tokens: bool = False,
) -> Scorer:
    """Create a Scorer with MemmapScoreWriter for disk-based scoring."""
    num_queries = len(query_grads[score_cfg.modules[0]])
    if attribute_tokens:
        writer = MemmapTokenScoreWriter(
            path,
            data,
            num_queries,
            dtype=dtype,
        )
    else:
        writer = MemmapSequenceScoreWriter(path, len(data), num_queries, dtype=dtype)
    return Scorer(
        query_grads=query_grads,
        modules=score_cfg.modules,
        writer=writer,
        device=device,
        dtype=dtype,
        unit_normalize=score_cfg.unit_normalize,
        score_mode="nearest" if score_cfg.score == "nearest" else "inner_product",
        attribute_tokens=attribute_tokens,
    )


def preprocess_grads(
    grad_dict: dict[str, torch.Tensor],
    grad_column_names: list[str],
    unit_normalize: bool,
    device: torch.device,
    accumulate_grads: Literal["mean", "sum", "none"] = "none",
    normalize_accumulated_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocess the gradients. Returns a dictionary of preprocessed gradients
    with shape [N, grad_dim] or [1, grad_dim]. Preprocessing includes some
    combination of per-item unit normalization, accumulation, accumulated
    gradient normalization, and dtype conversion."""

    # Short-circuit if possible
    if accumulate_grads == "none" and not unit_normalize:
        return {name: grad_dict[name].to(device=device) for name in grad_column_names}

    grads = {
        name: grad_dict[name].to(device=device, dtype=torch.float32)
        for name in grad_column_names
    }

    # Per-item unit normalization
    if unit_normalize:
        norms = torch.cat(list(grads.values()), dim=1).norm(dim=1, keepdim=True)
        grads = {k: v / norms for k, v in grads.items()}

    # Accumulate across items
    if accumulate_grads == "mean":
        grads = {name: grads[name].mean(0, keepdim=True) for name in grad_column_names}
    elif accumulate_grads == "sum":
        grads = {name: grads[name].sum(0, keepdim=True) for name in grad_column_names}
    elif accumulate_grads != "none":
        raise ValueError(f"Invalid accumulate_grads: {accumulate_grads}")

    # Normalize the accumulated gradient
    if normalize_accumulated_grad:
        grad_norm = torch.cat(
            [grads[name].flatten() for name in grad_column_names], dim=0
        ).norm()
        for name in grad_column_names:
            grads[name] /= grad_norm

    return grads


def precondition_grads(
    grads: dict[str, torch.Tensor],
    score_cfg: ScoreConfig,
    target_modules: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Precondition query gradients with the query and/or index preconditioners."""
    use_q = score_cfg.query_preconditioner_path is not None
    use_i = score_cfg.index_preconditioner_path is not None

    if use_q or use_i:
        q, i = {}, {}
        if use_q:
            assert score_cfg.query_preconditioner_path is not None
            q = GradientProcessor.load(
                Path(score_cfg.query_preconditioner_path),
                map_location=device,
            ).preconditioners
        if use_i:
            assert score_cfg.index_preconditioner_path is not None
            i = GradientProcessor.load(
                Path(score_cfg.index_preconditioner_path), map_location=device
            ).preconditioners

        mixed_preconditioner = (
            {
                k: q[k] * score_cfg.mixing_coefficient
                + i[k] * (1 - score_cfg.mixing_coefficient)
                for k in q
            }
            if (q and i)
            else (q or i)
        )

        # Compute H^(-1) via eigendecomposition and apply to query gradients
        h_inv = {
            name: compute_damped_inverse(H.to(device=device))
            for name, H in mixed_preconditioner.items()
        }

        grads = {
            name: (
                grads[name].to(device=device, dtype=h_inv[name].dtype) @ h_inv[name]
            ).cpu()
            for name in target_modules
        }

    return {name: grads[name] for name in score_cfg.modules}


def get_query_grads(score_cfg: ScoreConfig) -> dict[str, torch.Tensor]:
    """
    Load query gradients from the mmap index and return as a dict of tensors.
    Preconditioners may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    query_path = Path(score_cfg.query_path)
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query dataset not found at {score_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    with open(query_path / "info.json", "r") as f:
        metadata = json.load(f)
        target_modules = metadata["dtype"]["names"]
        grad_sizes = metadata["grad_sizes"]

    if not score_cfg.modules:
        score_cfg.modules = target_modules

    mmap = load_gradients(Path(score_cfg.query_path), structured=False)

    sizes = torch.tensor(list(grad_sizes.values()))
    module_offsets = torch.tensor([0] + torch.cumsum(sizes, dim=0).tolist())

    # Cast to float32 only for dtypes not natively supported by numpy (e.g. bfloat16)
    needs_cast = not np.issubdtype(mmap.dtype, np.floating)
    grads: dict[str, torch.Tensor] = {}
    for i, name in enumerate(grad_sizes.keys()):
        if name not in target_modules:
            continue
        sliced = mmap[:, module_offsets[i] : module_offsets[i + 1]]
        if needs_cast:
            grads[name] = torch.from_numpy(sliced.astype(np.float32))
        else:
            grads[name] = torch.from_numpy(sliced.copy())

    return grads


def score_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    ds: Dataset | IterableDataset,
    query_grads: dict[str, torch.Tensor],
):
    """
    Score worker executed per rank to produce and score gradients against a query.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    local_rank : int
        Local rank / GPU ID for this worker on the node.
    world_size : int
        Total number of workers participating in the run.
    index_cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    score_cfg : ScoreConfig
        Score configuration specifying query path, target modules, and scoring
        method (mean/nearest/individual).
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    query_grads : dict[str, torch.Tensor]
        Preprocessed query gradient tensors (often [1, grad_dim]) keyed by module name.
    """
    torch.cuda.set_device(local_rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, ds, index_cfg, target_modules)

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": index_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    score_dtype = (
        convert_precision_to_torch(score_cfg.precision)
        if score_cfg.precision != "auto"
        else get_gradient_dtype(model)
    )
    score_device = torch.device(f"cuda:{rank}")

    if isinstance(ds, Dataset):
        kwargs["batches"] = allocate_batches(ds["length"], index_cfg.token_batch_size)
        kwargs["scorer"] = create_scorer(
            index_cfg.partial_run_path,
            ds,
            query_grads,
            score_cfg,
            device=score_device,
            dtype=score_dtype,
            attribute_tokens=index_cfg.attribute_tokens,
        )

        collect_gradients(**kwargs)
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush(kwargs):
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

            kwargs["scorer"] = create_scorer(
                index_cfg.partial_run_path / f"shard-{shard_id:05d}",
                ds_shard,
                query_grads,
                score_cfg,
                device=score_device,
                dtype=score_dtype,
            )

            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(index_cfg.partial_run_path)


def score_dataset(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_device=torch.device("cuda:0"),
):
    """
    Score a dataset against an existing gradient index.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and other gradient collection settings.
    score_cfg : ScoreConfig
        Specifies the query path, target modules, and scoring method
        (mean/nearest/individual).
    """
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (index_cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(index_cfg), f, indent=2)
    with (index_cfg.partial_run_path / "score_config.json").open("w") as f:
        json.dump(asdict(score_cfg), f, indent=2)

    ds = setup_data_pipeline(index_cfg)

    query_grads = get_query_grads(score_cfg)
    query_grads = precondition_grads(
        query_grads, score_cfg, score_cfg.modules, preprocess_device
    )
    query_grads = preprocess_grads(
        query_grads,
        score_cfg.modules,
        score_cfg.unit_normalize,
        preprocess_device,
        accumulate_grads="mean" if score_cfg.score == "mean" else "none",
        normalize_accumulated_grad=score_cfg.score == "mean",
    )

    launch_distributed_run(
        "score",
        score_worker,
        [index_cfg, score_cfg, ds, query_grads],
        index_cfg.distributed,
    )

    if index_cfg.distributed.rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
