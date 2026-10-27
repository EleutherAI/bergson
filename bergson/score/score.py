import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

from bergson.collection import collect_gradients
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.data import (
    allocate_batches,
    allocate_dpo_batches,
    load_gradients,
)
from bergson.distributed import (
    cap_world_size_to_dataset,
    launch_distributed_run,
    parent_barrier,
)
from bergson.process_grads import (
    get_trackstar_hessian,
    normalize_and_aggregate_grads,
)
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    MemmapTokenScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import (
    assert_type,
    convert_precision_to_torch,
    get_device,
    get_device_index,
    get_gradient_dtype,
)
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def get_query_grads(
    score_cfg: ScoreConfig,
) -> tuple[dict[str, torch.Tensor], PreprocessConfig]:
    """
    Load query gradients from the mmap index and return as a dict of tensors.

    Returns
    -------
    tuple[dict[str, torch.Tensor], bool]
        The query gradients and whether they were already preconditioned
        (e.g. during a reduce step).
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

    preprocess_path = Path(query_path / "preprocess_config.yaml")
    if preprocess_path.exists():
        preprocess_cfg = PreprocessConfig.load(preprocess_path)
    else:
        preprocess_cfg = PreprocessConfig()

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

    return grads, preprocess_cfg


def _make_split_hessian(
    hessians: dict[str, torch.Tensor],
    modules: list[str],
    device: torch.device,
    dtype: torch.dtype,
):
    """Build a per-batch index transform for split (two-sided) preconditioning."""
    stacked = torch.stack([hessians[m] for m in modules])

    def transform(
        grads: dict[str, torch.Tensor],
        _modules: list[str] = modules,
        _stacked: torch.Tensor = stacked,
        _device: torch.device = device,
        _dtype: torch.dtype = dtype,
    ) -> dict[str, torch.Tensor]:
        g = torch.stack(
            [grads[m].to(_device, _dtype, non_blocking=True) for m in _modules],
            dim=1,
        )
        result = torch.bmm(g.permute(1, 0, 2), _stacked).permute(1, 0, 2)
        return {m: result[:, i] for i, m in enumerate(_modules)}

    return transform


def create_scorer(
    path: Path,
    data: Dataset,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    attribute_tokens: bool = False,
) -> Scorer:
    """Create a Scorer with MemmapScoreWriter for disk-based scoring.

    Loads query gradients from disk, preprocesses them if not already
    preprocessed, and constructs the Scorer.

    * Loads hessian from ``preprocess_cfg.hessian_path``.
    * Applies to query grads once here (unless already preconditioned).
    * Normalizes and aggregates (unless already done).
    * Builds an ``index_transform`` closure for per-batch index
      preconditioning in split mode (``unit_normalize=True``).
    """
    query_grads, query_preprocess_cfg = get_query_grads(score_cfg)
    # EKFAC inverse application can promote fp32 model gradients to fp64.
    # Keep that precision in the dot products and their on-disk score writer.
    if any(query.dtype == torch.float64 for query in query_grads.values()):
        dtype = torch.float64

    # Load hessian: H^(-1/2) for split, H^(-1) for one-sided
    hessians = get_trackstar_hessian(
        preprocess_cfg.hessian_path,
        device=device,
        power=-0.5 if preprocess_cfg.unit_normalize else -1,
        return_dtype=dtype,
    )

    # Maybe precondition query grads if it hasn't already been applied, e.g.
    # during reduce.
    if hessians and not bool(query_preprocess_cfg.hessian_path):
        query_grads = {
            m: query_grads[m].to(device=device, dtype=dtype) @ hessians[m]
            for m in score_cfg.modules
        }

    # Build index_transform for split (two-sided) preconditioning
    index_transform = (
        _make_split_hessian(
            hessians,
            score_cfg.modules,
            device,
            dtype,
        )
        if hessians and preprocess_cfg.unit_normalize
        else lambda x: x
    )

    # Maybe apply aggregation if it hasn't already been applied.
    normalize_aggregated_grad = (
        False
        if query_preprocess_cfg.normalize_aggregated_grad
        else preprocess_cfg.normalize_aggregated_grad
    )
    aggregation = (
        "none"
        if query_preprocess_cfg.aggregation != "none"
        else preprocess_cfg.aggregation
    )
    unit_normalize = (
        False if query_preprocess_cfg.unit_normalize else preprocess_cfg.unit_normalize
    )

    query_grads = normalize_and_aggregate_grads(
        query_grads,
        score_cfg.modules,
        unit_normalize=unit_normalize,
        device=torch.device("cpu") if score_cfg.offload_queries else device,
        aggregate_grads=aggregation,
        normalize_aggregated_grad=normalize_aggregated_grad,
    )

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
        unit_normalize=preprocess_cfg.unit_normalize,
        score_mode=score_cfg.score,
        attribute_tokens=attribute_tokens,
        index_transform=index_transform,
        offload_queries=score_cfg.offload_queries,
    )


def score_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    ds: Dataset | IterableDataset,
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
    preprocess_cfg : PreprocessConfig
        Preprocessing configuration for gradient normalization/preconditioning.
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    """
    torch.cuda.set_device(get_device_index(local_rank))

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(local_rank)),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, index_cfg, target_modules)

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
    score_device = torch.device(get_device(local_rank))

    if isinstance(ds, Dataset):
        kwargs["batches"] = (
            allocate_dpo_batches(
                ds["length"][:],
                index_cfg.token_batch_size,
                max_batch_size=index_cfg.max_batch_size,
            )
            if index_cfg.loss_fn == "dpo"
            else allocate_batches(
                ds["length"][:],
                index_cfg.token_batch_size,
                max_batch_size=index_cfg.max_batch_size,
            )
        )
        scorer_data = (
            ds.select(range(len(ds) // 2)) if index_cfg.loss_fn == "dpo" else ds
        )
        kwargs["scorer"] = create_scorer(
            index_cfg.partial_run_path,
            scorer_data,
            score_cfg,
            preprocess_cfg,
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
                ds_shard["length"][:],
                index_cfg.token_batch_size,
                max_batch_size=index_cfg.max_batch_size,
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

            kwargs["scorer"] = create_scorer(
                index_cfg.partial_run_path / f"shard-{shard_id:05d}",
                ds_shard,
                score_cfg,
                preprocess_cfg,
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
    preprocess_cfg: PreprocessConfig,
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
    preprocess_cfg : PreprocessConfig
        Preprocessing configuration for gradient normalization/preconditioning.
    """
    worker = score_worker
    if score_cfg.token_influence == 'output':
        from bergson.score.output_influence import check_output_influence_supported
        from bergson.score.output_worker import output_influence_worker
        check_output_influence_supported(index_cfg, preprocess_cfg)
        if score_cfg.score != 'individual':
            raise ValueError('Output influence requires individual dot-product scores')
        worker = output_influence_worker
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    index_cfg.save_yaml(index_cfg.partial_run_path / "index_config.yaml")
    score_cfg.save_yaml(index_cfg.partial_run_path / "score_config.yaml")

    ds, _ = setup_data_pipeline(index_cfg)

    dist_cfg = index_cfg.distributed
    if isinstance(ds, Dataset) and len(ds) < dist_cfg.world_size:
        dist_cfg = cap_world_size_to_dataset(dist_cfg, len(ds))
        print(
            f"reducing to nnode=1 and nproc_per_node={dist_cfg.nproc_per_node} for step"
        )

    launch_distributed_run(
        "score",
        worker,
        [index_cfg, score_cfg, preprocess_cfg, ds],
        dist_cfg,
    )

    if dist_cfg.rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)

    if dist_cfg.world_size < index_cfg.distributed.world_size:
        parent_barrier(index_cfg.distributed)
