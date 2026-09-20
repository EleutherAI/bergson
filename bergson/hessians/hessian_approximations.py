import gc
import os
import shutil
import warnings
from contextlib import ExitStack

import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedModel

from bergson.collector.collector import (
    CollectorComputer,
    HookCollectorBase,
    fwd_bwd_hessian_factory,
)
from bergson.config.config import AttentionConfig, HessianConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.distributed import init_dist, launch_distributed_run
from bergson.gradients import GradientProcessor
from bergson.hessians.autocorrelation import (
    AutocorrelationCollector,
    JointAutocorrelationCollector,
)
from bergson.hessians.eigenvectors import (
    LambdaCollector,
    compute_eigendecomposition,
    save_uncorrected_eigenvalues,
)
from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.shampoo import ShampooCollector
from bergson.hessians.tkfac import TraceCovarianceCollector
from bergson.utils.utils import (
    convert_precision_to_torch,
    setup_reproducibility,
)
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)

HESSIAN_APPROXIMATIONS = {
    "kfac": CovarianceCollector,
    "tkfac": TraceCovarianceCollector,
    "shampoo": ShampooCollector,
}

FACTOR_SUBDIRS = (
    "activation_sharded",
    "gradient_sharded",
    "eigen_activation_sharded",
    "eigen_gradient_sharded",
    "eigenvalue_sharded",
    "factor_eig_a",
    "factor_eig_g",
    "eigenvalue_correction_sharded",
)
"""Per-module factor stores written under a Hessian run path, each holding one
``shard_{rank}.safetensors`` per rank keyed by module name."""


def partition_modules(names: list[str], num_partitions: int) -> list[list[str]]:
    """Split ``names`` into ``num_partitions`` contiguous groups of near-equal
    size, in the order given so a group covers adjacent layers."""
    if num_partitions < 1:
        raise ValueError(f"module_partitions must be >= 1, got {num_partitions}")
    num_partitions = min(num_partitions, len(names))
    size, extra = divmod(len(names), num_partitions)
    groups, start = [], 0
    for i in range(num_partitions):
        end = start + size + (1 if i < extra else 0)
        groups.append(names[start:end])
        start = end
    return groups


def merge_partitions(run_path: str | os.PathLike, num_partitions: int, rank: int):
    """Merge this rank's shard file from every ``partition_{i}`` directory under
    ``run_path`` into the run's own factor stores.

    Partitions hold disjoint modules, so a merge is a union of the per-module
    tensors; they are read through the safetensors mmap so the merged file is
    never held in memory twice. Rank 0 deletes the partition directories once
    every rank has merged.
    """
    part_dirs = [
        os.path.join(run_path, f"partition_{p}") for p in range(num_partitions)
    ]
    shard = f"shard_{rank}.safetensors"
    for sub in FACTOR_SUBDIRS:
        files = [os.path.join(d, sub, shard) for d in part_dirs]
        present = [os.path.exists(f) for f in files]
        if not any(present):
            continue
        if not all(present):
            missing = [f for f, ok in zip(files, present) if not ok]
            raise FileNotFoundError(f"Partition shards missing for {sub}: {missing}")

        merged = {}
        with ExitStack() as stack:
            for f in files:
                handle = stack.enter_context(safe_open(f, framework="pt", device="cpu"))
                for key in handle.keys():
                    merged[key] = handle.get_tensor(key)
            os.makedirs(os.path.join(run_path, sub), exist_ok=True)
            save_file(merged, os.path.join(run_path, sub, shard))

    dist.barrier() if dist.is_initialized() else None
    if rank == 0:
        for d in part_dirs:
            shutil.rmtree(d)


def approximate_hessians(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    *,
    do_eigendecomposition: bool = True,
) -> str:
    """
    Approximate Hessian matrices using KFAC or EKFAC.

    For KFAC: Computes Kronecker-factored covariance matrices and their
    eigendecompositions.

    For EKFAC: Additionally computes eigenvalue corrections for more
    accurate Hessian approximation.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and gradient collection settings.
    hessian_cfg : HessianConfig
        Specifies the Hessian approximation method (kfac or ekfac).
    do_eigendecomposition : bool
        If True (default), compute the eigendecomposition of the covariance
        matrices. Not needed when doing approximate unrolling

    Returns
    -------
    str
        Path to the directory containing the computed Hessian approximations.
    """
    if hessian_cfg.ev_correction and index_cfg.projection_dim != 0:
        raise ValueError(
            "EK-FAC (ev_correction=True) does not support gradient "
            f"projection; got index_cfg.projection_dim={index_cfg.projection_dim}. "
            "Set projection_dim=0."
        )

    if hessian_cfg.method == "autocorrelation" and index_cfg.projection_dim == 0:
        warnings.warn(
            "Computing an autocorrelation (dense) Hessian with "
            "index_cfg.projection_dim=0 (uncompressed gradients); this scales "
            "quadratically with the (possibly large) uncompressed gradient "
            "dimension. Set projection_dim (e.g. 16) unless this is "
            "intentional."
        )

    if index_cfg.debug:
        setup_reproducibility()
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    ds, _ = setup_data_pipeline(index_cfg)

    launch_distributed_run(
        "hessian",
        hessian_worker,
        [index_cfg, hessian_cfg, ds, do_eigendecomposition],
        index_cfg.distributed,
    )

    rank = index_cfg.distributed.rank
    if rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)

    return index_cfg.run_path


def hessian_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    ds: Dataset,
    do_eigendecomposition: bool = True,
    target_modules: set[str] | None = None,
):
    """
    Worker function for distributed Hessian approximation.

    Parameters
    ----------
    rank : int
        Global rank of this worker.
    local_rank : int
        Local rank on this node.
    world_size : int
        Total number of workers.
    index_cfg : IndexConfig
        Configuration for model, data, and gradient collection.
    hessian_cfg : HessianConfig
        Configuration for Hessian approximation method (kfac or ekfac).
    ds : Dataset
        Dataset to use for covariance estimation.
    do_eigendecomposition : bool
        If True (default), compute the eigendecomposition after collection.
    target_modules : set[str] | None
        Optional override for the target module set. When `None` (default)
        we fall back to what `setup_model_and_peft` returns.
    """
    init_dist(rank, local_rank, world_size)

    model, peft_target_modules = setup_model_and_peft(index_cfg)
    if target_modules is None:
        target_modules = peft_target_modules

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }
    batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)

    # The autocorrelation Hessian is a dense per-module gradient Gram so
    # it computes in one pass and skips the factored eigendecomposition
    if hessian_cfg.method == "autocorrelation":
        processor = create_processor(model, index_cfg, target_modules)
        collector_cls = (
            JointAutocorrelationCollector
            if hessian_cfg.structure == "joint"
            else AutocorrelationCollector
        )
        collector = collector_cls(
            model=model.base_model,  # type: ignore
            data=ds,
            path=str(index_cfg.partial_run_path),
            processor=processor,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs,
            filter_modules=index_cfg.filter_modules,
        )
        computer = CollectorComputer(
            model=model,  # type: ignore
            data=ds,
            collector=collector,
            batches=batches,
            cfg=index_cfg,
        )
        computer.run_with_collector_hooks(
            desc=f"Approximating {hessian_cfg.method} Hessian"
        )
        return

    kwargs = {
        "model": model,
        "data": ds,
        "index_cfg": index_cfg,
        "hessian_cfg": hessian_cfg,
        "attention_cfgs": attention_cfgs,
        "batches": batches,
        "do_eigendecomposition": do_eigendecomposition,
    }

    if hessian_cfg.module_partitions == 1:
        fit_factored_hessians(
            **kwargs,
            target_modules=target_modules,
            path=str(index_cfg.partial_run_path),
        )
        return

    # Fit the factors one module group at a time so only a group's factors are
    # on the device, then merge the groups' shard files into the run layout.
    target_info = HookCollectorBase.discover_targets(
        model.base_model,  # type: ignore
        target_modules,
        index_cfg.include_bias,
        index_cfg.filter_modules,
    )
    groups = partition_modules(list(target_info), hessian_cfg.module_partitions)
    for i, group in enumerate(groups):
        print(f"Fitting module partition {i + 1}/{len(groups)} ({len(group)} modules)")
        fit_factored_hessians(
            **kwargs,
            target_modules=set(group),
            path=os.path.join(index_cfg.partial_run_path, f"partition_{i}"),
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    merge_partitions(index_cfg.partial_run_path, len(groups), rank)


def fit_factored_hessians(
    model: PreTrainedModel,
    data: Dataset,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    *,
    target_modules: set[str] | None,
    attention_cfgs: dict[str, AttentionConfig],
    batches: list[list[int]],
    path: str,
    do_eigendecomposition: bool,
):
    """Fit the covariances of ``target_modules``, eigendecompose them and, for
    EK-FAC, collect the eigenvalue corrections, writing everything under ``path``."""
    if target_modules is not None:
        attention_cfgs = {
            k: v for k, v in attention_cfgs.items() if k in target_modules
        }

    kwargs = {
        "model": model,
        "data": data,
        "index_cfg": index_cfg,
        "hessian_cfg": hessian_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
        "batches": batches,
        "path": path,
    }

    collect_hessians(**kwargs)
    _release_device_memory()

    dist.barrier() if dist.is_initialized() else None

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if not do_eigendecomposition:
        return

    total_processed = torch.load(
        f"{index_cfg.partial_run_path}/total_processed.pt",
        map_location="cpu",
        weights_only=False,
    )

    eigenvalues_a = compute_eigendecomposition(
        os.path.join(path, "activation_sharded"),
        total_processed=total_processed,
    )
    eigenvalues_g = compute_eigendecomposition(
        os.path.join(path, "gradient_sharded"),
        total_processed=total_processed,
    )

    dist.barrier() if dist.is_initialized() else None

    save_uncorrected_eigenvalues(
        partial_run_path=path,
        eigenvalues_a=eigenvalues_a,
        eigenvalues_g=eigenvalues_g,
        total_processed=total_processed,
        rank=rank,
        world_size=world_size,
    )

    if hessian_cfg.ev_correction:
        collect_hessians(**kwargs, ev_correction=True)
        _release_device_memory()


def _release_device_memory():
    """Drop a finished collector's device tensors before the next pass.

    The collector, its computer and the model hooks reference each other, so
    the tensors only go when the cycle collector runs; on a 14B model that
    left the covariance shards (20GB per rank) resident under the eigenvalue
    correction pass.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_hessians(
    model: PreTrainedModel,
    data: Dataset,
    index_cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    hessian_cfg: HessianConfig,
    ev_correction: bool = False,
    eigen_path: str | None = None,
    output_subdir: str = "eigenvalue_correction_sharded",
    path: str | None = None,
):
    """
    Compute Hessian approximations using the hooks specified in the collector.
    If ev_correction is True, uses LambdaCollector to compute eigenvalue corrections.
    ``path`` overrides where the collector writes; it defaults to the partial
    run path.
    """

    hessian_dtype = convert_precision_to_torch(hessian_cfg.hessian_dtype)

    collector_args = {
        "model": model.base_model,  # type: ignore
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs or {},
        "path": path or str(index_cfg.partial_run_path),
        "filter_modules": index_cfg.filter_modules,
        "processor": GradientProcessor(include_bias=index_cfg.include_bias),
        "dtype": hessian_dtype,
    }
    desc = f"Approximating Hessians with {hessian_cfg.method}"
    if ev_correction:
        collector = LambdaCollector(
            **collector_args,
            eigen_path=eigen_path,
            output_subdir=output_subdir,
        )
        desc += " (eigenvalue correction)"
    else:
        collector = HESSIAN_APPROXIMATIONS[hessian_cfg.method](**collector_args)

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=data,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )

    computer.forward_backward = fwd_bwd_hessian_factory(index_cfg, hessian_cfg)

    computer.run_with_collector_hooks(desc=desc)
