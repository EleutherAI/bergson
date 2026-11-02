import gc
import json
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
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
from bergson.hessians.sharded_computation import assign_module_owners
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
"""Factor stores under a Hessian run path, one ``shard_{rank}.safetensors`` each."""


DONE_SUFFIX = ".done"
"""Marks a finished pass of a fit, so a resumed fit can skip it."""


def _pass_done(path: str, name: str, setup: dict) -> bool:
    """Whether the pass ``name`` under ``path`` finished with the same ``setup``.
    Rank 0 decides for every rank, so they all skip the same passes."""
    done = [False]
    if not dist.is_initialized() or dist.get_rank() == 0:
        marker = os.path.join(path, name + DONE_SUFFIX)
        if os.path.exists(marker):
            with open(marker) as f:
                saved = json.load(f)
            if saved != setup:
                raise ValueError(
                    f"{marker} was written by a fit set up as {saved}, not "
                    f"{setup}. Rerun without resume to start the fit over."
                )
            done = [True]
    if dist.is_initialized():
        dist.broadcast_object_list(done, src=0)
    return done[0]


def _mark_pass_done(path: str, name: str, setup: dict):
    """Record that every rank finished the pass ``name`` under ``path``."""
    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(os.path.join(path, name + DONE_SUFFIX), "w") as f:
            json.dump(setup, f)


def partition_modules(names: list[str], num_partitions: int) -> list[list[str]]:
    """Split ``names`` into ``num_partitions`` contiguous groups of near-equal size."""
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


def balanced_partitions(
    target_info: dict[str, tuple[torch.device, torch.Size, bool]],
    num_partitions: int,
) -> list[list[str]]:
    """Split the modules into ``num_partitions`` groups whose factors take about
    the same memory on each device, so a group spreads over every GPU when the
    model is split across them. Each group keeps the modules' order."""
    if num_partitions < 1:
        raise ValueError(f"module_partitions must be >= 1, got {num_partitions}")
    num_partitions = min(num_partitions, len(target_info))
    by_device = defaultdict(dict)
    for name, info in target_info.items():
        by_device[info[0]][name] = info
    group_of = {}
    for modules in by_device.values():
        group_of.update(assign_module_owners(modules, num_partitions))
    groups = [
        [name for name in target_info if group_of[name] == i]
        for i in range(num_partitions)
    ]
    return [group for group in groups if group]


def merge_files(files: list[str], out_path: str):
    """Save the tensors of every safetensors file in ``files`` to ``out_path``.

    The loaded tensors map their source files until they're freed, and on NFS
    a deleted file that's still mapped keeps its directory from being removed.
    Returning frees them before ``merge_partitions`` deletes the sources."""
    merged = {}
    with ExitStack() as stack:
        for f in files:
            handle = stack.enter_context(safe_open(f, framework="pt", device="cpu"))
            for key in handle.keys():
                merged[key] = handle.get_tensor(key)
        save_file(merged, out_path)


def merge_partitions(run_path: str | os.PathLike, num_partitions: int, rank: int):
    """Merge this rank's shard from every ``partition_{i}`` under ``run_path`` into
    the run's factor stores; rank 0 then deletes the partition directories."""
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

        os.makedirs(os.path.join(run_path, sub), exist_ok=True)
        merge_files(files, os.path.join(run_path, sub, shard))

    dist.barrier() if dist.is_initialized() else None
    if rank == 0:
        # A partition deleted only in part must not look finished to a resume.
        for d in part_dirs:
            for marker in Path(d).glob(f"*{DONE_SUFFIX}"):
                marker.unlink()
        for d in part_dirs:
            shutil.rmtree(d)


def approximate_hessians(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    *,
    do_eigendecomposition: bool = True,
    resume: bool = False,
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
    resume : bool
        Keep what an interrupted fit left in ``index_cfg.partial_run_path`` and
        skip the passes it finished, rather than starting over.

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
    # Callers may already have saved files here, such as the run's config.
    if not resume and index_cfg.distributed.rank == 0:
        for marker in index_cfg.partial_run_path.rglob(f"*{DONE_SUFFIX}"):
            marker.unlink()
        for counts in index_cfg.partial_run_path.rglob("num_positions.json"):
            counts.unlink()
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
        for marker in index_cfg.partial_run_path.glob(f"*{DONE_SUFFIX}"):
            marker.unlink()
        (index_cfg.partial_run_path / "num_positions.json").unlink(missing_ok=True)
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

    # The autocorrelation Hessian is a dense gradient Gram, one per module or
    # one over the global projection, so it computes in one pass and skips the
    # factored eigendecomposition.
    if hessian_cfg.method == "autocorrelation":
        processor = create_processor(model, index_cfg, target_modules)
        collector_cls = (
            JointAutocorrelationCollector
            if processor.projection_target == "global"
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

    groups: list[tuple[set[str] | None, str]] = [
        (target_modules, str(index_cfg.partial_run_path))
    ]
    if hessian_cfg.module_partitions > 1:
        target_info = HookCollectorBase.discover_targets(
            model.base_model,  # type: ignore
            target_modules,
            index_cfg.include_bias,
            index_cfg.filter_modules,
        )
        partitions = balanced_partitions(target_info, hessian_cfg.module_partitions)
        groups = [
            (set(group), os.path.join(index_cfg.partial_run_path, f"partition_{i}"))
            for i, group in enumerate(partitions)
        ]

    # Passed without keeping a reference here, so the fit can free the model
    # while it eigendecomposes.
    handed_over = [model]
    del model
    fit_factored_hessians(
        handed_over.pop(),
        lambda: setup_model_and_peft(index_cfg)[0],  # type: ignore
        ds,
        index_cfg,
        hessian_cfg,
        groups=groups,
        attention_cfgs=attention_cfgs,
        batches=batches,
        do_eigendecomposition=do_eigendecomposition,
    )

    if len(groups) > 1:
        rank = dist.get_rank() if dist.is_initialized() else 0
        merge_partitions(index_cfg.partial_run_path, len(groups), rank)


def fit_factored_hessians(
    model: PreTrainedModel,
    reload_model: Callable[[], PreTrainedModel],
    data: Dataset,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    *,
    groups: list[tuple[set[str] | None, str]],
    attention_cfgs: dict[str, AttentionConfig],
    batches: list[list[int]],
    do_eigendecomposition: bool,
):
    """Fit the factors of each ``(target_modules, path)`` group, writing them
    under its path, in three phases that each skip the groups they finished:

    1. Collect each group's covariances, which are saved to disk.
    2. Eigendecompose them from the files, with the model freed when
       ``hessian_cfg.free_model_for_eigendecomposition`` is set.
    3. For EK-FAC, collect each group's eigenvalue corrections.

    ``reload_model`` loads a freed model again for phase 3, which gets the
    adapter weights of ``model``, as a fresh adapter's initialization is random.
    """
    adapter = None

    def get_model() -> PreTrainedModel:
        nonlocal model
        if model is None:
            model = reload_model()
            if adapter is not None:
                set_peft_model_state_dict(model, adapter)
        return model

    def collect(modules: set[str] | None, path: str, **kwargs) -> HookCollectorBase:
        return collect_hessians(
            get_model(),
            data,
            index_cfg,
            batches=batches,
            target_modules=modules,
            attention_cfgs=(
                attention_cfgs
                if modules is None
                else {k: v for k, v in attention_cfgs.items() if k in modules}
            ),
            hessian_cfg=hessian_cfg,
            path=path,
            **kwargs,
        )

    setups = {
        path: _fit_setup(index_cfg, hessian_cfg, modules) for modules, path in groups
    }
    rank = dist.get_rank() if dist.is_initialized() else 0

    def finished(i: int, path: str, step: str) -> bool:
        """Whether group ``i`` already did ``step``, logging either way."""
        done = _pass_done(path, step, setups[path])
        if rank == 0:
            state = "already done" if done else "running"
            print(f"Module group {i + 1}/{len(groups)}, {step}: {state}")
        return done

    for i, (modules, path) in enumerate(groups):
        if finished(i, path, "covariances"):
            continue
        collector = collect(modules, path)
        if isinstance(collector, CovarianceCollector) and rank == 0:
            # An MoE expert's covariances sum only the positions routed to it.
            with open(os.path.join(path, "num_positions.json"), "w") as f:
                json.dump(collector.num_positions, f)
        del collector
        _release_device_memory()
        _mark_pass_done(path, "covariances", setups[path])

    if not do_eigendecomposition:
        return

    if hessian_cfg.free_model_for_eigendecomposition:
        if isinstance(model, PeftModel):
            adapter = {
                k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()
            }
        model = None
    _release_device_memory()
    total_processed = torch.load(
        f"{index_cfg.partial_run_path}/total_processed.pt",
        map_location="cpu",
        weights_only=False,
    )
    for i, (_, path) in enumerate(groups):
        if finished(i, path, "eigenbasis"):
            continue
        num_positions = None
        if os.path.exists(os.path.join(path, "num_positions.json")):
            with open(os.path.join(path, "num_positions.json")) as f:
                num_positions = json.load(f)
        eigenvalues = [
            compute_eigendecomposition(
                os.path.join(path, sub),
                total_processed,
                num_positions,
                # With the model split across GPUs, the fit has them all.
                all_gpus=index_cfg.distributed.device_map_auto,
            )
            for sub in ("activation_sharded", "gradient_sharded")
        ]
        save_uncorrected_eigenvalues(
            partial_run_path=path,
            eigenvalues_a=eigenvalues[0],
            eigenvalues_g=eigenvalues[1],
            total_processed=total_processed,
            rank=rank,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            num_documents=len(data),
            num_positions=num_positions,
        )
        del eigenvalues
        _release_device_memory()
        _mark_pass_done(path, "eigenbasis", setups[path])

    if not hessian_cfg.ev_correction:
        return
    for i, (modules, path) in enumerate(groups):
        if finished(i, path, "corrections"):
            continue
        collect(modules, path, ev_correction=True, num_documents=len(data))
        _release_device_memory()
        _mark_pass_done(path, "corrections", setups[path])


def _fit_setup(
    index_cfg: IndexConfig, hessian_cfg: HessianConfig, modules: set[str] | None
) -> dict:
    """What decides a group's factors, which a resumed fit must match."""
    setup = {
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "modules": sorted(modules) if modules is not None else None,
        "index_cfg": {
            k: getattr(index_cfg, k)
            for k in ("model", "revision", "precision", "filter_modules")
        },
        "data": asdict(index_cfg.data),
        # Freeing the model doesn't change the factors, so a fit that ran out of
        # memory can resume with it set.
        "hessian_cfg": {
            k: v
            for k, v in asdict(hessian_cfg).items()
            if k != "free_model_for_eigendecomposition"
        },
    }
    # As it reads back from a marker.
    return json.loads(json.dumps(setup, default=str))


def _release_device_memory():
    """Drop a finished collector's device tensors; the collector, computer and
    hooks form a reference cycle, so they otherwise outlive the pass."""
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
    num_documents: int = 1,
) -> HookCollectorBase:
    """
    Compute Hessian approximations using the hooks specified in the collector.
    If ev_correction is True, uses LambdaCollector to compute eigenvalue corrections.
    ``path`` overrides where the collector writes. Returns the finished
    collector.
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
            num_documents=num_documents,
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
    return collector
