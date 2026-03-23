import os
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel

from bergson.collector.collector import (
    CollectorComputer,
    fwd_bwd_hessian_factory,
)
from bergson.config import AttentionConfig, HessianConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.distributed import launch_distributed_run
from bergson.hessians.eigenvectors import LambdaCollector, compute_eigendecomposition
from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.shampoo import ShampooCollector
from bergson.hessians.tkfac import TraceCovarianceCollector
from bergson.utils.utils import (
    convert_precision_to_torch,
    setup_reproducibility,
)
from bergson.utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)

HESSIAN_APPROXIMATIONS = {
    "kfac": CovarianceCollector,
    "tkfac": TraceCovarianceCollector,
    "shampoo": ShampooCollector,
}


def approximate_hessians(index_cfg: IndexConfig, hessian_cfg: HessianConfig) -> str:
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

    Returns
    -------
    str
        Path to the directory containing the computed Hessian approximations.
    """
    if index_cfg.debug:
        setup_reproducibility()
    index_cfg.run_path = index_cfg.run_path + f"/{hessian_cfg.method}"
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    # Save both configs
    index_cfg.save_json(index_cfg.partial_run_path / "index_config.json")
    hessian_cfg.save_json(index_cfg.partial_run_path / "hessian_config.json")

    ds, _ = setup_data_pipeline(index_cfg)

    launch_distributed_run(
        "hessian",
        hessian_worker,
        [index_cfg, hessian_cfg, ds],
        index_cfg.distributed,
    )

    rank = index_cfg.distributed.rank
    if rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)

    return index_cfg.run_path


def hessian_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    ds: Dataset,
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
    cfg : IndexConfig
        Configuration for model, data, and gradient collection.
    hessian_cfg : HessianConfig
        Configuration for Hessian approximation method (kfac or ekfac).
    ds : Dataset
        Dataset to use for covariance estimation.
    """
    """
    Build worker executed per rank to collect gradients to populate the index.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    local_rank : int
        Local rank / GPU ID for this worker on the node.
    world_size : int
        Total number of workers participating in the run.
    cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    """
    if torch.cuda.is_available():
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

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    kwargs = {
        "model": model,
        "data": ds,
        "index_cfg": index_cfg,
        "hessian_cfg": hessian_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
    kwargs["batches"] = batches
    collect_hessians(**kwargs)

    dist.barrier() if dist.is_initialized() else None

    total_processed = torch.load(
        f"{index_cfg.partial_run_path}/total_processed.pt",
        map_location="cpu",
        weights_only=False,
    )

    compute_eigendecomposition(
        os.path.join(index_cfg.partial_run_path, "activation_sharded"),
        total_processed=total_processed,
    )
    compute_eigendecomposition(
        os.path.join(index_cfg.partial_run_path, "gradient_sharded"),
        total_processed=total_processed,
    )

    dist.barrier() if dist.is_initialized() else None

    if hessian_cfg.ev_correction:
        collect_hessians(**kwargs, ev_correction=True)


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
):
    """
    Compute Hessian approximations using the hooks specified in the collector.
    If ev_correction is True, uses LambdaCollector to compute eigenvalue corrections.
    """

    hessian_dtype = (
        model.dtype
        if hessian_cfg.hessian_dtype == "auto"
        else convert_precision_to_torch(hessian_cfg.hessian_dtype)
    )

    collector_args = {
        "model": model.base_model,  # type: ignore
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs or {},
        "path": str(index_cfg.partial_run_path),
        "filter_modules": index_cfg.filter_modules,
    }
    desc = f"Approximating Hessians with {hessian_cfg.method}"
    if ev_correction:
        collector = LambdaCollector(**collector_args)
        desc += " (eigenvalue correction)"
    else:
        collector_args["dtype"] = hessian_dtype
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
