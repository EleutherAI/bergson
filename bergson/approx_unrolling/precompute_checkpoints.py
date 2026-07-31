import shutil
from copy import deepcopy
from pathlib import Path

import torch
from datasets import Dataset

from bergson.approx_unrolling.segment_aggregation import (
    DOCUMENTS_PROCESSED_FILENAME,
)
from bergson.collector.collector import (
    CollectorComputer,
    fwd_bwd_hessian_factory,
)
from bergson.config.config import (
    ApproxUnrollingConfig,
    HessianConfig,
    IndexConfig,
)
from bergson.data import allocate_batches
from bergson.distributed import init_dist, launch_distributed_run
from bergson.hessians.eigenvectors import LambdaCollector
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.utils.logger import get_logger
from bergson.utils.utils import convert_precision_to_torch
from bergson.utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)


def precompute_checkpoint_averaged_lambdas(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    approx_unrolling_cfg: ApproxUnrollingConfig,
    *,
    resume: bool = False,
    output_subdir: str = "averaged_ev_correct_sharded",
) -> None:
    """For each (segment, ckpt), compute lambda using the segment's eigvecs."""
    logger = get_logger("precompute_checkpoint_averaged_lambdas")
    base_run = Path(index_cfg.run_path)
    method = hessian_cfg.method
    n_ckpts = len(approx_unrolling_cfg.checkpoints)
    n_segments = approx_unrolling_cfg.segments
    per_segment = n_ckpts // n_segments

    for c, ckpt in enumerate(approx_unrolling_cfg.checkpoints):
        ckpt = str(ckpt)
        seg = c // per_segment
        idx_in_seg = c % per_segment
        ckpt_method_dir = base_run / f"segment_{seg}" / f"ckpt_{idx_in_seg}" / method
        eigen_path = base_run / f"segment_{seg}" / method
        out_path = ckpt_method_dir / output_subdir

        if out_path.exists():
            if resume:
                logger.info(
                    f"[seg {seg} ckpt {idx_in_seg}] skip — exists at {out_path}"
                )
                continue
            shutil.rmtree(out_path)

        if not eigen_path.exists():
            raise FileNotFoundError(
                f"Missing segment eigvec dir {eigen_path}; did step 2 finish?"
            )

        logger.info(
            f"[seg {seg} ckpt {idx_in_seg}] computing {output_subdir} at "
            f"model={ckpt!r} using eigvecs from {eigen_path}"
        )

        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.model = ckpt
        # Per-ckpt run_path so each one gets its own partial_run_path
        # (where CollectorComputer writes total_processed.pt).
        # TODO: Fix total_processed logic — currently lands in kfac.part/.
        ckpt_index_cfg.run_path = str(ckpt_method_dir)
        ckpt_index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

        ds, _ = setup_data_pipeline(ckpt_index_cfg)

        launch_distributed_run(
            "checkpoint_averaged_lambda",
            _lambda_worker,
            [
                ckpt_index_cfg,
                hessian_cfg,
                ds,
                ckpt_method_dir,
                eigen_path,
                output_subdir,
            ],
            ckpt_index_cfg.distributed,
        )
        torch.save(torch.tensor(len(ds)), out_path / DOCUMENTS_PROCESSED_FILENAME)


def _lambda_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    ds: Dataset,
    output_dir: Path,
    eigen_path: Path,
    output_subdir: str,
) -> None:
    """Lambda-only data pass for one checkpoint, writing into ``output_dir``."""
    init_dist(rank, local_rank, world_size)

    model, target_modules = setup_model_and_peft(index_cfg)
    attention_cfgs = {m: index_cfg.attention for m in index_cfg.split_attention_modules}
    batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    collector = LambdaCollector(
        model=model.base_model,  # type: ignore
        target_modules=target_modules,
        attention_cfgs=attention_cfgs,
        path=str(output_dir),
        eigen_path=str(eigen_path),
        output_subdir=output_subdir,
        filter_modules=index_cfg.filter_modules,
        dtype=convert_precision_to_torch(hessian_cfg.hessian_dtype),
    )

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=ds,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )
    computer.forward_backward = fwd_bwd_hessian_factory(index_cfg, hessian_cfg)
    computer.run_with_collector_hooks(desc=f"Lambda -> {output_dir}")


"""Precompute Hessian factors at multiple training checkpoints.

For each entry in ``approx_unrolling_cfg.checkpoints`` (an absolute path or
HF model ID), run the existing :func:`approximate_hessians` pipeline once.
Each checkpoint's output lands at
``<index_cfg.run_path>/segment_{l}/ckpt_{i}/<method>/``, where ``l`` is the
segment the checkpoint belongs to and ``i`` is its index *within* that
segment. EV correction is forced off because per-checkpoint lambda is
wasted work — the segment-averaging step recomputes it once it knows the
segment eigenbasis.
"""


def precompute_checkpoint_hessians(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    approx_unrolling_cfg: ApproxUnrollingConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Run :func:`approximate_hessians` once per checkpoint.

    The pipeline-level divisibility check (``n_ckpts % n_segments == 0``)
    runs in :func:`approx_unrolling_pipeline` before this is called.
    """
    logger = get_logger("precompute_checkpoint_hessians")
    base_run = Path(index_cfg.run_path)
    method = hessian_cfg.method
    n_ckpts = len(approx_unrolling_cfg.checkpoints)
    n_segments = approx_unrolling_cfg.segments
    per_segment = n_ckpts // n_segments

    for c, ckpt in enumerate(approx_unrolling_cfg.checkpoints):
        ckpt = str(ckpt)
        seg = c // per_segment
        idx_in_seg = c % per_segment
        ckpt_dir = base_run / f"segment_{seg}" / f"ckpt_{idx_in_seg}"
        out_path = ckpt_dir / method

        if out_path.exists():
            if not overwrite:
                logger.info(
                    f"[seg {seg} ckpt {idx_in_seg}] skip — exists at {out_path}"
                )
                continue
            shutil.rmtree(out_path)

        logger.info(
            f"[seg {seg} ckpt {idx_in_seg}] computing {method} at model={ckpt!r}"
        )

        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.run_path = str(out_path)
        ckpt_index_cfg.model = ckpt

        ckpt_hessian_cfg = deepcopy(hessian_cfg)
        ckpt_hessian_cfg.ev_correction = False

        approximate_hessians(
            ckpt_index_cfg, ckpt_hessian_cfg, do_eigendecomposition=False
        )
