import json
import os
import re
import shutil
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from bergson.config import (
    ApproxUnrollingConfig,
    DistributedConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.data import load_scores
from bergson.distributed import launch_distributed_run
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.score.score import score_dataset
from bergson.utils.utils import get_device, get_device_index


def compute_eta_K_per_segment(
    approx_unrolling_cfg: ApproxUnrollingConfig,
) -> list[float]:
    """Per-segment lr * K. Use ``lr_list * step_size_list`` if set on config;
    else equal-partition log_history.json into segments and sum per-step LRs."""
    cfg = approx_unrolling_cfg
    L = cfg.segments
    if cfg.lr_list and cfg.step_size_list:
        return [lr * k for lr, k in zip(cfg.lr_list, cfg.step_size_list)]

    # TODO: parsing 'checkpoint-N' from dir name is fragile.
    per_segment = len(cfg.checkpoints) // L
    ckpt_steps = [
        int(re.match(r"checkpoint-(\d+)$", Path(str(p)).name).group(1))  # type: ignore
        for p in cfg.checkpoints
    ]
    boundaries = [0] + [ckpt_steps[(l + 1) * per_segment - 1] for l in range(L)]
    log_path = Path(str(cfg.checkpoints[0])).parent / "log_history.json"
    with open(log_path) as f:
        step_to_lr = {e["step"]: e["learning_rate"] for e in json.load(f)}
    return [
        sum(
            step_to_lr.get(s, 0.0)
            for s in range(boundaries[l] + 1, boundaries[l + 1] + 1)
        )
        for l in range(L)
    ]


def f_s(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> exp(-eta_K*x)."""

    def fn(sigma: Tensor) -> Tensor:
        return torch.exp(-eta_K * sigma)

    return fn


def f_r(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> (1 - exp(-eta_K*x)) / x. Limit at x=0 is eta_K."""

    def fn(sigma: Tensor) -> Tensor:
        # Compute as eta_K * ((1 - exp(-x))/x); the parenthesized ratio is in
        # [0, 1] for x ≥ 0 and uses expm1 for accuracy near zero.
        x = eta_K * sigma
        is_zero = x == 0
        x_safe = x.masked_fill(is_zero, 1.0)
        ratio = -torch.expm1(-x_safe) / x_safe
        return eta_K * ratio.masked_fill(is_zero, 1.0)

    return fn


def apply_eigfn_to_query(
    src_grad_path: Path,
    dst_grad_path: Path,
    segment_dir: Path,
    eta_K: float,
    n_seg: int,
    fn_kind: str,
    distributed: DistributedConfig,
) -> None:
    """Apply F_r or F_S of one segment to a stored query gradient.

    ``fn_kind`` is "f_r" or "f_s". lambda is normalized by ``n_seg`` inside the
    worker (sum-of-squares -> expected eigenvalue) before fn is applied.
    """
    cfg = EkfacConfig(
        hessian_method_path=str(segment_dir),
        gradient_path=str(src_grad_path),
        run_path=str(dst_grad_path),
        ev_correction=True,
    )
    launch_distributed_run(
        "apply_eigfn_to_query",
        _apply_eigfn_worker,
        [cfg, eta_K, n_seg, fn_kind],
        distributed,
    )


def _apply_eigfn_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
    eta_K: float,
    n_seg: int,
    fn_kind: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(local_rank))
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

    base_fn = {"f_r": f_r, "f_s": f_s}[fn_kind](eta_K)
    fn = lambda x: base_fn(x / n_seg)  # noqa: E731
    EkfacApplicator(cfg, apply_fn=fn).compute_ivhp_sharded()


def walk_query_phase1(
    run_path: str | Path,
    method: str,
    eta_K_per_segment: list[float],
    distributed: DistributedConfig,
) -> list[Path]:
    """Phase 1: build u_0, u_1, ..., u_{L-1} by walking F_S backwards.

    u_{L-1} is the original query at <run>/query/.
    u_{k-1} = F_S(segment_k) applied to u_k for k = L-1, L-2, ..., 1.
    Outputs land at <run>/segment_{l}/u/ for l = 0 .. L-2.

    Returns ``[u_0_path, u_1_path, ..., u_{L-1}_path]``.
    """
    base = Path(run_path)
    L = len(eta_K_per_segment)
    u_paths: list[Path] = [Path("")] * L
    u_paths[L - 1] = base / "query"

    for k in range(L - 1, 0, -1):
        segment_dir = base / f"segment_{k}" / method
        dst = base / f"segment_{k - 1}" / "u"
        apply_eigfn_to_query(
            src_grad_path=u_paths[k],
            dst_grad_path=dst,
            segment_dir=segment_dir,
            eta_K=eta_K_per_segment[k],
            n_seg=_load_n_seg(segment_dir),
            fn_kind="f_s",
            distributed=distributed,
        )
        u_paths[k - 1] = dst

    return u_paths


def walk_query_phase2(
    run_path: str | Path,
    method: str,
    eta_K_per_segment: list[float],
    u_paths: list[Path],
    distributed: DistributedConfig,
) -> list[Path]:
    """Phase 2: build psi_0, psi_1, ..., psi_{L-1} via F_r per segment.

    psi_l = F_r(segment_l) applied to u_l for l = 0, 1, ..., L-1.
    Outputs land at <run>/segment_{l}/psi/.
    Global (1/N_train) factor is deferred to scoring time.

    Returns ``[psi_0_path, psi_1_path, ..., psi_{L-1}_path]``.
    """
    base = Path(run_path)
    L = len(eta_K_per_segment)
    psi_paths: list[Path] = []

    for l in range(L):
        segment_dir = base / f"segment_{l}" / method
        dst = base / f"segment_{l}" / "psi"
        apply_eigfn_to_query(
            src_grad_path=u_paths[l],
            dst_grad_path=dst,
            segment_dir=segment_dir,
            eta_K=eta_K_per_segment[l],
            n_seg=_load_n_seg(segment_dir),
            fn_kind="f_r",
            distributed=distributed,
        )
        psi_paths.append(dst)

    return psi_paths


def _load_n_seg(segment_dir: Path) -> int:
    return int(
        torch.load(
            segment_dir / "total_processed.pt",
            map_location="cpu",
            weights_only=False,
        ).item()
    )


def score_per_segment_and_aggregate(
    index_cfg: IndexConfig,
    psi_paths: list[Path],
    final_checkpoint: str,
) -> Path:
    """Phase 3: per-segment ``psi_l . g(z_m)`` scores, summed into one final.

    For each l, runs :func:`score_dataset` against the training data at the
    final checkpoint with ``psi_l`` as the query. Writes per-segment outputs
    to ``<run>/segment_{l}/scores/``, then sums into ``<run>/scores.npy``.
    """
    base_run = Path(index_cfg.run_path)
    L = len(psi_paths)
    score_dirs: list[Path] = []
    for l in range(L):
        scores_dir = base_run / f"segment_{l}" / "scores"
        if scores_dir.exists():
            shutil.rmtree(scores_dir)
        seg_index_cfg = deepcopy(index_cfg)
        seg_index_cfg.model = final_checkpoint
        seg_index_cfg.run_path = str(scores_dir)
        seg_index_cfg.projection_dim = 0
        seg_index_cfg.skip_hessians = True
        score_cfg = ScoreConfig(query_path=str(psi_paths[l]))
        score_dataset(seg_index_cfg, score_cfg, PreprocessConfig())
        score_dirs.append(scores_dir)

    total = load_scores(score_dirs[0]).get(slice(None), 0)
    for scores_dir in score_dirs[1:]:
        total = total + load_scores(scores_dir).get(slice(None), 0)
    out_path = base_run / "scores.npy"
    np.save(out_path, total)
    return out_path
