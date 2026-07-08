import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from bergson.cli.commands import Score
from bergson.config.config import (
    ApproxUnrollingConfig,
    DistributedConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config_io import save_run_config
from bergson.data import load_scores
from bergson.distributed import init_dist, launch_distributed_run
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.score.score import score_dataset


def compute_lr_times_steps_per_segment(
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
    # Prefer log_history.json if dumped at the parent dir; otherwise pull
    # log_history out of the final checkpoint's trainer_state.json (what HF
    # Trainer writes natively).
    parent = Path(str(cfg.checkpoints[0])).parent
    log_path = parent / "log_history.json"
    if log_path.exists():
        with open(log_path) as f:
            log_history = json.load(f)
    else:
        ts_path = Path(str(cfg.checkpoints[-1])) / "trainer_state.json"
        with open(ts_path) as f:
            log_history = json.load(f)["log_history"]
    step_to_lr = {e["step"]: e["learning_rate"] for e in log_history}
    return [
        sum(
            step_to_lr.get(s, 0.0)
            for s in range(boundaries[l] + 1, boundaries[l + 1] + 1)
        )
        for l in range(L)
    ]


def f_backward(lr_times_steps: float) -> Callable[[Tensor], Tensor]:
    """x -> exp(-lr_times_steps*x). This allows us to approximate the
    back propagated query gradient."""

    def fn(sigma: Tensor) -> Tensor:
        return torch.exp(-lr_times_steps * sigma)

    return fn


def f_segment(lr_times_steps: float) -> Callable[[Tensor], Tensor]:
    """x -> (1 - exp(-lr_times_steps*x)) / x. Limit at x=0 is lr_times_steps.
    This allows us to approximate the segment-wise contribution to the query
    over multiple checkpoints within a segment."""

    def fn(sigma: Tensor) -> Tensor:
        # Compute as lr_times_steps * ((1 - exp(-x))/x); the parenthesized ratio is in
        # [0, 1] for x ≥ 0 and uses expm1 for accuracy near zero.
        x = lr_times_steps * sigma
        is_zero = x == 0
        x_safe = x.masked_fill(is_zero, 1.0)
        ratio = -torch.expm1(-x_safe) / x_safe
        return lr_times_steps * ratio.masked_fill(is_zero, 1.0)

    return fn


def apply_eigfn_to_query(
    src_grad_path: Path,
    dst_grad_path: Path,
    segment_dir: Path,
    lr_times_steps: float,
    fn_kind: str,
    distributed: DistributedConfig,
) -> None:
    """Apply F_segment or F_backward of one segment to a stored query gradient.

    ``fn_kind`` is "f_segment" or "f_backward". The segment eigenvalues are
    already checkpoint-averaged (expected eigenvalues), so the eigenfunction is
    applied to them directly.
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
        [cfg, lr_times_steps, fn_kind],
        distributed,
    )


def _apply_eigfn_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
    lr_times_steps: float,
    fn_kind: str,
) -> None:
    init_dist(rank, local_rank, world_size)

    fn = {"f_segment": f_segment, "f_backward": f_backward}[fn_kind](lr_times_steps)
    EkfacApplicator(cfg, apply_fn=fn).compute_ivhp_sharded()


def walk_query_phase1(
    run_path: str | Path,
    method: str,
    lr_times_steps_per_segment: list[float],
    distributed: DistributedConfig,
) -> list[Path]:
    """Phase 1: build query_grad_0, ..., query_grad_{L-1} by walking F_backward.

    query_grad_{L-1} is the original query at <run>/query/.
    query_grad_{k-1} = F_backward(segment_k) applied to query_grad_k for
    k = L-1, ..., 1. Outputs land at <run>/segment_{l}/query_grad_backward/ for
    l = 0 .. L-2.

    Returns ``[query_grad_0_path, ..., query_grad_{L-1}_path]``.
    """
    base = Path(run_path)
    num_segments = len(lr_times_steps_per_segment)
    query_grad_paths: list[Path] = [Path("")] * num_segments
    query_grad_paths[num_segments - 1] = base / "query"

    for k in range(num_segments - 1, 0, -1):
        segment_dir = base / f"segment_{k}" / method
        dst = base / f"segment_{k - 1}" / "query_grad_backward"
        apply_eigfn_to_query(
            src_grad_path=query_grad_paths[k],
            dst_grad_path=dst,
            segment_dir=segment_dir,
            lr_times_steps=lr_times_steps_per_segment[k],
            fn_kind="f_backward",
            distributed=distributed,
        )
        query_grad_paths[k - 1] = dst

    return query_grad_paths


def walk_query_phase2(
    run_path: str | Path,
    method: str,
    lr_times_steps_per_segment: list[float],
    query_grad_paths: list[Path],
    distributed: DistributedConfig,
) -> list[Path]:
    """Phase 2: build query_grad_segment_0, ..., query_grad_segment_{L-1} via F_segment.

    query_grad_segment_l = F_segment(segment_l) applied to query_grad_l for
    l = 0, ..., L-1. Outputs land at <run>/segment_{l}/query_grad_segment/.
    Global (1/N_train) factor is deferred to scoring time.

    Returns ``[query_grad_segment_0_path, ..., query_grad_segment_{L-1}_path]``.
    """
    base = Path(run_path)
    num_segments = len(lr_times_steps_per_segment)
    query_grad_segment_paths: list[Path] = []

    for l in range(num_segments):
        segment_dir = base / f"segment_{l}" / method
        dst = base / f"segment_{l}" / "query_grad_segment"
        apply_eigfn_to_query(
            src_grad_path=query_grad_paths[l],
            dst_grad_path=dst,
            segment_dir=segment_dir,
            lr_times_steps=lr_times_steps_per_segment[l],
            fn_kind="f_segment",
            distributed=distributed,
        )
        query_grad_segment_paths.append(dst)

    return query_grad_segment_paths


def score_per_segment_and_aggregate(
    index_cfg: IndexConfig,
    query_grad_segment_paths: list[Path],
    final_checkpoint: str,
) -> Path:
    """Phase 3: per-segment ``query_grad_segment_l . g(z_m)`` scores, summed.

    For each l, runs :func:`score_dataset` against the training data at the
    final checkpoint with ``query_grad_segment_l`` as the query. Writes
    per-segment outputs to ``<run>/segment_{l}/scores/``, then sums into
    ``<run>/scores.npy``.
    """
    base_run = Path(index_cfg.run_path)
    num_segments = len(query_grad_segment_paths)
    score_dirs: list[Path] = []
    for l in range(num_segments):
        scores_dir = base_run / f"segment_{l}" / "scores"
        if scores_dir.exists():
            shutil.rmtree(scores_dir)
        seg_index_cfg = deepcopy(index_cfg)
        seg_index_cfg.model = final_checkpoint
        seg_index_cfg.run_path = str(scores_dir)
        seg_index_cfg.projection_dim = 0
        score_cfg = ScoreConfig(query_path=str(query_grad_segment_paths[l]))
        seg_preprocess_cfg = PreprocessConfig()
        save_run_config(
            Score(score_cfg, seg_index_cfg, seg_preprocess_cfg),
            seg_index_cfg.partial_run_path,
        )
        score_dataset(seg_index_cfg, score_cfg, seg_preprocess_cfg)
        score_dirs.append(scores_dir)

    total = load_scores(score_dirs[0]).get(slice(None), 0)
    for scores_dir in score_dirs[1:]:
        total = total + load_scores(scores_dir).get(slice(None), 0)
    out_path = base_run / "scores.npy"
    np.save(out_path, total)
    return out_path
