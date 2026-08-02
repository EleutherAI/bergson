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
    InversionConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config_io import load_subconfig, save_run_config
from bergson.data import load_scores
from bergson.distributed import init_dist, launch_distributed_run
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.score.score import score_dataset
from bergson.score.score_writer import save_sequence_scores, save_token_scores

from ..magic.trainer import LR_HISTORY_FILENAME
from .train_cfg_io import lr_history_path


def _checkpoint_step(p: str) -> int:
    """Extract the training step index for a checkpoint.

    Accepts a path whose final component is a ``checkpoint-<N>`` directory
    (HF Trainer), a ``step_<N>[.ckpt]`` directory (Bergson trainer), or a
    bare ``<N>`` step directory.
    """
    # TODO: Inferring the step from a checkpoint name via regex is brittle.
    # For now, we raise an error
    name = Path(p).name
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"step_(\d+)(?:\.ckpt)?$", name)
    if m:
        return int(m.group(1))
    if name.isdigit():
        return int(name)
    raise ValueError(
        f"Cannot infer a training step from checkpoint {p!r}: expected a path "
        "ending in 'checkpoint-<N>', 'step_<N>[.ckpt]', or a bare '<N>' step "
        "directory. Set "
        "`lr_list` and `step_size_list` on ApproxUnrollingConfig to specify "
        "the per-segment learning rate and step counts explicitly instead of "
        "inferring them from checkpoint names."
    )


def compute_lr_times_steps_per_segment(
    cfg: ApproxUnrollingConfig,
) -> list[float]:
    """Per-segment lr * K. Use ``lr_list * step_size_list`` if set on config;
    else equal-partition log_history.json into segments and sum per-step LRs.

    With SGD heavy-ball ``momentum`` beta, scale by the terminal velocity
    1/(1-beta) (Bae et al. 2024, App. D.2).
    """
    L = cfg.segments

    momentum = cfg.momentum if cfg.momentum is not None else 0.0
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    momentum_scale = 1.0 / (1.0 - momentum)
    if cfg.lr_list and cfg.step_size_list:
        return [
            lr * k * momentum_scale for lr, k in zip(cfg.lr_list, cfg.step_size_list)
        ]

    per_segment = len(cfg.checkpoints) // L
    ckpt_steps = [_checkpoint_step(p) for p in cfg.checkpoints]
    boundaries = [0] + [ckpt_steps[(l + 1) * per_segment - 1] for l in range(L)]
    # The bergson run these came from, else a history dumped beside them,
    # else the final checkpoint's trainer_state.json (what HF Trainer writes).
    log_path = lr_history_path(cfg.checkpoints) or (
        Path(str(cfg.checkpoints[0])).parent / LR_HISTORY_FILENAME
    )
    if log_path.exists():
        with open(log_path) as f:
            log_history = json.load(f)
    else:
        ts_path = Path(str(cfg.checkpoints[-1])) / "trainer_state.json"
        with open(ts_path) as f:
            log_history = json.load(f)["log_history"]
    step_to_lr = {e["step"]: e["learning_rate"] for e in log_history}
    return [
        momentum_scale
        * sum(
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


def f_one_minus_exp(lr_times_steps: float) -> Callable[[Tensor], Tensor]:
    """x -> 1 - exp(-lr_times_steps*x), the numerator of :func:`f_segment`.

    Used by the Eq-43 hybrid: the 1/x that completes f_segment is supplied by
    the EK-FAC inverse applied afterwards rather than evaluated on the diagonal.
    """

    def fn(sigma: Tensor) -> Tensor:
        return -torch.expm1(-lr_times_steps * sigma)

    return fn


def apply_eigfn_to_query(
    src_grad_path: Path,
    dst_grad_path: Path,
    segment_dir: Path,
    lr_times_steps: float,
    fn_kind: str,
    distributed: DistributedConfig,
    preconditioner_path: str = "",
    inversion_cfg: InversionConfig | None = None,
) -> None:
    """Apply F_segment or F_backward of one segment to a stored query gradient.

    ``preconditioner_path`` selects the optimizer-preconditioned variant."""
    cfg = EkfacConfig(
        hessian_method_path=str(segment_dir),
        gradient_path=str(src_grad_path),
        run_path=str(dst_grad_path),
        ev_correction=True,
        preconditioner_path=preconditioner_path,
    )
    launch_distributed_run(
        "apply_eigfn_to_query",
        _apply_eigfn_worker,
        # F_segment is the Eq-43 hybrid: diagonal exponential, EK-FAC H^-1.
        [
            cfg,
            lr_times_steps,
            fn_kind,
            inversion_cfg if fn_kind == "f_segment" else None,
        ],
        distributed,
    )


def _apply_eigfn_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
    lr_times_steps: float,
    fn_kind: str,
    inversion_cfg: InversionConfig | None,
) -> None:
    init_dist(rank, local_rank, world_size)

    # Segment eigenvalues are already checkpoint-averaged, so the eigenfunction
    # is applied to them directly (no per-example normalization).
    if cfg.preconditioner_path and inversion_cfg is not None:
        # The EK-FAC inverse applied afterwards supplies the 1/x, so
        # multiply by the numerator only.
        fn = f_one_minus_exp(lr_times_steps)
    else:
        fn = {"f_segment": f_segment, "f_backward": f_backward}[fn_kind](lr_times_steps)
    EkfacApplicator(
        cfg, inversion_cfg=inversion_cfg, apply_fn=fn
    ).compute_ivhp_sharded()


def walk_query_phase1(
    run_path: str | Path,
    method: str,
    lr_times_steps_per_segment: list[float],
    distributed: DistributedConfig,
    preconditioner_paths: list[str] | None = None,
    inversion_cfg: InversionConfig | None = None,
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
            preconditioner_path=preconditioner_paths[k] if preconditioner_paths else "",
        )
        query_grad_paths[k - 1] = dst

    return query_grad_paths


def walk_query_phase2(
    run_path: str | Path,
    method: str,
    lr_times_steps_per_segment: list[float],
    query_grad_paths: list[Path],
    distributed: DistributedConfig,
    preconditioner_paths: list[str] | None = None,
    inversion_cfg: InversionConfig | None = None,
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
            preconditioner_path=preconditioner_paths[l] if preconditioner_paths else "",
        )
        query_grad_segment_paths.append(dst)

    return query_grad_segment_paths


def score_per_segment_and_aggregate(
    index_cfg: IndexConfig,
    query_grad_segment_paths: list[Path],
    segment_checkpoints: list[list[str]],
    query_batch_size: int | None = None,
) -> Path:
    """Phase 3: per-segment ``query_grad_segment_l . g_bar_l(z_m)`` scores, summed.

    ``g_bar_l`` is the segment's expected training gradient, estimated over the
    segment's checkpoints (Bae et al. 2024, Eq. 20). Scores are linear in the
    training gradient, so the per-checkpoint scores are averaged; they land at
    ``<run>/segment_{l}/scores_ckpt_{c}/``.
    """
    base_run = Path(index_cfg.run_path)
    score_dirs: list[Path] = []

    def _oriented(scores_dir: Path):
        scores = load_scores(scores_dir)[:]
        score_cfg = load_subconfig(scores_dir, "score_cfg", ScoreConfig)
        if score_cfg is None:
            raise FileNotFoundError(
                f"No score_cfg found at {scores_dir}; cannot determine the "
                "scores' orientation for aggregation."
            )
        return -scores if score_cfg.higher_is_better else scores

    total = None
    for l, ckpts in enumerate(segment_checkpoints):
        seg_total = None
        for c, ckpt in enumerate(ckpts):
            scores_dir = base_run / f"segment_{l}" / f"scores_ckpt_{c}"
            if index_cfg.distributed._node_rank == 0 and scores_dir.exists():
                shutil.rmtree(scores_dir)
            seg_index_cfg = deepcopy(index_cfg)
            seg_index_cfg.model = ckpt
            seg_index_cfg.run_path = str(scores_dir)
            seg_index_cfg.projection_dim = 0
            score_cfg = ScoreConfig(
                query_path=str(query_grad_segment_paths[l]),
                higher_is_better=True,
                query_batch_size=query_batch_size,
            )
            seg_preprocess_cfg = PreprocessConfig()
            save_run_config(
                Score(score_cfg, seg_index_cfg, seg_preprocess_cfg),
                seg_index_cfg.partial_run_path,
            )
            score_dataset(seg_index_cfg, score_cfg, seg_preprocess_cfg)
            score_dirs.append(scores_dir)

            ckpt_scores = _oriented(scores_dir)
            seg_total = ckpt_scores if seg_total is None else seg_total + ckpt_scores

        assert seg_total is not None, "each segment has >= 1 checkpoint"
        seg_scores = seg_total / len(ckpts)
        total = seg_scores if total is None else total + seg_scores
    assert total is not None, "num_segments >= 1 is validated by the pipeline"

    out_path = base_run / "scores"
    if index_cfg.attribute_tokens:
        offsets = np.load(score_dirs[0] / "offsets.npy")
        save_token_scores(out_path, total, offsets)
    else:
        save_sequence_scores(out_path, total)

    out_index_cfg = deepcopy(index_cfg)
    out_index_cfg.run_path = str(out_path)
    save_run_config(
        Score(ScoreConfig(higher_is_better=False), out_index_cfg, PreprocessConfig()),
        out_path,
    )
    return out_path
