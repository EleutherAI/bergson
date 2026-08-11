"""Per-step attribution-score magnitude, plotted against training step.

A MAGIC run takes the shuffled training documents ``batch_size`` at a time, so
grouping the saved scores into ``batch_size``-row blocks gives one value per
optimizer step. ``bergson score_trajectory <run>`` plots the batch-median of
``log10|score|`` -- the score *level* -- against step and writes
``score_vs_step.png``. The level can sweep many orders of magnitude when a run
is not healthy; see the MAGIC docs.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import yaml

from ..data import load_scores_loss_signed


def batch_size_from_config(run_path: str) -> int:
    """Read ``batch_size`` from a run's ``config.yaml`` (``steps[0]``)."""
    with open(Path(run_path) / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    step = next(iter(cfg["steps"][0].values()))
    return int(step["batch_size"])


def per_step_level(scores: np.ndarray, batch_size: int) -> np.ndarray:
    """Per-step median ``log10|score|``.

    ``scores`` is ``(n_rows, ...)`` in training order; the leading axis is the
    document/chunk axis and any trailing axes (token position, query) are pooled
    into the step. Step ``s`` is rows ``[s*bs : (s+1)*bs)``. A step with no
    finite non-zero score comes back ``NaN``.
    """
    a = np.abs(np.asarray(scores, dtype=np.float64))
    ns = a.shape[0] // batch_size
    a = a[: ns * batch_size].reshape(ns, -1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # An all-NaN step (a MAGIC backward leaves early steps undefined) is
        # expected, not an error; nanmedian warns on it -- silence just that.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(
            np.where(np.isfinite(a) & (a > 0), np.log10(a), np.nan), axis=1
        )


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    half = w // 2
    pad = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def _fill_and_smooth(lvl_raw: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    """Forward-fill dead steps, back-fill a leading dead run, rolling-median.

    Returns ``(level, n_leading_dead)`` -- the curve step-normalisation would
    divide by, and how many all-NaN steps lead the run.
    """
    fin = np.isfinite(lvl_raw)
    idx = np.where(fin, np.arange(len(lvl_raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    lvl = lvl_raw[idx]
    first = int(np.argmax(fin)) if fin.any() else 0
    if first:
        lvl[:first] = lvl[first]
    return _rolling_median(lvl, window), first


def plot_score_trajectory(
    run_path: str, *, window: int = 0, out: str | None = None
) -> str:
    """Write ``<run_path>/score_vs_step.png`` for a finished MAGIC run.

    Reads ``<run_path>/scores`` and ``<run_path>/config.yaml``. With
    ``window > 1`` also overlays the window-``N`` step-normalisation curve and
    the residual (post-normalisation) level, which should sit near 0. Returns
    the output path.
    """
    # matplotlib is an optional (``bergson[viz]``) dependency, so it is imported
    # here rather than at module top: importing this module must not require it.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise ImportError(
            "score_trajectory needs matplotlib; install it with "
            "`pip install matplotlib` or `pip install 'bergson[viz]'`."
        ) from e

    run_path = str(run_path)
    scores, _ = load_scores_loss_signed(os.path.join(run_path, "scores"))
    scores = np.asarray(scores.detach().cpu().numpy(), dtype=np.float64)
    batch_size = batch_size_from_config(run_path)

    lvl_raw = per_step_level(scores, batch_size)
    ns = len(lvl_raw)
    if ns == 0:
        raise ValueError(
            f"{run_path}/scores has fewer than batch_size={batch_size} rows; "
            "nothing to plot."
        )
    live = int(np.isfinite(lvl_raw).sum())
    if live == 0:
        raise ValueError(f"{run_path}/scores is all zero / non-finite.")
    steps = np.arange(ns)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(
        steps,
        lvl_raw,
        s=4,
        c="#9ecae1",
        alpha=0.55,
        label="per-step level (batch-median log10|score|)",
    )

    if window and window > 1:
        lvl_used, dead_lead = _fill_and_smooth(lvl_raw, window)
        a = np.abs(scores[: ns * batch_size].reshape(ns, -1))
        with np.errstate(all="ignore"):
            post = np.nanmedian(
                np.where(
                    np.isfinite(a) & (a > 0),
                    np.log10(a) - lvl_used[:, None],
                    np.nan,
                ),
                axis=1,
            )
        ax.plot(
            steps,
            lvl_used,
            c="#08519c",
            lw=1.6,
            label=f"normalisation curve (filled + window-{window})",
        )
        ax.plot(
            steps,
            post,
            c="#e6550d",
            lw=1.0,
            alpha=0.85,
            label="post-normalisation level (target ~0)",
        )
        ax.axhline(0, c="grey", lw=0.6, ls=":")
        if dead_lead:
            ax.axvspan(0, dead_lead, color="red", alpha=0.08)
            ax.text(
                dead_lead,
                ax.get_ylim()[1],
                f"  {dead_lead} leading all-NaN steps (back-filled)",
                va="top",
                ha="left",
                fontsize=8,
                color="#a50f15",
            )

    rng = float(np.nanmax(lvl_raw) - np.nanmin(lvl_raw))
    name = os.path.basename(os.path.normpath(run_path))
    ax.set_title(
        f"{name}   batch_size={batch_size}, {ns} steps, "
        f"{100 * live / ns:.1f}% live, {rng:.1f} decades of range"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("level = log10|attribution score|  (dex)")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = out or os.path.join(run_path, "score_vs_step.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
