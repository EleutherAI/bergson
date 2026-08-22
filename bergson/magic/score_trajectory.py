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

import numpy as np

from ..config.config_io import read_config
from ..data import load_scores_loss_signed


def batch_size_from_config(run_path: str) -> int:
    """Read the MAGIC ``batch_size`` from a run's ``config.yaml``.

    Every step is searched, and a ``magic`` step wins: a run directory written
    by a pipeline holds the whole pipeline's steps, so the first one need not be
    the step that produced the scores.
    """
    steps = read_config(run_path)["steps"]
    fallback = None
    for step in steps:
        for name, cmd_dict in step.items():
            batch_size = (cmd_dict or {}).get("batch_size")
            if batch_size is None:
                continue
            if name.lower() == "magic":
                return int(batch_size)
            if fallback is None:
                fallback = int(batch_size)
    if fallback is None:
        raise ValueError(
            f"No step in {run_path}/config.yaml has a `batch_size`; run_path "
            "must point at a finished MAGIC run directory."
        )
    return fallback


def per_step_level(scores: np.ndarray, batch_size: int) -> np.ndarray:
    """Per-step median ``log10|score|``.

    ``scores`` is ``(n_rows, ...)`` in training order; the leading axis is the
    document/chunk axis and any trailing axes (token position, query) are pooled
    into the step. Step ``s`` is rows ``[s*bs : (s+1)*bs)``; a trailing partial
    step is dropped. A step with no finite non-zero score comes back ``NaN``.
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


def plot_score_trajectory(run_path: str, *, out: str | None = None) -> str:
    """Write ``<run_path>/score_vs_step.png`` for a finished MAGIC run.

    Reads ``<run_path>/scores`` and ``<run_path>/config.yaml``. Returns the
    output path.
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
    finite = np.isfinite(lvl_raw)
    live = int(finite.sum())
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

    # A MAGIC backward can leave the leading steps with no score at all, which
    # the scatter shows only as absence; mark the run so it reads as a gap.
    dead_lead = int(np.argmax(finite))
    if dead_lead:
        ax.axvspan(0, dead_lead, color="red", alpha=0.08)
        ax.text(
            dead_lead,
            ax.get_ylim()[1],
            f"  {dead_lead} steps with no score (all-NaN)",
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
