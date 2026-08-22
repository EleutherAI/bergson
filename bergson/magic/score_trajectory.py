"""Per-step attribution-score magnitude, plotted against training step.

A MAGIC run takes the shuffled training documents ``batch_size`` at a time, so
grouping the scores into ``batch_size``-row blocks gives one value per optimizer
step. Every run that saves scores also writes ``score_vs_step.png`` beside them:
the batch-median of ``log10|score|`` -- the score *level* -- against step. The
level can sweep many orders of magnitude when a run is not healthy; see the
MAGIC docs.
"""

import importlib.util
import os
import warnings

import numpy as np


def has_matplotlib() -> bool:
    """Whether the optional plotting dependency is importable."""
    return importlib.util.find_spec("matplotlib") is not None


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


def plot_score_trajectory(scores: np.ndarray, batch_size: int, out: str) -> str | None:
    """Write ``out``, the per-step score level against training step.

    Returns the path written, or ``None`` when matplotlib is missing or the
    scores hold nothing to plot -- the scores themselves are already saved, so a
    run must not fail here.
    """
    if not has_matplotlib():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lvl = per_step_level(scores, batch_size)
    finite = np.isfinite(lvl)
    ns, live = len(lvl), int(finite.sum())
    if not live:
        return None
    steps = np.arange(ns)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(
        steps,
        lvl,
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

    rng = float(np.nanmax(lvl) - np.nanmin(lvl))
    name = os.path.basename(os.path.dirname(os.path.normpath(out)))
    ax.set_title(
        f"{name}   batch_size={batch_size}, {ns} steps, "
        f"{100 * live / ns:.1f}% live, {rng:.1f} decades of range"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("level = log10|attribution score|  (dex)")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
