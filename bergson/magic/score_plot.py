"""Plot the magnitude of MAGIC attribution scores against training step."""

import warnings
from pathlib import Path

import numpy as np


def per_step_level(scores: np.ndarray, batch_size: int) -> np.ndarray:
    """Return the median ``log10|score|`` of each optimizer step.

    A run consumes the shuffled training documents ``batch_size`` at a time, so
    rows ``[s * batch_size, (s + 1) * batch_size)`` are the batch step ``s``
    trained on. Trailing axes (token position, query) are pooled into the step,
    a trailing partial step is dropped, and a step with no finite nonzero score
    comes back NaN.
    """
    a = np.abs(np.asarray(scores, dtype=np.float64))
    num_steps = a.shape[0] // batch_size
    a = a[: num_steps * batch_size].reshape(num_steps, -1)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A step with no score at all is expected, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(
            np.where(np.isfinite(a) & (a > 0), np.log10(a), np.nan), axis=1
        )


def plot_score_trajectory(
    scores: np.ndarray, batch_size: int, out: Path
) -> Path | None:
    """Write the per-step score level to ``out``, or return None if it can't be.

    The scores are already saved by the time this runs, so a missing matplotlib
    or a failed plot must not take the run down with it.
    """
    try:
        from matplotlib.figure import Figure

        level = per_step_level(scores, batch_size)
        if not np.isfinite(level).any():
            return None

        fig = Figure(figsize=(9, 4))
        ax = fig.subplots()
        ax.scatter(np.arange(len(level)), level, s=4)
        ax.set_xlabel("training step")
        ax.set_ylabel("median log10|score|")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        return out
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"Score plot failed ({type(e).__name__}: {e}); scores are saved.")
        return None
