"""PAC labeling (arXiv 2506.10908): certify how much of a dataset a cheap
labeler may label.

Given cheap labels ``Y_hat`` with uncertainty scores ``U`` and an expensive
expert labeler, the procedure expert-labels a uniform random sample of size
``m``, forms an upper confidence bound on the cheap labeler's cumulative loss
below each uncertainty threshold, and returns the smallest threshold ``u_hat``
whose bound exceeds the target ``eps``. Items with ``U >= u_hat`` get expert
labels and the rest keep the cheap label, so the average loss against the
expert labels is at most ``eps`` with probability at least ``1 - alpha``. The
loss must be bounded in ``[0, 1]``. Nothing is assumed about ``U``: a poor
uncertainty score saves fewer expert labels and keeps the guarantee.

The bound is the betting confidence sequence of Waudby-Smith & Ramdas
(arXiv 2010.09686) by default. The threshold search tests ``eps`` directly
against the capital process at each threshold, which is valid without the
bound being an interval and costs ``O(m)`` per threshold.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

BoundMethod = Literal["betting", "clt", "hoeffding"]


def _betting_lambdas(z: NDArray, alpha: float, c: float = 0.75) -> NDArray:
    """Predictable plug-in bets for the WSR capital process of ``z``.

    ``lambda_t`` uses the running mean and variance of ``z[:t-1]`` and is
    capped at ``c``; the per-``m`` cap ``c / (1 - m)`` keeps every factor of
    the process positive and is applied by the caller.
    """
    t = np.arange(1, len(z) + 1)
    mu = (0.5 + np.cumsum(z)) / (t + 1)
    resid = np.cumsum((z - mu) ** 2)
    var_prev = np.concatenate([[0.25], (0.25 + resid[:-1]) / (t[:-1] + 1)])
    lam = np.sqrt(2 * np.log(1 / alpha) / (var_prev * t * np.log1p(t)))
    return np.minimum(lam, c)


def betting_log_capital(
    z: NDArray, m: NDArray | float, alpha: float, c: float = 0.75
) -> NDArray:
    """Log of the WSR capital ``prod_t 1 + lambda_t (m - z_t)`` for candidate
    means ``m``. ``z`` must lie in ``[0, 1]``. The capital is a nonnegative
    supermartingale for every ``m >= mean(z)``, so it exceeds ``1 / alpha``
    with probability at most ``alpha`` at any such ``m``.
    """
    z = np.asarray(z, dtype=np.float64)
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    lam = _betting_lambdas(z, alpha, c)
    lam_m = np.minimum(lam[:, None], c / np.maximum(1 - m[None, :], 1e-12))
    return np.log1p(lam_m * (m[None, :] - z[:, None])).sum(0)


def betting_upper_bound(
    z: ArrayLike, alpha: float, grid: int = 4096, c: float = 0.75
) -> float:
    """``1 - alpha`` upper confidence bound on the mean of i.i.d. ``z`` in
    ``[0, 1]`` from the betting confidence sequence, resolved on a uniform
    grid of candidate means.
    """
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return 1.0
    ms = np.linspace(0, 1, grid + 1)
    accept = betting_log_capital(z, ms, alpha, c) < np.log(1 / alpha)
    return float(ms[accept].max()) if accept.any() else 1.0


def clt_upper_bound(z: ArrayLike, alpha: float) -> float:
    """Asymptotic ``1 - alpha`` upper bound ``mean + z_{1-alpha} sd / sqrt(m)``."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return 1.0
    sd = z.std(ddof=1) if len(z) > 1 else 0.5
    return float(min(1.0, z.mean() + norm.ppf(1 - alpha) * sd / np.sqrt(len(z))))


def hoeffding_upper_bound(z: ArrayLike, alpha: float) -> float:
    """``1 - alpha`` Hoeffding upper bound for ``z`` in ``[0, 1]``."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return 1.0
    return float(min(1.0, z.mean() + np.sqrt(np.log(1 / alpha) / (2 * len(z)))))


UPPER_BOUNDS: dict[str, Callable[[NDArray, float], float]] = {
    "betting": betting_upper_bound,
    "clt": clt_upper_bound,
    "hoeffding": hoeffding_upper_bound,
}


def _certifies(z: NDArray, eps: float, alpha: float, method: BoundMethod) -> bool:
    """Whether the sample certifies ``mean(z) <= eps`` at level ``1 - alpha``."""
    if method == "betting":
        return bool(betting_log_capital(z, eps, alpha)[0] >= np.log(1 / alpha))
    return UPPER_BOUNDS[method](z, alpha) <= eps


@dataclass
class PacLabeling:
    """Output of :func:`pac_label`."""

    u_hat: float
    """Uncertainty threshold; items with ``U >= u_hat`` need expert labels."""

    expert: NDArray
    """Boolean mask over items that receive expert labels: the sampled items
    plus every item at or above ``u_hat``."""

    sample: NDArray
    """Indices of the uniform sample whose expert labels set the threshold."""

    eps: float
    alpha: float

    @property
    def budget_save(self) -> float:
        """Fraction of items that keep the cheap label."""
        return 1.0 - float(self.expert.mean())


def sample_indices(n: int, m: int, seed: int | np.random.Generator = 0) -> NDArray:
    """``m`` indices drawn uniformly with replacement from ``range(n)``, the
    i.i.d. sample the confidence bound needs."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=m)


def pac_threshold(
    sample_loss: ArrayLike,
    sample_u: ArrayLike,
    eps: float,
    alpha: float,
    method: BoundMethod = "betting",
) -> float:
    """Smallest uncertainty ``u`` among the sampled values at which the
    cumulative loss below ``u`` cannot be certified to be at most ``eps``.

    ``sample_loss[j]`` is the loss of the cheap label on sampled item ``j``
    (``[0, 1]``) and ``sample_u[j]`` its uncertainty. Returns ``inf`` when
    every threshold is certified, so the cheap labeler may label everything.
    Thresholds between consecutive sampled uncertainties share a bound, so
    testing the sampled values is exact.
    """
    loss = np.asarray(sample_loss, dtype=np.float64)
    u = np.asarray(sample_u, dtype=np.float64)
    if loss.min() < 0 or loss.max() > 1:
        raise ValueError("losses must lie in [0, 1]")
    if u.ndim != 1 or loss.shape != u.shape:
        raise ValueError("sample_loss and sample_u must be 1-D and equal length")

    for cand in np.unique(u):
        z = loss * (u <= cand)
        if not _certifies(z, eps, alpha, method):
            return float(cand)
    return float("inf")


def pac_label(
    u: ArrayLike,
    sample: ArrayLike,
    sample_loss: ArrayLike,
    eps: float,
    alpha: float,
    method: BoundMethod = "betting",
) -> PacLabeling:
    """Algorithm 1 of arXiv 2506.10908.

    ``u`` holds every item's uncertainty, ``sample`` the indices returned by
    :func:`sample_indices` and ``sample_loss`` the cheap label's loss on each
    sampled item against its expert label. The result marks which items need
    expert labels so the final labeling has average loss at most ``eps`` with
    probability at least ``1 - alpha``.
    """
    u = np.asarray(u, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.int64)
    u_hat = pac_threshold(sample_loss, u[sample], eps, alpha, method)
    expert = u >= u_hat
    expert[sample] = True
    return PacLabeling(u_hat=u_hat, expert=expert, sample=sample, eps=eps, alpha=alpha)


def loss_curve(
    sample_loss: ArrayLike,
    sample_u: ArrayLike,
    alpha: float,
    thresholds: ArrayLike,
    method: BoundMethod = "betting",
) -> NDArray:
    """Upper confidence bound on the cumulative loss below each threshold."""
    loss = np.asarray(sample_loss, dtype=np.float64)
    u = np.asarray(sample_u, dtype=np.float64)
    bound = UPPER_BOUNDS[method]
    return np.array([bound(loss * (u <= t), alpha) for t in np.asarray(thresholds)])
