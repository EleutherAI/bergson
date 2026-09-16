import numpy as np
import pytest

from bergson.pac import (
    betting_upper_bound,
    clt_upper_bound,
    hoeffding_upper_bound,
    loss_curve,
    pac_label,
    pac_threshold,
    sample_indices,
)


def synthetic(n: int, rng: np.random.Generator, informative: bool = True):
    """Items whose 0-1 loss probability rises with the uncertainty score."""
    u = rng.uniform(size=n)
    p = 0.4 * u**3 if informative else np.full(n, 0.05)
    loss = (rng.uniform(size=n) < p).astype(np.float64)
    return u, loss


def realized_error(loss: np.ndarray, expert: np.ndarray) -> float:
    return float((loss * ~expert).mean())


@pytest.mark.parametrize("method", ["betting", "hoeffding", "clt"])
def test_upper_bound_covers_mean(method):
    rng = np.random.default_rng(0)
    bound = {
        "betting": betting_upper_bound,
        "hoeffding": hoeffding_upper_bound,
        "clt": clt_upper_bound,
    }[method]
    p, misses = 0.1, 0
    for _ in range(200):
        z = (rng.uniform(size=300) < p).astype(float)
        misses += bound(z, 0.1) < p
    assert misses / 200 <= 0.16


def test_betting_bound_tightens_with_sample_size():
    rng = np.random.default_rng(1)
    z = (rng.uniform(size=4000) < 0.2).astype(float)
    assert betting_upper_bound(z[:100], 0.05) > betting_upper_bound(z, 0.05) > 0.2


def test_pac_labels_meet_error_target():
    rng = np.random.default_rng(2)
    n, eps, alpha = 5000, 0.02, 0.1
    failures = 0
    for _ in range(100):
        u, loss = synthetic(n, rng)
        sample = sample_indices(n, 400, rng)
        out = pac_label(u, sample, loss[sample], eps, alpha)
        failures += realized_error(loss, out.expert) > eps
    assert failures / 100 <= alpha + 0.05


def test_informative_uncertainty_saves_labels():
    rng = np.random.default_rng(3)
    n = 20000
    u, loss = synthetic(n, rng)
    sample = sample_indices(n, 1000, rng)
    out = pac_label(u, sample, loss[sample], eps=0.02, alpha=0.1)
    assert 0.3 < out.budget_save < 0.9
    assert realized_error(loss, out.expert) <= 0.02


def test_uninformative_uncertainty_keeps_guarantee():
    rng = np.random.default_rng(4)
    n = 20000
    u, loss = synthetic(n, rng, informative=False)
    sample = sample_indices(n, 1000, rng)
    out = pac_label(u, sample, loss[sample], eps=0.02, alpha=0.1)
    # Constant 5% loss: the oracle labels at most 40% cheaply at eps 2%.
    assert out.budget_save <= 0.4
    assert realized_error(loss, out.expert) <= 0.02


def test_threshold_monotone_in_eps():
    rng = np.random.default_rng(5)
    u, loss = synthetic(5000, rng)
    sample = sample_indices(5000, 500, rng)
    thresholds = [
        pac_threshold(loss[sample], u[sample], eps, 0.1) for eps in (0.005, 0.02, 0.05)
    ]
    assert thresholds == sorted(thresholds)


def test_all_cheap_when_loss_is_zero():
    u = np.linspace(0, 1, 100)
    sample = np.arange(100)
    out = pac_label(u, sample, np.zeros(100), eps=0.2, alpha=0.1)
    assert out.u_hat == float("inf")
    assert out.expert.all()  # the sample itself is expert labeled
    assert pac_threshold(np.zeros(30), u[:30], 0.2, 0.1) == float("inf")


def test_loss_curve_is_monotone():
    rng = np.random.default_rng(6)
    u, loss = synthetic(2000, rng)
    curve = loss_curve(loss, u, 0.1, np.linspace(0, 1, 11), method="hoeffding")
    assert np.all(np.diff(curve) >= 0)


def test_rejects_invalid_losses():
    with pytest.raises(ValueError):
        pac_threshold([0.5, 1.5], [0.1, 0.2], 0.1, 0.1)


def test_importance_weighted_sample_keeps_guarantee():
    rng = np.random.default_rng(7)
    n, eps, alpha, budget = 20000, 0.01, 0.1, 1000
    u, loss = synthetic(n, rng)
    pi = np.where(u > 0.8, 1.0, 0.2)
    m = int(budget / pi.mean())
    failures, saves = 0, []
    for _ in range(40):
        sample = sample_indices(n, m, rng)
        labeled = rng.uniform(size=m) < pi[sample]
        z = np.where(labeled, loss[sample] / pi[sample], 0.0)
        out = pac_label(u, sample, z, eps, alpha, z_max=1 / pi.min(), labeled=labeled)
        failures += realized_error(loss, out.expert) > eps
        saves.append(out.budget_save)
    uniform = pac_label(u, (s := sample_indices(n, budget, rng)), loss[s], eps, alpha)
    assert failures / 40 <= alpha + 0.05
    # Labels spent above the threshold buy nothing, so the weighted sample
    # only matches the uniform one at the same label budget.
    assert abs(np.mean(saves) - uniform.budget_save) < 0.1
