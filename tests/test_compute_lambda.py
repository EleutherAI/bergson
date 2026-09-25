import torch

from bergson.utils.math import compute_lambda


def _make_eigen(eigvals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: create (eigenvalues, eigenvectors=identity) for a diagonal PSD matrix."""
    d = eigvals.shape[0]
    return eigvals, torch.eye(d, dtype=eigvals.dtype)


def test_target_zero_returns_one():
    """target_components=0 means no downweighting → λ = 1.0."""
    eigvals = torch.ones(100, dtype=torch.float64)
    eigen = {"mod": _make_eigen(eigvals)}
    assert compute_lambda(eigen, eigen, target_components=0) == 1.0


def test_target_exceeds_total_clamps():
    """target_components > total dims uses the last component."""
    q = torch.tensor([4.0, 2.0], dtype=torch.float64)
    i = torch.tensor([6.0, 3.0], dtype=torch.float64)
    q_eigen = {"mod": _make_eigen(q)}
    i_eigen = {"mod": _make_eigen(i)}

    # target=10 > total=2, should clamp to total and use last component
    # sorted_q = [4, 2], sorted_i = [6, 3] → at k=1: λ = 3/(2+3) = 0.6
    lam = compute_lambda(q_eigen, i_eigen, target_components=10)
    assert abs(lam - 0.6) < 1e-6


def test_formula_direct():
    """Verify λ = σ_train[k] / (σ_eval[k] + σ_train[k]) directly."""
    # Query eigenvalues (sorted desc): [10, 8, 5, 3, 1]
    q_eigvals = torch.tensor([10.0, 8.0, 5.0, 3.0, 1.0], dtype=torch.float64)
    # Index eigenvalues (sorted desc): [20, 6, 4, 2, 0.5]
    i_eigvals = torch.tensor([20.0, 6.0, 4.0, 2.0, 0.5], dtype=torch.float64)

    q_eigen = {"mod": _make_eigen(q_eigvals)}
    i_eigen = {"mod": _make_eigen(i_eigvals)}

    # target=1: k=0, σ_eval=10, σ_train=20, λ = 20/30 = 2/3
    lam = compute_lambda(q_eigen, i_eigen, target_components=1)
    assert abs(lam - 20.0 / 30.0) < 1e-6

    # target=3: k=2, σ_eval=5, σ_train=4, λ = 4/9
    lam = compute_lambda(q_eigen, i_eigen, target_components=3)
    assert abs(lam - 4.0 / 9.0) < 1e-6

    # target=5: k=4, σ_eval=1, σ_train=0.5, λ = 0.5/1.5 = 1/3
    lam = compute_lambda(q_eigen, i_eigen, target_components=5)
    assert abs(lam - 1.0 / 3.0) < 1e-6


def test_multiple_modules_pools_globally():
    """Eigenvalues are pooled across modules before sorting."""
    # Module A: eigenvalues [100, 10]
    qa = torch.tensor([100.0, 10.0], dtype=torch.float64)
    ia = torch.tensor([50.0, 5.0], dtype=torch.float64)

    # Module B: eigenvalues [80, 1]
    qb = torch.tensor([80.0, 1.0], dtype=torch.float64)
    ib = torch.tensor([40.0, 2.0], dtype=torch.float64)

    q_eigen = {"a": _make_eigen(qa), "b": _make_eigen(qb)}
    i_eigen = {"a": _make_eigen(ia), "b": _make_eigen(ib)}

    # Global sorted query:  [100, 80, 10, 1]
    # Global sorted index: [50, 40, 5, 2]

    # target=1: k=0, λ = 50 / (100 + 50) = 1/3
    lam = compute_lambda(q_eigen, i_eigen, target_components=1)
    assert abs(lam - 50.0 / 150.0) < 1e-6

    # target=2: k=1, λ = 40 / (80 + 40) = 1/3
    lam = compute_lambda(q_eigen, i_eigen, target_components=2)
    assert abs(lam - 40.0 / 120.0) < 1e-6

    # target=3: k=2, λ = 5 / (10 + 5) = 1/3
    lam = compute_lambda(q_eigen, i_eigen, target_components=3)
    assert abs(lam - 5.0 / 15.0) < 1e-6


def test_no_common_modules_returns_default():
    """When there are no overlapping module names, return the default 0.99."""
    q_eigen = {"mod_a": _make_eigen(torch.ones(10))}
    i_eigen = {"mod_b": _make_eigen(torch.ones(10))}
    assert compute_lambda(q_eigen, i_eigen) == 0.99
