import torch

from bergson.utils.math import compute_lambda


def _make_eigen(eigvals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper: create (eigenvalues, eigenvectors=identity) for a diagonal PSD matrix."""
    d = eigvals.shape[0]
    return eigvals, torch.eye(d, dtype=eigvals.dtype)


def test_identical_preconditioners():
    """When R_eval == R_train, σ_eval[k] == σ_train[k] so λ = 0.5."""
    eigvals = torch.arange(100, 0, -1, dtype=torch.float64)
    eigen = {"mod": _make_eigen(eigvals)}
    lam = compute_lambda(query_eigen=eigen, index_eigen=eigen, target_components=50)
    assert abs(lam - 0.5) < 1e-6


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


def test_large_train_singular_values_gives_high_lambda():
    """When R_train has much larger singular values (e.g. C4 long sequences),
    λ should be close to 1.0, matching the paper's λ=0.99 for C4."""
    d = 1000
    q_eigvals = torch.ones(d, dtype=torch.float64)
    # Train singular values 100x larger (simulating longer sequences)
    i_eigvals = torch.ones(d, dtype=torch.float64) * 100.0

    q_eigen = {"mod": _make_eigen(q_eigvals)}
    i_eigen = {"mod": _make_eigen(i_eigvals)}

    # At any k: λ = 100 / (1 + 100) ≈ 0.99
    lam = compute_lambda(q_eigen, i_eigen, target_components=100)
    assert abs(lam - 100.0 / 101.0) < 1e-6


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


def test_result_in_unit_interval():
    """Lambda is always in [0, 1]."""
    torch.manual_seed(42)
    for _ in range(20):
        d = int(torch.randint(10, 200, (1,)).item())
        q_eigvals = torch.rand(d, dtype=torch.float64).clamp(min=1e-6)
        i_eigvals = torch.rand(d, dtype=torch.float64).clamp(min=1e-6)

        q_eigen = {"mod": _make_eigen(q_eigvals)}
        i_eigen = {"mod": _make_eigen(i_eigvals)}
        target = int(torch.randint(1, d, (1,)).item())
        lam = compute_lambda(q_eigen, i_eigen, target_components=target)
        assert 0.0 <= lam <= 1.0, f"λ={lam} out of range"


def test_eigenvectors_are_ignored():
    """Result depends only on eigenvalues, not eigenvectors."""
    eigvals_q = torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64)
    eigvals_i = torch.tensor([6.0, 3.0, 2.0], dtype=torch.float64)

    # Identity eigenvectors
    e1 = (eigvals_q, torch.eye(3, dtype=torch.float64))
    f1 = (eigvals_i, torch.eye(3, dtype=torch.float64))

    # Random orthogonal eigenvectors
    torch.manual_seed(123)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    e2 = (eigvals_q, Q)
    f2 = (eigvals_i, Q)

    lam1 = compute_lambda({"m": e1}, {"m": f1}, target_components=2)
    lam2 = compute_lambda({"m": e2}, {"m": f2}, target_components=2)
    assert abs(lam1 - lam2) < 1e-10
