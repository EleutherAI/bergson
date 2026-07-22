"""Random projections must preserve inner products, for every module shape.

Projecting two gradients and taking their inner product should, in expectation,
return the unprojected inner product, which requires entries with variance
``1/m`` so that ``E[AᵀA] = I``.

A shape-dependent factor matters because a score sums ``⟨g_m, q_m⟩`` over
modules, so it reweights modules against each other. Attention (``d×d``) and MLP
(``d×4d``) blocks differ by 4×.
"""

import math

import pytest
import torch
import yaml

from bergson.collector.collector import create_projection_matrix

TRIALS = 2000
# Tolerance for a TRIALS-sample Monte Carlo mean.
RTOL = 0.05


def _mean_projected_inner_product(o, i, p, projection_type, trials=TRIALS):
    """E[⟨L G1 Rᵀ, L G2 Rᵀ⟩] / ⟨G1, G2⟩ over independent projection draws."""
    g = torch.Generator().manual_seed(3)
    G1 = torch.randn(o, i, generator=g)
    G2 = G1 + 0.3 * torch.randn(o, i, generator=g)
    true = (G1 * G2).sum().item()

    device, dtype = torch.device("cpu"), torch.float32
    acc = 0.0
    for t in range(trials):
        L = create_projection_matrix(
            f"L{o}_{i}_{p}_{t}", p, o, dtype, device, projection_type
        )
        R = create_projection_matrix(
            f"R{o}_{i}_{p}_{t}", p, i, dtype, device, projection_type
        )
        acc += ((L @ G1 @ R.T) * (L @ G2 @ R.T)).sum().item()
    return acc / trials / true


@pytest.mark.parametrize("projection_type", ["rademacher", "normal"])
@pytest.mark.parametrize("o,i,p", [(32, 32, 16), (64, 64, 32), (128, 128, 32)])
def test_projection_preserves_inner_products(o, i, p, projection_type):
    """Two-sided projection is unbiased for the unprojected inner product."""
    got = _mean_projected_inner_product(o, i, p, projection_type)
    assert got == pytest.approx(1.0, rel=RTOL), (
        f"{o}x{i} p={p} {projection_type}: projected inner product is "
        f"{got:.4f}x the true one; expected ~1.0. "
        f"A factor of {p * p / (o * i):.4f} indicates 1/sqrt(n) row normalization."
    )


@pytest.mark.parametrize("projection_type", ["rademacher", "normal"])
def test_projection_scale_does_not_depend_on_module_shape(projection_type):
    """Differently-shaped modules keep their relative weight.

    Modules are summed into a single score, so a shape-dependent scale
    reweights attention against MLP.
    """
    square = _mean_projected_inner_product(64, 64, 32, projection_type)
    wide = _mean_projected_inner_product(64, 256, 32, projection_type)

    assert square == pytest.approx(wide, rel=2 * RTOL), (
        f"scale depends on module shape: 64x64 -> {square:.4f}, "
        f"64x256 -> {wide:.4f}"
    )


@pytest.mark.parametrize("projection_type", ["rademacher", "normal"])
def test_projection_entry_variance_is_one_over_output_dim(projection_type):
    """Directly pin the JL scaling that makes E[AᵀA] = I."""
    p, n = 64, 256
    A = create_projection_matrix(
        "variance-probe", p, n, torch.float32, torch.device("cpu"), projection_type
    )
    assert A.shape == (p, n)

    var = A.pow(2).mean().item()
    assert var == pytest.approx(1.0 / p, rel=0.1), (
        f"entry variance {var:.3e}, expected {1.0 / p:.3e} (=1/p); "
        f"{1.0 / n:.3e} indicates 1/n row normalization"
    )

    # E[AᵀA] = I: the diagonal is what carries the inner-product scale.
    gram_diag = (A.T @ A).diagonal().mean().item()
    assert gram_diag == pytest.approx(
        1.0, rel=0.1
    ), f"mean diag(AᵀA) = {gram_diag:.4f}, expected 1.0"


def test_single_sided_projection_is_unbiased():
    """The global-projection path applies one matrix per module and sums.

    Each module draws an independent matrix, so cross terms vanish in
    expectation and the sum reproduces the full inner product.
    """
    dims = [64, 256]  # deliberately different, to catch a per-module factor
    p = 32
    g = torch.Generator().manual_seed(11)
    xs = [torch.randn(d, generator=g) for d in dims]
    ys = [x + 0.3 * torch.randn(d, generator=g) for x, d in zip(xs, dims)]
    true = sum(float((x * y).sum()) for x, y in zip(xs, ys))

    device, dtype = torch.device("cpu"), torch.float32
    acc = 0.0
    for t in range(TRIALS):
        total = 0.0
        for k, d in enumerate(dims):
            R = create_projection_matrix(f"g{k}_{t}", p, d, dtype, device, "rademacher")
            total += float(((xs[k] @ R.T) * (ys[k] @ R.T)).sum())
        acc += total

    assert acc / TRIALS / true == pytest.approx(1.0, rel=RTOL)


def test_projection_matrix_is_deterministic_in_identifier():
    """Index and query sides must agree; the seed is the only thing tying them."""
    args = (16, 64, torch.float32, torch.device("cpu"))
    a = create_projection_matrix("same/left", *args, "rademacher")
    b = create_projection_matrix("same/left", *args, "rademacher")
    c = create_projection_matrix("other/left", *args, "rademacher")

    assert torch.equal(a, b), "same identifier must give the same matrix"
    assert not torch.equal(a, c), "different identifiers must differ"
    assert math.isclose(a.pow(2).mean().item(), 1.0 / 16, rel_tol=0.1)


def test_row_norm_scale_reproduces_legacy_matrices():
    """``row_norm`` gives row-normalized matrices with entry variance 1/n."""
    args = (32, 64, torch.float32, torch.device("cpu"))
    legacy = create_projection_matrix("m", *args, "rademacher", "row_norm")
    jl = create_projection_matrix("m", *args, "rademacher", "jl")

    torch.testing.assert_close(
        legacy, legacy / legacy.norm(dim=1, keepdim=True), rtol=1e-6, atol=1e-6
    )
    assert legacy.pow(2).mean().item() == pytest.approx(1.0 / 64, rel=0.1)
    assert jl.pow(2).mean().item() == pytest.approx(1.0 / 32, rel=0.1)


def test_processor_config_without_projection_scale_loads_as_row_norm(tmp_path):
    """A config with no ``projection_scale`` key loads as ``row_norm``."""
    from bergson.gradients import GradientProcessor

    GradientProcessor(projection_dim=8).save(tmp_path)
    cfg_path = tmp_path / "processor_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["projection_scale"] == "jl"

    del cfg["projection_scale"]
    cfg_path.write_text(yaml.safe_dump(cfg))

    assert GradientProcessor.load(tmp_path).projection_scale == "row_norm"
