"""Subsampled randomized Hadamard sketches of module gradients."""

import pytest
import torch

from bergson.collector.collector import project_global
from bergson.collector.hadamard import _sketch, fwht, hadamard_project
from bergson.gradients import GradientProcessor, OuterProductGradients


def hadamard_matrix(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while len(H) < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)])
    return H / len(H) ** 0.5


def test_fwht_matches_the_hadamard_matrix():
    x = torch.randn(3, 8)
    torch.testing.assert_close(fwht(x), x @ hadamard_matrix(8).T)
    torch.testing.assert_close(fwht(fwht(x)), x)
    torch.testing.assert_close(fwht(x.T, 0), fwht(x).T)


def test_matches_the_dense_transform():
    """The sketch is sqrt(N / m) S H D x on the zero-padded flattened gradient."""
    x = torch.randn(3, 5, 6)
    out = hadamard_project("m", x, 12)
    signs, entries, scale = _sketch("m", 12, 30, x.device)
    padded = torch.nn.functional.pad(x.flatten(1), (0, 2))
    dense = (padded * signs) @ hadamard_matrix(32).T
    torch.testing.assert_close(out, dense[:, entries] * scale)
    assert scale == (32 / 12) ** 0.5


def test_inner_products_are_unbiased():
    """Averaged over sketches, inner products match, with a spread close to a
    dense Rademacher projection's."""
    torch.manual_seed(0)
    x, y = torch.randn(2, 1, 5, 6)
    exact = (x * y).sum().item()

    def estimates(projection_type: str) -> torch.Tensor:
        out = []
        for seed in range(2000):
            sx, sy = (
                project_global(f"m/{seed}", v, 16, projection_type)  # type: ignore
                for v in (x, y)
            )
            out.append((sx * sy).sum())
        return torch.stack(out)

    hadamard, rademacher = estimates("hadamard"), estimates("rademacher")
    assert abs(hadamard.mean().item() - exact) < 4 * hadamard.std().item() / 2000**0.5
    assert hadamard.std() < 1.5 * rademacher.std()


def test_vectors_match_formed_gradients():
    """Per-token and per-example gradients sketch as their formed [O, W] matrices."""
    torch.manual_seed(0)
    g, a, bias = torch.randn(4, 5), torch.randn(4, 7), torch.randn(4, 5)
    bias_col = torch.zeros(7)
    bias_col[6] = 1
    per_token = OuterProductGradients(g, a, bias, bias_col)
    torch.testing.assert_close(
        hadamard_project("m", per_token, 32),
        hadamard_project("m", per_token.materialize(), 32),
    )

    per_example = OuterProductGradients(g.view(2, 2, 5), a.view(2, 2, 7))
    torch.testing.assert_close(
        hadamard_project("m", per_example, 32),
        hadamard_project("m", per_example.materialize(), 32),
    )


def test_sketch_follows_the_identifier():
    x = torch.randn(3, 5, 6)
    first, again = hadamard_project("a", x, 16), hadamard_project("a", x, 16)
    torch.testing.assert_close(first, again)
    assert not torch.allclose(first, hadamard_project("b", x, 16))


def test_more_entries_than_the_transform_has():
    """Inner products stay unbiased once ``m`` exceeds the padded size."""
    x, y = torch.randn(2, 1, 3, 2)
    estimates = []
    for seed in range(2000):
        sx, sy = (hadamard_project(f"m/{seed}", v, 40) for v in (x, y))
        estimates.append((sx * sy).sum())
    estimates = torch.stack(estimates)
    exact = (x * y).sum()
    assert abs(estimates.mean() - exact) < 4 * estimates.std() / 2000**0.5


def test_hadamard_needs_a_global_target():
    with pytest.raises(ValueError, match="global"):
        GradientProcessor(projection_dim=16, projection_type="hadamard")
    with pytest.raises(ValueError, match="jl"):
        project_global("m", torch.randn(2, 3, 4), 8, "hadamard", "row_norm")


def test_smaller_sketches_are_prefixes():
    x = torch.randn(3, 5, 6)
    full = hadamard_project("m", x, 64)
    torch.testing.assert_close(
        hadamard_project("m", x, 24), full[:, :24] * (64 / 24) ** 0.5
    )
