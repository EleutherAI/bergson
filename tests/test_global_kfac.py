"""Global projection of a factored inverse-Hessian query.

``apply_hessian`` preconditions the query unprojected, then randomly
down-projects it to match the index it will be scored against. Under
``projection_target="global"`` that projection sums every module into one vector.

The factors are written by hand: identity eigenvectors and unit eigenvalues, so
the tests do not depend on what the inversion does to them. kfac, tkfac and
shampoo write these same artifacts and share the apply path.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bergson.collector.collector import HookCollectorBase, project_global
from bergson.config import InversionConfig
from bergson.data import create_index, load_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.utils.utils import get_device

MODULES = {"layers.0.mlp": (4, 6), "layers.1.mlp": (3, 5)}
# The projection matrices come from a device-seeded generator, so an expectation
# has to be drawn on the device the applicator drew its own on.
DEVICE = get_device(0)
NUM_QUERIES = 3
PROJ_DIM = 8


def write_factors(path: Path, modules: dict | None = None) -> None:
    """Identity eigenbases and unit eigenvalues for every module."""
    modules = MODULES if modules is None else modules
    eigen_a = {name: torch.eye(i) for name, (_, i) in modules.items()}
    eigen_g = {name: torch.eye(o) for name, (o, _) in modules.items()}
    lambdas = {name: torch.ones(o, i) for name, (o, i) in modules.items()}

    for subdir, shard in (
        ("eigen_activation_sharded", eigen_a),
        ("eigen_gradient_sharded", eigen_g),
        ("eigenvalue_sharded", lambdas),
    ):
        (path / subdir).mkdir(parents=True, exist_ok=True)
        save_file(shard, str(path / subdir / "shard_0.safetensors"))


def write_queries(path: Path) -> dict[str, torch.Tensor]:
    """An unprojected per-module query index, as pipeline step 1 builds it."""
    torch.manual_seed(0)
    grads = {name: torch.randn(NUM_QUERIES, o * i) for name, (o, i) in MODULES.items()}
    grad_sizes = {name: o * i for name, (o, i) in MODULES.items()}
    buffer = create_index(
        path, num_grads=NUM_QUERIES, grad_sizes=grad_sizes, dtype=np.float32
    )
    buffer[:] = torch.cat([grads[name] for name in grad_sizes], dim=1).numpy()
    buffer.flush()
    return grads


def apply(tmp_path: Path, name: str, **cfg_kwargs) -> Path:
    """Run the applicator over the fixture and return its output directory."""
    out = tmp_path / name
    applicator = EkfacApplicator(
        EkfacConfig(
            hessian_method_path=str(tmp_path / "factors"),
            gradient_path=str(tmp_path / "queries"),
            run_path=str(out),
            ev_correction=False,
            **cfg_kwargs,
        ),
        InversionConfig(),
    )
    applicator.compute_ivhp_sharded()
    return out


def load(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(load_gradients(path))).float()


def down_project(name: str, block: torch.Tensor, dim: int) -> torch.Tensor:
    """``name``'s global projection of ``block``, as the index draws it."""
    return project_global(
        HookCollectorBase.projection_identifier(name, "single", None),
        block.to(DEVICE),
        dim,
        "rademacher",
        "jl",
    ).cpu()


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    write_factors(tmp_path / "factors")
    write_queries(tmp_path / "queries")
    return tmp_path


def test_global_sum_matches_projecting_each_module(fixture: Path):
    """The one vector is the sum of each module's own global projection."""
    unprojected = load(apply(fixture, "plain", projection_dim=0))
    projected = load(
        apply(fixture, "global", projection_dim=PROJ_DIM, projection_target="global")
    )
    assert projected.shape == (NUM_QUERIES, PROJ_DIM)

    expected = torch.zeros(NUM_QUERIES, PROJ_DIM)
    offset = 0
    for name, (o, i) in MODULES.items():
        expected += down_project(
            name, unprojected[:, offset : offset + o * i], PROJ_DIM
        )
        offset += o * i
    torch.testing.assert_close(projected, expected)


def test_projection_follows_the_seed(fixture: Path):
    """A seeded index needs a query down-projected with the same seeded matrices."""
    common = dict(projection_dim=PROJ_DIM, projection_target="global")
    plain = np.asarray(load_gradients(apply(fixture, "plain", **common)))
    seeded = np.asarray(
        load_gradients(apply(fixture, "seeded", projection_seed=7, **common))
    )
    again = np.asarray(
        load_gradients(apply(fixture, "again", projection_seed=7, **common))
    )

    assert not np.allclose(plain, seeded)
    np.testing.assert_allclose(seeded, again)


def test_global_rejects_a_module_without_factors(tmp_path: Path):
    """A module the index projected but the Hessian omits would drop a term."""
    write_factors(tmp_path / "factors", dict(list(MODULES.items())[:1]))
    write_queries(tmp_path / "queries")

    with pytest.raises(ValueError, match="no Hessian factors"):
        apply(tmp_path, "global", projection_dim=PROJ_DIM, projection_target="global")


def test_global_without_a_projection_is_unprojected(fixture: Path):
    """No projection means the target is ignored."""
    plain = np.asarray(load_gradients(apply(fixture, "plain", projection_dim=0)))
    unprojected = np.asarray(
        load_gradients(
            apply(fixture, "global", projection_dim=0, projection_target="global")
        )
    )
    np.testing.assert_array_equal(unprojected, plain)
