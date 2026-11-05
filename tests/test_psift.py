"""Preconditioning both sides with ``H^-1/2`` (P-SIFT) gives the influence
``g^T H^-1 q`` as an inner product."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from bergson.config import InversionConfig
from bergson.hessians.preconditioner import FactoredPreconditioner

MODULES = {"layers.0.mlp": (4, 6), "layers.1.mlp": (3, 5)}


def write_factors(path: Path) -> None:
    """Random orthonormal eigenbases and positive eigenvalues for every module."""
    torch.manual_seed(0)
    eigen_a = {
        n: torch.linalg.qr(torch.randn(i, i))[0].contiguous()
        for n, (_, i) in MODULES.items()
    }
    eigen_g = {
        n: torch.linalg.qr(torch.randn(o, o))[0].contiguous()
        for n, (o, _) in MODULES.items()
    }
    lambdas = {n: torch.rand(o, i) + 0.1 for n, (o, i) in MODULES.items()}
    for subdir, shard in (
        ("eigen_activation_sharded", eigen_a),
        ("eigen_gradient_sharded", eigen_g),
        ("eigenvalue_sharded", lambdas),
    ):
        (path / subdir).mkdir(parents=True, exist_ok=True)
        save_file(shard, str(path / subdir / "shard_0.safetensors"))


def test_half_powers_give_the_influence(tmp_path: Path):
    write_factors(tmp_path)

    def load(power: float) -> FactoredPreconditioner:
        return FactoredPreconditioner.from_path(
            tmp_path, inversion_cfg=InversionConfig(damping_factor=0.1), power=power
        )

    half, inverse = load(-0.5), load(-1.0)
    for name, (o, i) in MODULES.items():
        g, q = torch.randn(3, o * i), torch.randn(2, o * i)
        influence = g @ inverse.apply({name: q})[name].T
        for rotate_back in (True, False):
            g_half = half.apply({name: g}, rotate_back=rotate_back)[name]
            q_half = half.apply({name: q}, rotate_back=rotate_back)[name]
            torch.testing.assert_close(g_half @ q_half.T, influence)
