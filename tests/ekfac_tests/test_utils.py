"""Common utilities for EKFAC tests."""

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor


def add_tensor_dicts(a: dict[str, Tensor], b: dict[str, Tensor]) -> dict[str, Tensor]:
    """Add two dictionaries of tensors element-wise."""
    assert set(a.keys()) == set(b.keys()), "Keys must match"
    return {k: a[k] + b[k] for k in a}


def tensor_dict_to_device(
    d: dict[str, Tensor], device: str | torch.device
) -> dict[str, Tensor]:
    """Move all tensors in a dictionary to the specified device."""
    return {k: v.to(device) for k, v in d.items()}


def load_sharded_covariances(sharded_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load and concatenate sharded covariance files.

    Args:
        sharded_dir: Directory containing shard_0.safetensors, shard_1.safetensors, etc.

    Returns:
        Dictionary mapping layer names to concatenated covariance tensors.
    """
    sharded_dir = Path(sharded_dir)
    shard_files = sorted(sharded_dir.glob("shard_*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {sharded_dir}")

    shards = [load_file(str(f)) for f in shard_files]

    # Concatenate shards along first dimension
    result = {}
    for key in shards[0]:
        result[key] = torch.cat([shard[key] for shard in shards], dim=0)

    return result


def format_per_layer_errors(
    gt: dict[str, torch.Tensor],
    run: dict[str, torch.Tensor],
) -> str:
    """Format per-layer error details for debugging."""
    lines = []
    for k in sorted(gt.keys()):
        rel_err = (gt[k] - run[k]).norm() / gt[k].norm()
        lines.append(f"  {k}: rel_error={rel_err:.2e}")
    return "\n".join(lines)


def load_covariances(
    run_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Load activation and gradient covariances from an EKFAC run.

    Args:
        run_path: Path to the run directory containing influence_results/.

    Returns:
        Tuple of (activation_covariances, gradient_covariances, total_processed).
    """
    run_path = Path(run_path)
    results_path = run_path / "influence_results"

    A_cov = load_sharded_covariances(results_path / "activation_sharded")
    G_cov = load_sharded_covariances(results_path / "gradient_sharded")
    total_processed = torch.load(results_path / "total_processed.pt").item()

    return A_cov, G_cov, total_processed


# Widths of a small network whose layers split unevenly across ranks.
WIDTHS = (5, 7, 4, 3)


def small_network() -> nn.Sequential:
    torch.manual_seed(0)
    layers = []
    for in_dim, out_dim in zip(WIDTHS, WIDTHS[1:]):
        layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
    return nn.Sequential(*layers[:-1])


def rank_batches(rank: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(100 + rank)
    batches = []
    for b in range(3):
        n, s = 1 + (rank + b) % 3, 2 + (2 * rank + b) % 4
        x = torch.randn(n, s, WIDTHS[0], generator=generator)
        batches.append((x, torch.rand(n, s, generator=generator) > 0.3))
    return batches


def run_batches(collector, batches: list[tuple[Tensor, Tensor]]) -> None:
    """Run ``batches`` through ``collector.model`` with its hooks, then tear down."""
    network = collector.model
    for x, mask in batches:
        # with_batch returns the collector, whose context registers the hooks.
        with collector.with_batch(mask):
            (network(x) ** 2 * mask[..., None]).sum().backward()
        network.zero_grad()
    collector.teardown()
