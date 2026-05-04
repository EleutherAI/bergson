import gc
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from tqdm import tqdm

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger
from bergson.utils.utils import (
    assert_type,
    get_device,
)


def fair_distribute_by_cost(
    key_dimensions: dict[str, int],
    world_size: int,
) -> list[list[str]]:
    """
    Distribute keys fairly across ranks based on eigendecomposition cost O(d³).

    Uses snake ordering to balance computational load while ensuring each rank
    gets the same number of keys.

    Args:
        key_dimensions: Map from key name to matrix dimension d (for [d, d] matrices)
        world_size: Number of ranks

    Returns:
        List of lists, where result[rank] contains the keys assigned to that rank.

    Raises:
        ValueError: If number of keys is not divisible by world_size.
    """

    # Sort by cost (d³) in descending order for greedy assignment
    sorted_keys = sorted(
        key_dimensions.keys(),
        key=lambda k: key_dimensions[k] ** 3,
        reverse=True,
    )

    # Snake ordering: distribute to balance costs
    # Round 0: assign to ranks 0, 1, 2, ..., w-1
    # Round 1: assign to ranks w-1, w-2, ..., 0
    # This mimics tournament seeding and tends to balance total costs
    result: list[list[str]] = [[] for _ in range(world_size)]

    for i, key in enumerate(sorted_keys):
        round_num = i // world_size
        pos_in_round = i % world_size

        if round_num % 2 == 0:
            # Forward direction
            rank = pos_in_round
        else:
            # Reverse direction
            rank = world_size - 1 - pos_in_round

        result[rank].append(key)

    return result


@dataclass(kw_only=True)
class LambdaCollector(HookCollectorBase):
    """
    Computes eigenvalue corrections for EKFAC (Eq. 20 from paper).

    Transforms activations and gradients using precomputed eigenvectors,
    then computes outer products for diagonal correction terms.
    """

    path: str
    eigen_path: str | None = None
    """Override read location for eigvecs; defaults to ``self.path``."""

    output_subdir: str = "eigenvalue_correction_sharded"
    """Subdir under ``self.path`` to write lambda shards into."""

    def setup(self) -> None:
        """Load eigenvectors and initialize storage."""
        self.shard_computer = ShardedMul()
        self.device = get_device(self.rank)

        eigen_src = self.eigen_path or self.path

        # Load precomputed eigenvectors
        self.eigen_a = load_file(
            os.path.join(
                eigen_src, f"eigen_activation_sharded/shard_{self.rank}.safetensors"
            ),
            device=self.device,
        )
        self.eigen_g = load_file(
            os.path.join(
                eigen_src, f"eigen_gradient_sharded/shard_{self.rank}.safetensors"
            ),
            device=self.device,
        )

        # Initialize accumulators
        self.eigenvalue_corrections = {}
        self.transformed_a_cache = {}

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Transform activations using eigenvectors and cache."""
        name = assert_type(str, module._name)
        # a shape: [N, S, I]

        # Transform: a @ eigen_a
        transformed = self.shard_computer._matmul(
            vector_nsa=a, matrix_cb=self.eigen_a[name]
        )  # shape [N, S, I]

        # Cache for use in backward pass
        self.transformed_a_cache[name] = transformed

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Transform gradients and compute eigenvalue corrections."""
        name = assert_type(str, module._name)
        # g shape: [N, S, O]

        # Transform: g @ eigen_g
        transformed_g = self.shard_computer._matmul(
            vector_nsa=g, matrix_cb=self.eigen_g[name]
        )  # shape [N, S, O]

        # Compute outer product: sum_n (transformed_a_n^T @ transformed_g_n)
        # Einstein notation: [N, S, I] x [N, S, O] -> [N, O, I]
        transformed_grad_shard = torch.einsum(
            "N S I, N S O -> N O I", self.transformed_a_cache[name], transformed_g
        )

        # Square and sum over batch
        transformed_grad_shard = (transformed_grad_shard**2).sum(dim=0).contiguous()

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(transformed_grad_shard, op=dist.ReduceOp.SUM)

        # Extract our shard
        shard_size = transformed_grad_shard.shape[0] // self.world_size
        start_row = self.rank * shard_size
        end_row = (self.rank + 1) * shard_size

        # Accumulate (with CPU offloading for memory efficiency)
        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = transformed_grad_shard[
                start_row:end_row, :
            ].contiguous()
        else:
            self.eigenvalue_corrections[name] = self.eigenvalue_corrections[name].to(
                device=self.device
            )
            self.eigenvalue_corrections[name].add_(
                transformed_grad_shard[start_row:end_row, :].contiguous()
            )
            self.eigenvalue_corrections[name] = self.eigenvalue_corrections[name].to(
                device="cpu", non_blocking=False
            )

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for lambda collection."""
        pass

    def teardown(self) -> None:
        """Save eigenvalue corrections to disk."""
        output_path = os.path.join(self.path, self.output_subdir)
        os.makedirs(output_path, exist_ok=True)

        save_file(
            self.eigenvalue_corrections,
            os.path.join(output_path, f"shard_{self.rank}.safetensors"),
        )


def _compute_full_matrix(
    name: str,
    shard_path: str | os.PathLike,
    rank: int,
    world_size: int,
) -> Tensor:
    """
    Load a full matrix from sharded covariance files.
    Needed to compute eigendecomposition.
    """
    files = os.listdir(shard_path)
    assert (
        len(files) == world_size
    ), f"Expected {world_size} shards, found {len(files)} in {shard_path}"

    device = get_device(rank)
    full_matrix = None

    if world_size == 1:
        full_path_rank = os.path.join(shard_path, "shard_0.safetensors")
        with safe_open(full_path_rank, framework="pt", device=device) as f:
            full_matrix = f.get_tensor(name)
    else:
        full_matrix_list = []
        for shard_id in range(world_size):
            shard_path_rank = os.path.join(shard_path, f"shard_{shard_id}.safetensors")
            with safe_open(shard_path_rank, framework="pt", device=device) as f:
                local_matrix = f.get_tensor(name)

            full_matrix_list.append(local_matrix)

        # Concatenate all shards to form the full matrix
        full_matrix = torch.cat(full_matrix_list, dim=0)

    return full_matrix


def compute_eigendecomposition(
    covariance_path: str,
    total_processed: int | Tensor,
) -> dict[str, Tensor]:
    """
    Compute eigendecomposition from covariance matrices (Eq. 18 from paper).

    The function discovers keys from shard metadata (fast, no tensor loading).
    Keys are distributed across workers, and full matrices are reconstructed
    via _compute_full_matrix(). Output sharding is inferred from the
    eigenvector shapes.

    Args:
        covariance_path: Full path to the covariance sharded directory.
        total_processed: Number of samples used to compute covariance.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = get_device(rank)

    # Handle total_processed as tensor if needed
    if isinstance(total_processed, int):
        total_processed = torch.tensor(total_processed, device=device)
    else:
        total_processed = total_processed.to(device)

    # Discover keys and dimensions from shard metadata
    first_shard_path = os.path.join(covariance_path, "shard_0.safetensors")
    with safe_open(first_shard_path, framework="pt") as f:
        all_keys = list(f.keys())
        original_dtype = f.get_tensor(all_keys[0]).dtype
        # Get dimensions for fair distribution (columns not sharded, shape[-1]=d)
        key_dimensions = {key: f.get_tensor(key).shape[-1] for key in all_keys}

    # Distribute keys fairly based on O(d³) eigendecomposition cost
    all_assignments = fair_distribute_by_cost(key_dimensions, world_size)
    keys_for_this_rank = all_assignments[rank]

    covariance_eigenvectors = {}
    covariance_eigenvalues: dict[str, Tensor] = {}

    for key in tqdm(
        keys_for_this_rank,
        disable=False,
        desc=f"Rank {rank}: Computing eigenvectors",
        position=rank,
        leave=False,
    ):
        matrix = _compute_full_matrix(
            name=key,
            shard_path=covariance_path,
            rank=rank,
            world_size=world_size,
        )

        # original_dtype = matrix.dtype
        matrix_normalized = matrix.to(torch.float64) / total_processed
        matrix_normalized = (matrix_normalized + matrix_normalized.T).div(2)

        if not torch.isfinite(matrix_normalized).all():
            raise ValueError(
                f"Covariance matrix for {key} contains NaNs or Infs. "
                "Consider using fp32."
            )

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)
        except Exception as e:
            raise RuntimeError(f"Eigendecomposition failed for {key}") from e

        # TODO: Maybe possible to avoid CPU transfer here?
        eigenvectors = eigenvectors.to(original_dtype).to(device="cpu").contiguous()
        covariance_eigenvectors[key] = eigenvectors
        covariance_eigenvalues[key] = (
            eigenvalues.to(original_dtype).to(device="cpu").contiguous()
        )

    covariance_eigenvectors = _gather_and_shard_along_dim0(
        input_dict=covariance_eigenvectors,
        full_shape_per_key={k: (m, m) for k, m in key_dimensions.items()},
        dtype=original_dtype,  # type: ignore
        rank=rank,
        world_size=world_size,
        device=device,
    )
    covariance_eigenvalues = _gather_and_shard_along_dim0(
        input_dict=covariance_eigenvalues,
        full_shape_per_key={k: (m,) for k, m in key_dimensions.items()},
        dtype=original_dtype,  # type: ignore
        rank=rank,
        world_size=world_size,
        device=device,
    )

    # Generic output path by adding eigen_prefix to the path
    dirname = os.path.dirname(covariance_path)
    basename = os.path.basename(covariance_path)
    output_path = os.path.join(dirname, "eigen_" + basename)

    os.makedirs(output_path, exist_ok=True)
    save_file(
        covariance_eigenvectors,
        os.path.join(output_path, f"shard_{rank}.safetensors"),
    )

    gc.collect()

    get_logger().info(f"Saved eigenvectors to {output_path}")

    return covariance_eigenvalues


def save_uncorrected_eigenvalues(
    partial_run_path: str | os.PathLike,
    eva_a_local: dict[str, Tensor],
    eva_g_local: dict[str, Tensor],
    total_processed: int | Tensor,
    rank: int,
    world_size: int,
) -> None:
    """Write the outer product of the activation eigenvalues and the local
    gradient eigenvalues into `eigenvalue_sharded`.

    Both eigenvalue dicts are sharded along the eigenvector-index dim by
    `_gather_and_shard_along_dim0`.

    The output is scaled by `total_processed` to match the effective scaling
    of `LambdaCollector`.
    """
    out_dir = os.path.join(str(partial_run_path), "eigenvalue_sharded")
    os.makedirs(out_dir, exist_ok=True)

    device = get_device(rank)
    # Mirror compute_eigendecomposition: keep total_processed in its native
    # dtype on the right device. PyTorch type promotion handles float * int
    # cleanly so the outer product stays in the eigenvalues' dtype rather
    # than getting upcast to float32 / float64 by an explicit .float() cast.
    if isinstance(total_processed, int):
        total_processed = torch.tensor(total_processed, device=device)
    else:
        total_processed = total_processed.to(device)

    payload: dict[str, Tensor] = {}
    for key, eva_g_shard in eva_g_local.items():
        eva_g_shard = eva_g_shard.to(device)
        eva_a_shard = eva_a_local[key].to(device)

        if world_size > 1:
            eva_a_full = torch.empty(
                eva_a_shard.shape[0] * world_size,
                device=device,
                dtype=eva_a_shard.dtype,
            )
            dist.all_gather_into_tensor(eva_a_full, eva_a_shard.contiguous())
        else:
            eva_a_full = eva_a_shard

        outer = torch.outer(eva_g_shard, eva_a_full) * total_processed
        payload[key] = outer.to(device="cpu").contiguous()

    save_file(
        payload,
        os.path.join(out_dir, f"shard_{rank}.safetensors"),
    )

    get_logger().info(f"Saved uncorrected eigenvalues to {out_dir}")


def save_identity_eigen(
    partial_run_path: str | os.PathLike,
    dim_per_key: dict[str, int],
    sub_dir: str,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Write per-rank shards of identity Q-side matrices to `sub_dir`.

    `dim_per_key` maps each module name to the size `d` of its
    `[d, d]` identity Q.
    """
    payload: dict[str, Tensor] = {}
    for name, d in dim_per_key.items():
        if d % world_size != 0:
            raise ValueError(
                f"dim={d} for {name} is not divisible by world_size={world_size}."
            )
        shard_size = d // world_size
        shard = torch.zeros(shard_size, d, dtype=dtype)
        shard.diagonal(offset=rank * shard_size).fill_(1.0)
        payload[name] = shard

    out_dir = Path(str(partial_run_path)) / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(payload, out_dir / f"shard_{rank}.safetensors")


def save_identity_factors(
    partial_run_path: str | os.PathLike,
    layer_dims: dict[str, tuple[int, int]],
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Synthesise Q_A=I, Q_G=I and λ=1 directly to disk in the same on-disk
    layout as a real KFAC run, so apply_hessian works unchanged.

    `layer_dims` maps each target module name to its weight shape `(O, I)`.
    """
    partial_run_path = Path(str(partial_run_path))
    save_identity_eigen(
        partial_run_path,
        {n: i for n, (_, i) in layer_dims.items()},
        "eigen_activation_sharded",
        rank,
        world_size,
        dtype,
    )
    save_identity_eigen(
        partial_run_path,
        {n: o for n, (o, _) in layer_dims.items()},
        "eigen_gradient_sharded",
        rank,
        world_size,
        dtype,
    )

    lambda_dir = partial_run_path / "eigenvalue_kfac_sharded"
    lambda_dir.mkdir(parents=True, exist_ok=True)
    lambda_payload = {
        n: torch.ones(o // world_size, i, dtype=dtype)
        for n, (o, i) in layer_dims.items()
    }
    save_file(lambda_payload, lambda_dir / f"shard_{rank}.safetensors")

    if rank == 0:
        torch.save(torch.tensor(1.0), partial_run_path / "total_processed.pt")

    get_logger().info(
        f"Saved identity factors ({len(layer_dims)} modules) to {partial_run_path}"
    )


def _gather_and_shard_along_dim0(
    input_dict: dict[str, Tensor],
    full_shape_per_key: dict[str, tuple[int, ...]],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: str,
) -> dict[str, Tensor]:
    """
    Gather per-key tensors across ranks via all-reduce, then re-shard along
    dim 0. Used for both the eigenvector matrix `[m, m]` and the 1D
    eigenvalue array `[m]`.
    """
    if world_size == 1:
        return input_dict

    result_dict: dict[str, Tensor] = {}
    for key, full_shape in full_shape_per_key.items():
        if key in input_dict:
            tensor = input_dict[key].to(device=device, dtype=dtype)
        else:
            tensor = torch.zeros(full_shape, device=device, dtype=dtype)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        m = full_shape[0]
        shard_size = m // world_size
        shard = tensor[rank * shard_size : (rank + 1) * shard_size].contiguous()
        result_dict[key] = shard.to(device="cpu")

        del tensor
        gc.collect()

    return result_dict
