import gc
import os
from dataclasses import dataclass

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

    def setup(self) -> None:
        """Load eigenvectors and initialize storage."""
        self.shard_computer = ShardedMul()
        self.device = get_device(self.rank)

        # Load precomputed eigenvectors
        self.eigen_a = load_file(
            os.path.join(
                self.path, f"eigen_activation_sharded/shard_{self.rank}.safetensors"
            ),
            device=self.device,
        )
        self.eigen_g = load_file(
            os.path.join(
                self.path, f"eigen_gradient_sharded/shard_{self.rank}.safetensors"
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
        output_path = os.path.join(self.path, "eigenvalue_correction_sharded")
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
) -> None:
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
    covariance_eigenvalues = {}

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
        # Keep eigenvalues in float64 on CPU — 1D, cheap, and KfacPreconditioner
        # wants precision for the (Λ_G ⊗ Λ_A) outer product.
        covariance_eigenvalues[key] = (
            eigenvalues.to(device="cpu").contiguous()
        )

    # Merge eigenvectors across ranks and re-shard for output
    covariance_eigenvectors = _merge_and_shard_eigenvectors(
        input_dict=covariance_eigenvectors,
        all_keys=all_keys,
        key_dimensions=key_dimensions,
        dtype=original_dtype,  # type: ignore
        rank=rank,
        world_size=world_size,
        device=device,
    )
    # Generic output path by adding eigen_prefix to the path
    dirname = os.path.dirname(covariance_path)
    basename = os.path.basename(covariance_path)
    output_path = os.path.join(dirname, "eigen_" + basename)
    eigenvalues_output_path = os.path.join(dirname, "eigenvalues_" + basename)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(eigenvalues_output_path, exist_ok=True)

    save_file(
        covariance_eigenvectors,
        os.path.join(output_path, f"shard_{rank}.safetensors"),
    )
    # Each rank writes only its own keys' eigenvalues; the loader side
    # concatenates across shards by key, which is a no-op for 1-shard keys.
    save_file(
        covariance_eigenvalues,
        os.path.join(eigenvalues_output_path, f"shard_{rank}.safetensors"),
    )

    gc.collect()

    get_logger().info(f"Saved eigenvectors to {output_path}")


def _merge_and_shard_eigenvectors(
    input_dict: dict[str, Tensor],
    all_keys: list[str],
    key_dimensions: dict[str, int],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: str,
) -> dict[str, Tensor]:
    """
    Redistribute eigenvectors across ranks.

    Each rank currently has full eigenvectors [m, m] for some keys.
    After this function, each rank has shards [m/world_size, m] for all keys.

    Matrix size m is inferred from the tensor shapes (no hardcoding needed).
    """
    if world_size == 1:
        return input_dict

    # Build reverse lookup: key -> owner rank
    all_assignments = fair_distribute_by_cost(key_dimensions, world_size)
    key_to_owner = {key: r for r, keys in enumerate(all_assignments) for key in keys}

    result_dict = {}

    for key in all_keys:
        owner_rank = key_to_owner[key]

        # Get or broadcast matrix size
        if key in input_dict:
            tensor = input_dict[key].to(device=device, dtype=dtype)
            m = tensor.shape[0]
            size_tensor = torch.tensor([m], dtype=torch.long, device=device)
        else:
            size_tensor = torch.zeros(1, dtype=torch.long, device=device)
            tensor = None  # Will be created after we know the size

        dist.broadcast(size_tensor, src=owner_rank)
        m = int(size_tensor.item())

        # Create zero tensor if we don't own this key
        if tensor is None:
            tensor = torch.zeros([m, m], device=device, dtype=dtype)

        # All-reduce to combine (only owner has non-zero values)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Extract our shard: [m, m] -> [m/world_size, m]
        shard_size = m // world_size
        shard = tensor[rank * shard_size : (rank + 1) * shard_size, :].contiguous()
        result_dict[key] = shard.to(device="cpu", non_blocking=True)

        del tensor
        gc.collect()

    return result_dict
