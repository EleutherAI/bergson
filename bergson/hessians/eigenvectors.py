import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from tqdm import tqdm

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import (
    ShardedMul,
    assign_module_owners,
    gather_batch_shapes,
    gather_to_owner,
    owned_to_row_shards,
    shard_bounds,
)
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

    Distributed, each module belongs to one rank, which joins its full
    eigenvectors from every rank's shard and receives every rank's positions
    for it; teardown saves the usual row shards.
    """

    path: str
    eigen_path: str | None = None
    """Override read location for eigvecs; defaults to ``self.path``."""

    output_subdir: str = "eigenvalue_correction_sharded"
    """Subdir under ``self.path`` to write lambda shards into."""

    dtype: torch.dtype = torch.float32
    """Accumulation dtype for the eigenvalue corrections. Activations,
    gradients, and eigenvectors are cast to this before the rotation and
    squared accumulation, and shards are saved in it."""

    num_documents: int = 1
    """Divides the summed corrections into a mean over documents."""

    def setup(self) -> None:
        """Load eigenvectors and initialize storage."""
        self.shard_computer = ShardedMul()
        self.device = get_device(self.rank)

        eigen_src = self.eigen_path or self.path

        # Initialize accumulators
        self.eigenvalue_corrections = {}
        self.transformed_a_cache = {}
        # Each module's owning rank; None in a single process.
        self.owners: dict[str, int] | None = None

        if dist.is_initialized():
            self._setup_owned(eigen_src)
            return

        # Each on its module's device, in the accumulation dtype.
        self.eigen_a, self.eigen_g = (
            {
                k: vectors[k].to(self.target_info[k][0], self.dtype)
                for k in self.target_info
            }
            for vectors in (
                load_file(
                    os.path.join(eigen_src, subdir, f"shard_{self.rank}.safetensors")
                )
                for subdir in ("eigen_activation_sharded", "eigen_gradient_sharded")
            )
        )

    def _setup_owned(self, eigen_src: str) -> None:
        """Load the full eigenvectors of the modules this rank owns, joining the
        row shards every rank wrote."""
        self.owners = assign_module_owners(self.target_info, self.world_size)
        owned = [name for name, owner in self.owners.items() if owner == self.rank]

        def load(subdir: str) -> dict[str, Tensor]:
            files = [
                os.path.join(eigen_src, subdir, f"shard_{rank}.safetensors")
                for rank in range(self.world_size)
            ]
            full: dict[str, list[Tensor]] = {name: [] for name in owned}
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for name in owned:
                        full[name].append(f.get_tensor(name))
            return {
                name: torch.cat(rows).to(self.device, self.dtype)
                for name, rows in full.items()
            }

        self.eigen_a = load("eigen_activation_sharded")
        self.eigen_g = load("eigen_gradient_sharded")
        self.correction_shapes = {
            name: (out_dim, in_dim + collect_bias)
            for name, (_, (out_dim, in_dim), collect_bias) in self.target_info.items()
        }
        # Every owned module has a correction to send at teardown, even one whose
        # hooks never ran.
        self.eigenvalue_corrections = {
            name: torch.zeros(
                self.correction_shapes[name], device=self.device, dtype=self.dtype
            )
            for name in owned
        }

    def _gather_rotated(self, x: Tensor, eigenvectors: dict[str, Tensor], name: str):
        """Every rank's ``x`` [N, S, D] rotated by ``name``'s eigenvectors, as one
        [N_r, S_r, D] tensor per rank on the owner; ``None`` elsewhere."""
        # Per module, since fused MoE experts see their own [N, S].
        assert self.owners is not None
        shapes = gather_batch_shapes(*x.shape[:2], device=x.device)
        rows = max(n * s for n, s in shapes)
        stacked = gather_to_owner(x.flatten(0, 1), rows, self.owners[name])
        if stacked is None:
            return None
        rotated = stacked.to(self.dtype) @ eigenvectors[name]
        return [
            rotated[r * rows : r * rows + n * s].view(n, s, -1)
            for r, (n, s) in enumerate(shapes)
        ]

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Transform activations using eigenvectors and cache."""
        name = assert_type(str, module._name)
        # a shape: [N, S, I]

        # Augment with a ones column to match the [I+1, I+1] activation
        # covariance eigenvectors computed when the bias gradient is collected.
        if module._collect_bias:
            a = torch.cat([a, a.new_ones(*a.shape[:-1], 1)], dim=-1)  # [N, S, I+1]

        if self.owners is not None:
            self.transformed_a_cache[name] = self._gather_rotated(a, self.eigen_a, name)
            return

        # Transform: a @ eigen_a
        transformed = self.shard_computer._matmul(
            vector_nsa=a.to(self.dtype), matrix_cb=self.eigen_a[name]
        )  # shape [N, S, I]

        # Cache for use in backward pass
        self.transformed_a_cache[name] = transformed

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Transform gradients and compute eigenvalue corrections."""
        name = assert_type(str, module._name)
        # g shape: [N, S, O]

        if self.owners is not None:
            rotated_g = self._gather_rotated(g, self.eigen_g, name)
            rotated_a = self.transformed_a_cache.pop(name)
            if rotated_g is None:
                return
            # Square each document's rotated gradient and sum, as below.
            for a_r, g_r in zip(rotated_a, rotated_g):
                self.eigenvalue_corrections[name].add_(
                    (torch.einsum("N S I, N S O -> N O I", a_r, g_r) ** 2).sum(dim=0)
                )
            return

        # Transform: g @ eigen_g
        transformed_g = self.shard_computer._matmul(
            vector_nsa=g.to(self.dtype), matrix_cb=self.eigen_g[name]
        )  # shape [N, S, O]

        # Compute outer product: sum_n (transformed_a_n^T @ transformed_g_n)
        # Einstein notation: [N, S, I] x [N, S, O] -> [N, O, I]
        transformed_grad_shard = torch.einsum(
            "N S I, N S O -> N O I", self.transformed_a_cache[name], transformed_g
        )

        # Square and sum over batch
        transformed_grad = (transformed_grad_shard**2).sum(dim=0)

        # Accumulate (with CPU offloading for memory efficiency)
        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = transformed_grad.to(device="cpu")
        else:
            self.eigenvalue_corrections[name] = self.eigenvalue_corrections[name].to(
                device=transformed_grad.device
            )
            self.eigenvalue_corrections[name].add_(transformed_grad)
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

        # The eigenvectors are no longer needed; free them before the transfers.
        self.eigen_a.clear()
        self.eigen_g.clear()
        corrections = self.eigenvalue_corrections
        if self.owners is not None:
            corrections = owned_to_row_shards(
                corrections,
                self.correction_shapes,
                self.owners,
                self.dtype,
                self.device,
            )
        save_file(
            {k: v / self.num_documents for k, v in corrections.items()},
            os.path.join(output_path, f"shard_{self.rank}.safetensors"),
        )
        self.transformed_a_cache.clear()
        self.eigenvalue_corrections.clear()


def _eigh(
    key: str, matrix: Tensor, total_processed: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Eigenvalues and eigenvectors of ``matrix / total_processed``, computed in
    fp64 on ``total_processed``'s device and returned there in ``dtype``."""
    matrix_normalized = matrix.to(total_processed.device, torch.float64)
    # Free the caller's copy early when it passed the only reference.
    del matrix
    matrix_normalized = matrix_normalized + matrix_normalized.T
    matrix_normalized.div_(2 * total_processed)

    if not torch.isfinite(matrix_normalized).all():
        raise ValueError(
            f"Covariance matrix for {key} contains NaNs or Infs. "
            "Consider using fp32."
        )

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)
    except Exception as e:
        raise RuntimeError(f"Eigendecomposition failed for {key}") from e

    del matrix_normalized
    return eigenvalues.to(dtype), eigenvectors.to(dtype).contiguous()


def _module_count(
    key: str, total_processed: Tensor, num_positions: dict[str, int] | None
) -> Tensor:
    """The number of positions ``key``'s covariances summed, on
    ``total_processed``'s device. An expert no position was routed to has zero
    covariances, so its count is clamped to one rather than dividing by zero."""
    if num_positions is None:
        return total_processed
    return torch.tensor(max(num_positions[key], 1), device=total_processed.device)


def compute_eigendecomposition(
    covariance_path: str,
    total_processed: int | Tensor,
    num_positions: dict[str, int] | None = None,
    all_gpus: bool = False,
) -> dict[str, Tensor]:
    """Eigendecompose the covariances saved under ``covariance_path`` (Eq. 18).

    Each covariance belongs to one rank, chosen to balance the O(d³) cost, which
    joins it from every rank's row shard. With ``all_gpus``, a single process
    spreads the covariances over every visible GPU instead, and each GPU
    decomposes its own in a thread, one at a time. Each covariance is divided
    by its entry in ``num_positions``, the positions it summed, when given,
    else by ``total_processed``.

    Saves this rank's row shard of every eigenvector matrix under the
    ``eigen_`` sibling of ``covariance_path``, and returns this rank's row shard
    of every module's eigenvalues on the CPU.
    """
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    files = [
        os.path.join(covariance_path, f"shard_{r}.safetensors")
        for r in range(world_size)
    ]

    with safe_open(files[0], framework="pt") as f:
        keys = list(f.keys())
        dtype = f.get_tensor(keys[0]).dtype if keys else torch.float32
        dims = {key: f.get_slice(key).get_shape()[-1] for key in keys}

    owned = keys
    owners = None
    if distributed:
        assignment = fair_distribute_by_cost(dims, world_size)
        owners = {
            key: r for r, ranks_keys in enumerate(assignment) for key in ranks_keys
        }
        owned = assignment[rank]
        devices = [torch.device(get_device(rank))]
    elif all_gpus and torch.cuda.device_count() > 1:
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device(get_device(0))]
    per_device = fair_distribute_by_cost({k: dims[k] for k in owned}, len(devices))

    eigenvectors: dict[str, Tensor] = {}
    eigenvalues: dict[str, Tensor] = {}
    progress = tqdm(
        total=len(owned),
        desc=f"Rank {rank}: Computing eigenvectors",
        position=rank,
        leave=False,
    )

    def decompose(device: torch.device, device_keys: list[str]):
        # Threads start on the default device.
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            for key in device_keys:
                rows = []
                for path in files:
                    with safe_open(path, framework="pt", device="cpu") as f:
                        rows.append(f.get_tensor(key))
                matrix = torch.cat(rows).to(device)
                del rows
                count = _module_count(
                    key, torch.as_tensor(total_processed, device=device), num_positions
                )
                values, vectors = _eigh(key, matrix, count, dtype)
                del matrix
                eigenvectors[key] = vectors.cpu()
                eigenvalues[key] = values.cpu()
                del values, vectors
                progress.update()

    if devices[0].type == "cuda":
        # PyTorch loads its GPU linear algebra library on first use, which isn't
        # thread safe.
        torch.linalg.eigh(torch.eye(2, device=devices[0]))
    try:
        with ThreadPoolExecutor(len(devices)) as pool:
            futures = [
                pool.submit(decompose, device, device_keys)
                for device, device_keys in zip(devices, per_device)
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        progress.close()

    value_shards = eigenvalues
    if owners is not None:
        device = devices[0]
        eigenvectors = owned_to_row_shards(
            eigenvectors, {k: (m, m) for k, m in dims.items()}, owners, dtype, device
        )
        value_shards = owned_to_row_shards(
            eigenvalues, {k: (m,) for k, m in dims.items()}, owners, dtype, device
        )

    output_path = os.path.join(
        os.path.dirname(covariance_path), "eigen_" + os.path.basename(covariance_path)
    )
    os.makedirs(output_path, exist_ok=True)
    save_file(eigenvectors, os.path.join(output_path, f"shard_{rank}.safetensors"))
    get_logger().info(f"Saved eigenvectors to {output_path}")
    return value_shards


def save_uncorrected_eigenvalues(
    partial_run_path: str | os.PathLike,
    eigenvalues_a: dict[str, Tensor],
    eigenvalues_g: dict[str, Tensor],
    total_processed: int | Tensor,
    rank: int,
    world_size: int,
    num_documents: int = 1,
    num_positions: dict[str, int] | None = None,
) -> None:
    """Take sharded eigenvalues_a and eigenvalues_g. Computes the sharded
    outer product by keeping eigenvalues_g in its sharded form and gathering
    eigenvalues_a to all ranks.

    The factor covariances are averaged over the dataset's tokens, so we scale up
    by ``total_processed`` to a sum over all tokens, then divide by
    ``num_documents`` to get the average document value. When ``num_positions``
    is given, each module's own position count, which the eigendecomposition
    averaged over, replaces ``total_processed``.
    """
    out_dir = os.path.join(str(partial_run_path), "eigenvalue_sharded")
    os.makedirs(out_dir, exist_ok=True)

    # Per-factor eigenvalues, saved to enable factored Tikhonov damping.
    # This damping method uses the individual eigenvalues of the A and G factors.
    # `factor_eig_a` holds the full activation
    # eigenvalues λ_A [I] (replicated on every rank); `factor_eig_g` holds this
    # rank's part of the row-sharded gradient eigenvalues λ_G [O].
    factor_a_dir = os.path.join(str(partial_run_path), "factor_eig_a")
    factor_g_dir = os.path.join(str(partial_run_path), "factor_eig_g")
    os.makedirs(factor_a_dir, exist_ok=True)
    os.makedirs(factor_g_dir, exist_ok=True)

    device = get_device(rank)
    # Mirror compute_eigendecomposition: keep total_processed in its native
    # dtype on the right device. PyTorch type promotion handles float * int
    # cleanly so the outer product stays in the eigenvalues' dtype rather
    # than getting upcast to float32 / float64 by an explicit .float() cast.
    if isinstance(total_processed, int):
        total_processed = torch.tensor(total_processed, device=device)
    else:
        total_processed = total_processed.to(device)

    outer_product_sharded: dict[str, Tensor] = {}
    factor_eig_a: dict[str, Tensor] = {}
    factor_eig_g: dict[str, Tensor] = {}
    for key, eigenvalue_g_shard in eigenvalues_g.items():
        eigenvalue_g_shard = eigenvalue_g_shard.to(device)
        eigenvalue_a_shard = eigenvalues_a[key].to(device)

        if world_size > 1:
            # Shards may be uneven, so sum the shard sizes to get the full dimension
            # then broadcast each rank's shard into place.
            full_dim = torch.tensor(eigenvalue_a_shard.shape[0], device=device)
            dist.all_reduce(full_dim, op=dist.ReduceOp.SUM)
            m = int(full_dim.item())

            eigenvalue_a_full = torch.empty(
                m, device=device, dtype=eigenvalue_a_shard.dtype
            )
            for rank_index in range(world_size):
                start_row, end_row = shard_bounds(m, rank_index, world_size)
                if rank_index == rank:
                    shard = eigenvalue_a_shard.contiguous()
                else:
                    shard = torch.empty(
                        end_row - start_row,
                        device=device,
                        dtype=eigenvalue_a_shard.dtype,
                    )
                dist.broadcast(shard, src=rank_index)
                eigenvalue_a_full[start_row:end_row] = shard
        else:
            eigenvalue_a_full = eigenvalue_a_shard

        count = _module_count(key, total_processed, num_positions)
        per_document = count / num_documents
        outer = torch.outer(eigenvalue_g_shard, eigenvalue_a_full) * per_document
        outer_product_sharded[key] = outer.to(device="cpu").contiguous()

        # Scale the per-factor eigenvalues by the same factor as the outer
        # product; splitting it as sqrt keeps λ_A·λ_G == outer.
        scale = per_document.to(eigenvalue_a_full.dtype).sqrt()
        factor_eig_a[key] = (eigenvalue_a_full * scale).to(device="cpu").contiguous()
        factor_eig_g[key] = (eigenvalue_g_shard * scale).to(device="cpu").contiguous()

    save_file(
        outer_product_sharded,
        os.path.join(out_dir, f"shard_{rank}.safetensors"),
    )
    save_file(factor_eig_a, os.path.join(factor_a_dir, f"shard_{rank}.safetensors"))
    save_file(factor_eig_g, os.path.join(factor_g_dir, f"shard_{rank}.safetensors"))

    get_logger().info(f"Saved uncorrected eigenvalues to {out_dir}")
