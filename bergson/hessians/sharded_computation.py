import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor

from bergson.utils.utils import get_device


def shard_bounds(dim: int, rank: int, world_size: int) -> tuple[int, int]:
    """Range [start, end) of ``rank``'s shard of a dimension of size
    ``dim`` split across ``world_size`` ranks.

    Rank 0 takes the remainder rows when ``dim`` is not evenly divisible,
    so the shard sizes are [base + remainder, base, ..., base].
    """
    base, remainder = divmod(dim, world_size)
    if rank == 0:
        return 0, base + remainder

    start = remainder + rank * base
    return start, start + base


class ShardedMul:
    """Distributed-aware linear-algebra primitives for applying a factored Hessian:
    eigenbasis rotations and in-place eigenvalue scaling.

    Each method has a single-process fast path (``not self.dist``) operating on
    full tensors and a distributed path operating on per-rank row-shards; both
    leave every rank holding the full result. The eigenvalue *math* lives in
    :mod:`bergson.hessians.inversion`.
    """

    def __init__(
        self,
    ):
        self.dist = dist.is_initialized()

        self.rank = dist.get_rank() if self.dist else 0
        self.world_size = dist.get_world_size() if self.dist else 1
        self.device = torch.device(get_device(self.rank))

    def shard_bounds(self, dim: int, rank: int | None = None) -> tuple[int, int]:
        """Row range [start, end) of ``rank``'s shard (default: this rank)."""
        return shard_bounds(dim, self.rank if rank is None else rank, self.world_size)

    def _init_covariance_dict(
        self,
        activation_covariance_dict: dict,
        gradient_covariance_dict: dict,
        dtype: torch.dtype,
        target_info: dict[str, tuple[torch.device, torch.Size, bool]],
    ):
        """Initialize the covariance matrices for activations and gradients."""

        for name, (device, weight_shape, collect_bias) in target_info.items():
            # Activation covariance A^T A has shape [in_dim, in_dim], or
            # [in + 1, in + 1] when the bias is collected.
            in_dim = weight_shape[1] + (1 if collect_bias else 0)
            in_start, in_end = self.shard_bounds(in_dim)
            activation_covariance_dict[name] = torch.zeros(
                (in_end - in_start, in_dim), device=self.device, dtype=dtype
            )

            # Gradient covariance G^T G has shape [out_dim, out_dim]
            out_dim = weight_shape[0]
            out_start, out_end = self.shard_bounds(out_dim)
            gradient_covariance_dict[name] = torch.zeros(
                (out_end - out_start, out_dim), device=self.device, dtype=dtype
            )

    def _matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        """Vector-matrix multiplication.
        - If not distributed, this does usual multiplication with a=c.
        - If distributed, assumes that c is this rank's shard of a and
        does sharded multiplication.
        """

        start, end = self.shard_bounds(vector_nsa.shape[2])
        assert matrix_cb.shape[0] == end - start, (
            f"Vector shape {vector_nsa.shape} not compatible with matrix shape "
            f"{matrix_cb.shape} and world_size {self.world_size}"
        )

        if not self.dist:
            result_nsb = torch.einsum(
                "n s c, c b-> n s b", vector_nsa.to(matrix_cb.dtype), matrix_cb
            )

        else:
            result_nsb = self._sharded_matmul(vector_nsa, matrix_cb)

        return result_nsb

    def _transpose_matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        if not self.dist:
            result_nsb = torch.einsum(
                "n s c, b c -> n s b", vector_nsa.to(matrix_cb.dtype), matrix_cb
            )
        else:
            result_nsb = self._sharded_transpose_matmul(vector_nsa, matrix_cb)
        return result_nsb

    def global_mean(self, x_shard: Tensor, full_numel: int) -> Tensor:
        """Mean of a row-sharded tensor over its full extent.

        ``x_shard`` is this rank's row-shard; ``full_numel`` is the element count
        of the full (un-sharded) tensor. Sums locally, all-reduces when
        distributed, then divides, so every rank gets the global mean.
        """
        total = x_shard.sum()
        if self.dist:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return total / full_numel

    def scale_rows_in_place(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        inverse_eigvals_ci: Float[Tensor, "c i"],
    ):
        """In-place scale of ``matrix_noi`` rows by ``inverse_eigvals_ci``.

        ``inverse_eigvals_ci`` is this rank's row-shard of the regularized inverse
        eigenvalues (see :func:`~bergson.hessians.inversion.eigenvalue_multiplier`).
        Non-distributed, the shard already spans all of ``o``; distributed, each
        rank's shard is broadcast and applied to its row block, so every rank ends
        with the fully-scaled ``matrix_noi``.
        """
        if not self.dist:
            matrix_noi.mul_(inverse_eigvals_ci)
        else:
            self._sharded_scale_rows_in_place(matrix_noi, inverse_eigvals_ci)

    def _sharded_scale_rows_in_place(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        inverse_eigvals_ci: Float[Tensor, "c i"],
    ):
        """Sharded in-place ``matrix_noi[:, row_block_r, :] *= inverse_eigvals_r``.

        Each rank broadcasts its ``[c, i]`` row-shard of the inverse eigenvalues in
        turn; every rank multiplies that shard into the matching ``[start:end]``
        row block, so all ranks end with the fully-scaled ``matrix_noi``.
        """
        o = matrix_noi.shape[1]
        for rank_index in range(self.world_size):
            start_row, end_row = self.shard_bounds(o, rank_index)
            if rank_index == self.rank:
                shard = inverse_eigvals_ci
            else:
                shard = torch.zeros(
                    (end_row - start_row, inverse_eigvals_ci.shape[1]),
                    device=inverse_eigvals_ci.device,
                    dtype=inverse_eigvals_ci.dtype,
                )

            dist.broadcast(shard, src=rank_index)
            matrix_noi[:, start_row:end_row, :].mul_(shard)

            if self.rank != rank_index:
                del shard

    def _sharded_matmul(
        self,
        vector_nsa: Float[Tensor, "n s a"],
        matrix_cb: Float[Tensor, "c b"],
    ) -> Float[Tensor, "n s b"]:
        """
        Sharded matrix multiplication for distributed training.
        Assumes that c is this rank's shard of a (see shard_bounds).
        vector: [n, s, a]
        matrix_shard: [c, b]
        Returns: [n, s, b]
        """
        n, s, a = vector_nsa.shape
        b = matrix_cb.shape[1]

        result_nsb = torch.zeros(
            (n, s, b),
            device=vector_nsa.device,
            dtype=matrix_cb.dtype,
        )

        for rank_index in range(self.world_size):
            start_row, end_row = self.shard_bounds(a, rank_index)
            if rank_index == self.rank:
                shard_cb = matrix_cb
            else:
                shard_cb = torch.zeros(
                    (end_row - start_row, b),
                    device=matrix_cb.device,
                    dtype=matrix_cb.dtype,
                )

            dist.broadcast(shard_cb, src=rank_index)
            result_nsb += torch.einsum(
                "n s c, c b-> n s b",
                vector_nsa[..., start_row:end_row].to(shard_cb.dtype),
                shard_cb,
            )  # [B, c]
            if self.rank != rank_index:
                del shard_cb

        return result_nsb

    def _sharded_transpose_matmul(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        matrix_bc: Float[Tensor, "b c"],
    ):
        """
        Sharded matrix multiplication for distributed training.
        Assumes that c is this rank's shard of i if left or of o if right
        (see shard_bounds).
        gradients: [n, o, i]
        matrix_shard: [c, b] where b=i if left or b=o if right
        Returns: [n, o, b] if left or [n, b, i] if right
        """

        x, y = (matrix_noi.shape[1], matrix_bc.shape[1])

        result_nxy = torch.zeros(
            matrix_noi.shape[0], x, y, device=matrix_noi.device, dtype=matrix_bc.dtype
        )

        for rank_index in range(self.world_size):
            start_row, end_row = self.shard_bounds(y, rank_index)
            if rank_index == self.rank:
                shard_bc = matrix_bc
            else:
                shard_bc = torch.zeros(
                    (end_row - start_row, matrix_bc.shape[1]),
                    device=matrix_bc.device,
                    dtype=matrix_bc.dtype,
                )
            dist.broadcast(shard_bc, src=rank_index)

            result_nxy[:, :, start_row:end_row].copy_(
                torch.einsum(
                    "n o i, c i -> n o c", matrix_noi.to(shard_bc.dtype), shard_bc
                )
            )

            if self.rank != rank_index:
                del shard_bc

        return result_nxy
