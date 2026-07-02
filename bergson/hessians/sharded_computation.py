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
    def __init__(self, local: bool = False):
        # ``local`` forces the single-process code paths even under
        # torch.distributed, for callers that hold full (unsharded) matrices
        # and partition their work across ranks themselves.
        self.dist = dist.is_initialized() and not local

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

    def _hadamard(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        lambda_ci: Float[Tensor, "c i"],
        lambda_damp_factor: float = 0.1,
    ):
        if not self.dist:
            global_lambda_mean = lambda_ci.mean()
            inverse_lambda = (
                lambda_ci + lambda_damp_factor * global_lambda_mean
            ).reciprocal()
            matrix_noi.mul_(inverse_lambda)
        else:
            self._sharded_hadamard(matrix_noi, lambda_ci, lambda_damp_factor)

    def _apply_eigfn(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        lambda_ci: Float[Tensor, "c i"],
        fn,
    ):
        """In-place: matrix_noi[:, row_block_r, :] *= fn(λ_r) per rank r."""
        if not self.dist:
            matrix_noi.mul_(fn(lambda_ci))
        else:
            self._sharded_apply_eigfn(matrix_noi, lambda_ci, fn)

    def _sharded_apply_eigfn(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        lambda_ci: Float[Tensor, "c i"],
        fn,
    ):
        """Sharded in-place ``matrix_noi *= fn(λ)`` (function-aware hadamard)."""
        o = matrix_noi.shape[1]
        for rank_index in range(self.world_size):
            start_row, end_row = self.shard_bounds(o, rank_index)
            if rank_index == self.rank:
                shard_ci = lambda_ci
            else:
                shard_ci = torch.zeros(
                    (end_row - start_row, lambda_ci.shape[1]),
                    device=lambda_ci.device,
                    dtype=lambda_ci.dtype,
                )

            dist.broadcast(shard_ci, src=rank_index)

            matrix_noi[:, start_row:end_row, :].mul_(fn(shard_ci))

            if self.rank != rank_index:
                del shard_ci

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

    def _sharded_hadamard(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        lambda_ci: Float[Tensor, "c i"],
        lambda_damp_factor: float = 0.1,
    ):
        """
        Sharded in-place element-wise multiplication for distributed training.
        gradients: [n, o, i]
        matrix_shard: [c, i] where c is this rank's shard of o (see
        shard_bounds)

        """
        o = matrix_noi.shape[1]

        # Shards may be uneven, so compute the global mean from the global sum
        global_lambda_mean = lambda_ci.sum()
        dist.all_reduce(global_lambda_mean, op=dist.ReduceOp.SUM)
        global_lambda_mean /= o * lambda_ci.shape[1]

        for rank_index in range(self.world_size):
            start_row, end_row = self.shard_bounds(o, rank_index)
            if rank_index == self.rank:
                shard_ci = lambda_ci
            else:
                shard_ci = torch.zeros(
                    (end_row - start_row, lambda_ci.shape[1]),
                    device=lambda_ci.device,
                    dtype=lambda_ci.dtype,
                )

            dist.broadcast(shard_ci, src=rank_index)

            inverse_lambda = (
                shard_ci + lambda_damp_factor * global_lambda_mean
            ).reciprocal()

            matrix_noi[:, start_row:end_row, :].mul_(inverse_lambda)

            if self.rank != rank_index:
                del shard_ci

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
