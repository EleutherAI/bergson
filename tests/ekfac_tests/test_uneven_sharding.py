"""Numerical tests for ShardedMul with unevenly sharded dimensions.

With include_bias=True the activation dimension becomes I+1, which is
generally not divisible by the world size. Rank 0 takes the remainder rows
(see shard_bounds). These tests check every sharded op against its dense
single-process reference under a 2-process gloo group on CPU.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.hessians.inversion import eigenvalue_multiplier
from bergson.hessians.sharded_computation import ShardedMul, shard_bounds

WORLD_SIZE = 2
DIM = 5  # odd on purpose: shards are 3 and 2 rows
N, S, O = 2, 3, 4


def _shard_worker(rank, world_size, port, result_dict):
    """Run all sharded ops and store rank 0's results for comparison."""
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        sharder = ShardedMul()

        # Same seeded data on every rank
        g = torch.Generator().manual_seed(0)
        matrix = torch.randn(DIM, DIM, generator=g)
        vector = torch.randn(N, S, DIM, generator=g)
        grads = torch.randn(N, O, DIM, generator=g)
        lambda_full = torch.randn(O, DIM, generator=g).abs()

        start, end = shard_bounds(DIM, rank, world_size)
        matrix_shard = matrix[start:end].contiguous()

        results = {}
        results["matmul"] = sharder._matmul(vector, matrix_shard)
        results["transpose_matmul"] = sharder._transpose_matmul(grads, matrix_shard)

        o_start, o_end = shard_bounds(O, rank, world_size)
        lambda_shard = lambda_full[o_start:o_end].contiguous()

        # Per-factor eigenvalues for factored Tikhonov (λ_A full, λ_G sharded).
        lambda_a_full = torch.randn(DIM, generator=g).abs()
        lambda_g_full = torch.randn(O, generator=g).abs()
        lambda_g_shard = lambda_g_full[o_start:o_end].contiguous()

        # The scale is now: compute this rank's inverse-eigenvalue shard via the
        # shared `eigenvalue_multiplier` (using globally-reduced means), then apply
        # it in-place across shards with `scale_rows_in_place`. This is exactly what
        # FactoredPreconditioner._scale does.
        mean = sharder.global_mean(lambda_shard, O * DIM)

        def _scaled(inverse_eigvals_shard):
            out = grads.clone()
            sharder.scale_rows_in_place(out, inverse_eigvals_shard)
            return out

        results["hadamard"] = _scaled(
            eigenvalue_multiplier("damped_inverse", lambda_shard, mean, 0.1)
        )
        results["factored_tikhonov"] = _scaled(
            eigenvalue_multiplier(
                "factored_tikhonov",
                lambda_shard,
                mean,
                0.1,
                factor_a=lambda_a_full,
                factor_g=lambda_g_shard,
                mean_a=lambda_a_full.clamp_min(0).mean(),
                mean_g=sharder.global_mean(lambda_g_shard.clamp_min(0), O),
            )
        )
        results["tikhonov_filtered"] = _scaled(
            eigenvalue_multiplier("tikhonov_filtered", lambda_shard, mean, 0.1)
        )
        results["pseudoinverse"] = _scaled(
            eigenvalue_multiplier("pseudoinverse", lambda_shard, mean, 0.1)
        )
        # Custom eigenvalue function (the approx-unrolling apply_fn path).
        results["apply_eigfn"] = _scaled(torch.rsqrt(lambda_shard))

        if rank == 0:
            result_dict.update(results)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sharded_ops_match_dense_with_uneven_shards():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    result_dict = manager.dict()
    mp.spawn(
        _shard_worker,
        args=(WORLD_SIZE, port, result_dict),
        nprocs=WORLD_SIZE,
        join=True,
    )

    # Dense references with the same seeded data
    g = torch.Generator().manual_seed(0)
    matrix = torch.randn(DIM, DIM, generator=g)
    vector = torch.randn(N, S, DIM, generator=g)
    grads = torch.randn(N, O, DIM, generator=g)
    lambda_full = torch.randn(O, DIM, generator=g).abs()

    torch.testing.assert_close(result_dict["matmul"], vector @ matrix)
    torch.testing.assert_close(result_dict["transpose_matmul"], grads @ matrix.T)

    inverse_lambda = (
        lambda_full + 0.1 * lambda_full.mean()
    ).reciprocal()  # _hadamard dense path
    torch.testing.assert_close(result_dict["hadamard"], grads * inverse_lambda)

    # Factored Tikhonov dense reference (must draw in the same generator order
    # as the worker: lambda_a_full then lambda_g_full after lambda_full).
    lambda_a_full = torch.randn(DIM, generator=g).abs()
    lambda_g_full = torch.randn(O, generator=g).abs()
    lambda_abs = 0.1 * lambda_full.mean()
    sqrt_lambda = lambda_abs.sqrt()
    pi = (lambda_a_full.mean() / lambda_g_full.mean()).sqrt()
    damped = (
        lambda_full
        + (pi * sqrt_lambda) * lambda_g_full.unsqueeze(1)
        + (sqrt_lambda / pi) * lambda_a_full.unsqueeze(0)
        + lambda_abs
    )
    torch.testing.assert_close(
        result_dict["factored_tikhonov"], grads * damped.reciprocal()
    )

    # Tikhonov-filtered and pseudoinverse dense references.
    mean = lambda_full.mean()
    alpha_sq = (0.1 * mean) ** 2
    tikhonov_mult = lambda_full / (lambda_full**2 + alpha_sq)
    torch.testing.assert_close(result_dict["tikhonov_filtered"], grads * tikhonov_mult)

    tol = 0.1 * mean
    pinv_mult = torch.where(
        lambda_full > tol, lambda_full.reciprocal(), torch.zeros_like(lambda_full)
    )
    torch.testing.assert_close(result_dict["pseudoinverse"], grads * pinv_mult)

    torch.testing.assert_close(
        result_dict["apply_eigfn"], grads * torch.rsqrt(lambda_full)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
