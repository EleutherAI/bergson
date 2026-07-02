import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from simple_parsing import ArgumentParser
from torch import Tensor

from bergson.data import create_index, load_gradients
from bergson.distributed import init_dist
from bergson.hessians.sharded_computation import ShardedMul, shard_bounds
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device


@dataclass
class EkfacConfig:
    hessian_method_path: str
    gradient_path: str
    run_path: str
    ev_correction: bool
    """If True, use the corrected eigenvalues, this requires
    `hessian_method_path` to have been created with
    `HessianConfig.ev_correction=True`."""
    debug: bool = False
    lambda_damp_factor: float = 0.1
    query_chunk_size: int = 32
    """Number of query gradients to transform at once. Every rank holds
    roughly two full fp32 model gradients per chunk row in GPU memory, so
    this bounds peak memory independently of the query count."""

    max_local_factor_gib: float = 8.0
    """When the Hessian factors total at most this many GiB, every rank
    loads the full (unsharded) factors and transforms a disjoint slice of
    the queries with purely local matmuls. This avoids the per-module
    broadcast rounds of the sharded path, which dominate wall-clock time
    for small models. Set to 0 to force the sharded path."""

    inversion: Literal["damped_inverse", "cauchy", "pseudoinverse"] = "damped_inverse"
    """Eigenvalue function used to invert the Hessian. With c =
    lambda_damp_factor and eigenvalues lambda:

    - "damped_inverse" (default): 1 / (lambda + c*mean(lambda)) — uniform
      Tikhonov damping.
    - "cauchy": lambda / (lambda^2 + (c*mean(lambda))^2) — Lorentzian-
      filtered inverse.
    - "pseudoinverse": 1/lambda where lambda > c*mean(lambda), else 0 —
      truncated Moore-Penrose pseudoinverse.

    Non-default inversions require the local (unsharded-factor) path, where
    each module's full eigenvalue grid is on one device."""


# Inversion modes expressible as a pure function of (lambda, mean(lambda), c),
# ported from the damping-transfer fork's bergson/hessians/inversion.py.
INVERSION_FNS = {
    "cauchy": lambda lam, c: lam / (lam * lam + (c * lam.mean()) ** 2),
    "pseudoinverse": lambda lam, c: torch.where(
        lam > c * lam.mean(), lam.reciprocal(), torch.zeros_like(lam)
    ),
}


class EkfacApplicator:
    def __init__(self, cfg: EkfacConfig, apply_fn=None):
        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path
        self.apply_fn = apply_fn

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = get_device(self.rank)

        self.sharded_computer = ShardedMul()

    def _load_factors(self, subdir: str, full: bool) -> dict[str, Tensor]:
        """Load this rank's factor shard, or concatenate all shards into the
        full matrices when ``full`` (shards are row ranges in rank order)."""
        path = Path(self.path) / subdir
        if not full:
            return load_file(
                str(path / f"shard_{self.rank}.safetensors"), device=self.device
            )

        shard_paths = sorted(
            path.glob("shard_*.safetensors"), key=lambda p: int(p.stem.split("_")[1])
        )
        shards = [load_file(str(p), device=self.device) for p in shard_paths]
        return {k: torch.cat([s[k] for s in shards], dim=0) for k in shards[0]}

    def compute_ivhp_sharded(self):
        lambda_dir = (
            "eigenvalue_correction_sharded"
            if self.cfg.ev_correction
            else "eigenvalue_sharded"
        )

        # When the full factors fit comfortably on one GPU, shard the
        # *queries* over ranks instead of the factor rows: each rank
        # transforms its own slice with local matmuls and writes its own
        # rows, with no collectives. The sharded path pays a broadcast round
        # per (module, rank) per chunk, which dominates for small models.
        factor_bytes = sum(
            f.stat().st_size
            for d in ("eigen_activation_sharded", "eigen_gradient_sharded", lambda_dir)
            for f in (Path(self.path) / d).glob("shard_*.safetensors")
        )
        local = (
            self.world_size > 1
            and factor_bytes <= self.cfg.max_local_factor_gib * 2**30
        )
        if local:
            self.logger.info(
                f"Factors total {factor_bytes / 2**30:.2f} GiB; sharding queries "
                "over ranks with local matmuls"
            )
            self.sharded_computer = ShardedMul(local=True)

        if self.cfg.inversion != "damped_inverse" and self.apply_fn is None:
            if not local and self.world_size > 1:
                raise ValueError(
                    f"inversion={self.cfg.inversion!r} needs the full eigenvalue "
                    "grid per module (its mean); use the local path."
                )
            inversion_fn = INVERSION_FNS[self.cfg.inversion]
            damp = self.cfg.lambda_damp_factor
            self.apply_fn = lambda lam: inversion_fn(lam, damp)
            self.logger.info(f"Using {self.cfg.inversion} inversion (c={damp})")

        eigen_a = self._load_factors("eigen_activation_sharded", full=local)
        eigen_g = self._load_factors("eigen_gradient_sharded", full=local)
        lambda_factor = self._load_factors(lambda_dir, full=local)

        for k, v in lambda_factor.items():
            eigen_a[k] = eigen_a[k].to(dtype=torch.float32)
            eigen_g[k] = eigen_g[k].to(dtype=torch.float32)
            lambda_factor[k] = v.to(dtype=torch.float32)

        grad_sizes = {
            name: eigen_g[name].shape[1] * eigen_a[name].shape[1] for name in eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        num_grads = info["num_grads"]

        # In local mode each rank writes its own query rows; in sharded mode
        # every rank computes the full result (the sharded ops broadcast and
        # accumulate), so only the main rank writes it out.
        grad_buffer = None
        if local or self.rank == 0:
            grad_buffer = create_index(
                Path(self.cfg.run_path),
                num_grads=num_grads,
                grad_sizes=grad_sizes,
                dtype=np.float32,
            )

        if local:
            row_start, row_end = shard_bounds(num_grads, self.rank, self.world_size)
        else:
            row_start, row_end = 0, num_grads

        chunk_size = self.cfg.query_chunk_size or num_grads
        self.logger.info(
            f"Loaded gradients for {num_grads} queries and computing IVHP "
            f"for rows {row_start}:{row_end} in chunks of {chunk_size}..."
        )

        # Process the queries in chunks: every rank holds ~two full fp32
        # model gradients per chunk row, so this bounds peak GPU memory
        # independently of the query count. In sharded mode all ranks iterate
        # chunks and modules in the same order, as the sharded ops broadcast
        # internally.
        for start in range(row_start, row_end, chunk_size):
            end = min(start + chunk_size, row_end)

            # One contiguous read of the chunk's records: per-module field
            # slices of the structured memmap fault scattered pages across
            # the whole file, which is pathologically slow on network
            # filesystems.
            records = np.array(mmap[start:end])

            # Forward rotation into eigenbasis: Q_S^T @ G @ Q_A
            transformed_gradients: dict[str, Tensor] = {}
            for k, v in eigen_a.items():
                gradients_noi = torch.from_numpy(records[k]).to(
                    device=self.device, dtype=torch.float32
                )
                gradients_noi = gradients_noi.view(
                    -1, eigen_g[k].shape[1], eigen_a[k].shape[1]
                )
                transformed_gradients[k] = self.sharded_computer._matmul(
                    vector_nsa=gradients_noi, matrix_cb=v
                )

            self.logger.debug("Finished G @ Q_A")

            for k, v in eigen_g.items():
                transformed_gradients[k] = self.sharded_computer._matmul(
                    vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
                ).transpose(-2, -1)

            self.logger.debug("Finished G' = Q_S^T @ G @ Q_A")

            # Apply eigenvalue function in eigenbasis (default = damped inverse).
            for k, v in lambda_factor.items():
                if self.apply_fn is None:
                    self.sharded_computer._hadamard(
                        matrix_noi=transformed_gradients[k],
                        lambda_ci=v,
                        lambda_damp_factor=self.cfg.lambda_damp_factor,
                    )
                else:
                    self.sharded_computer._apply_eigfn(
                        matrix_noi=transformed_gradients[k],
                        lambda_ci=v,
                        fn=self.apply_fn,
                    )

            self.logger.debug("Finished G' / lambda")

            # Rotate back to parameter space: Q_S @ G' @ Q_A^T
            for k, v in eigen_g.items():
                transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                    vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
                ).transpose(-2, -1)

            self.logger.debug("Finished Q_S @ G'")

            for k, v in eigen_a.items():
                transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                    vector_nsa=transformed_gradients[k], matrix_cb=v
                )

            self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")

            torch.cuda.synchronize()
            if grad_buffer is not None:
                for k, v in transformed_gradients.items():
                    # Blocking copy: a non_blocking GPU->CPU transfer is not
                    # finished when .numpy() reads the buffer, silently
                    # corrupting a random subset of modules.
                    grad_buffer[k][start:end] = v.cpu().flatten(1).numpy()

            self.logger.info(f"Transformed queries {start}:{end} / {num_grads}")
            del transformed_gradients, records
            gc.collect()

        del eigen_a, eigen_g, lambda_factor
        if grad_buffer is not None:
            grad_buffer.flush()
        if dist.is_initialized():
            # Ranks read the index back in the score step; wait for the write.
            dist.barrier()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker function for distributed IVHP computation."""
    init_dist(rank, local_rank, world_size)

    applicator = EkfacApplicator(cfg)
    applicator.compute_ivhp_sharded()


if __name__ == "__main__":
    from bergson.config import DistributedConfig
    from bergson.distributed import launch_distributed_run

    parser = ArgumentParser()
    parser.add_arguments(EkfacConfig, dest="cfg")
    args = parser.parse_args()

    launch_distributed_run(
        "apply_hessian",
        apply_worker,
        [args.cfg],
        DistributedConfig(),
    )
