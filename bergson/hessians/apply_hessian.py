import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from simple_parsing import ArgumentParser

from bergson.config import InversionConfig
from bergson.data import column_offsets, create_index, load_gradients
from bergson.distributed import init_dist
from bergson.hessians.preconditioner import (
    DiagonalFactoredPreconditioner,
    FactoredPreconditioner,
)
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device, numpy_to_tensor


@dataclass
class EkfacConfig:
    hessian_method_path: str
    gradient_path: str
    run_path: str
    ev_correction: bool
    """If True, use the corrected eigenvalues, this requires
    `hessian_method_path` to have been created with
    `HessianConfig.ev_correction=True`."""
    apply_batch_size: int = 32
    """Number of query gradients moved on-device and preconditioned at a time."""
    precond_path: str = ""
    """Safetensors of a diagonal optimizer preconditioner (module name ->
    [out, in] grid). When set, ``apply_fn`` is evaluated elementwise on a
    diagonal approximation of P^1/2 H P^1/2 in parameter space (the
    Adam/AdamW approximate-unrolling variant) instead of on the eigenvalues
    in the EKFAC eigenbasis. Requires ``apply_fn``."""
    precond_post_multiply: bool = False
    """With ``precond_path``: multiply the applied function's output by the
    preconditioner grid — the P^1/2 F(M) P^1/2 sandwich of the segment
    eigenfunction. Leave False for the backward eigenfunction, whose
    P^1/2 exp(-t M) P^-1/2 sandwich cancels."""
    debug: bool = False


class EkfacApplicator:
    """Distributed runner that applies a factored inverse Hessian to stored
    gradients and writes the result to disk.

    Loads each rank's factor shards into a
    :class:`~bergson.hessians.preconditioner.FactoredPreconditioner`, reads the
    gradients from the mmap into memory, applies the preconditioner, and
    writes the transformed gradients to ``run_path``. The low-level logic lives in the
    preconditioner. Pass ``inversion_cfg`` for a standard inversion, or ``apply_fn``
    for a custom eigenvalue function (e.g. approximate unrolling) — not both.
    """

    def __init__(
        self,
        cfg: EkfacConfig,
        inversion_cfg: InversionConfig | None = None,
        apply_fn=None,
    ):
        if inversion_cfg is not None and apply_fn is not None:
            raise ValueError("Pass either inversion_cfg or apply_fn, not both.")

        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path
        self.apply_fn = apply_fn
        self.inversion_cfg = inversion_cfg

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )
        get_logger("FactoredPreconditioner", level="DEBUG" if cfg.debug else "INFO")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = get_device(self.rank)

    def compute_ivhp_sharded(self):
        if self.cfg.precond_path:
            if self.apply_fn is None:
                raise ValueError("precond_path requires apply_fn.")
            preconditioner = DiagonalFactoredPreconditioner.from_shards(
                self.path,
                self.cfg.precond_path,
                rank=self.rank,
                device=self.device,
                apply_fn=self.apply_fn,
                multiply_by_precond=self.cfg.precond_post_multiply,
                ev_correction=self.cfg.ev_correction,
            )
        else:
            preconditioner = FactoredPreconditioner.from_shards(
                self.path,
                rank=self.rank,
                device=self.device,
                inversion_cfg=(
                    None if self.apply_fn is not None else self.inversion_cfg
                ),
                apply_fn=self.apply_fn,
                ev_correction=self.cfg.ev_correction,
            )

        grad_sizes = {
            name: preconditioner.eigen_g[name].shape[1]
            * preconditioner.eigen_a[name].shape[1]
            for name in preconditioner.eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)
        in_offsets = column_offsets(info["grad_sizes"])

        num_queries = mmap.shape[0]
        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=num_queries,
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )
        out_offsets = column_offsets(grad_sizes)
        self.logger.info(
            f"Loaded gradients for {num_queries} queries and computing IVHP..."
        )

        # Precondition the queries
        for start in range(0, num_queries, self.cfg.apply_batch_size):
            end = min(start + self.cfg.apply_batch_size, num_queries)

            # The gradients are mmap'd read-only which pytorch doesn't
            # support: suppress the warning (the preconditioner returns
            # fresh tensors.
            grads: dict[str, torch.Tensor] = {}
            for name in preconditioner.eigen_a:
                lo, hi = in_offsets[name]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The given NumPy array is not writable",
                        category=UserWarning,
                    )
                    grads[name] = numpy_to_tensor(mmap[start:end, lo:hi]).to(
                        device=self.device, dtype=torch.float32
                    )

            transformed = preconditioner.apply(grads)
            del grads

            self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T batch")

            # Stage the async D2H copies, synchronize once, then read them
            # into the numpy buffer. `.numpy()` is a host read so the
            # sync must sit between it and the copies.
            staged = {
                name: v.to(device="cpu", non_blocking=True)
                for name, v in transformed.items()
            }
            del transformed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for name, t in staged.items():
                lo, hi = out_offsets[name]
                grad_buffer[start:end, lo:hi] = t.flatten(1).numpy()

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    cfg: EkfacConfig,
    inversion_cfg: InversionConfig,
):
    """Worker function for distributed IVHP computation."""
    init_dist(rank, local_rank, world_size)

    applicator = EkfacApplicator(cfg, inversion_cfg=inversion_cfg)
    applicator.compute_ivhp_sharded()


if __name__ == "__main__":
    from bergson.config import DistributedConfig
    from bergson.distributed import launch_distributed_run

    parser = ArgumentParser()
    parser.add_arguments(EkfacConfig, dest="cfg")
    parser.add_arguments(InversionConfig, dest="inversion_cfg")
    args = parser.parse_args()

    launch_distributed_run(
        "apply_hessian",
        apply_worker,
        [args.cfg, args.inversion_cfg],
        DistributedConfig(),
    )
