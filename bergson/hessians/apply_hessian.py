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
from bergson.data import create_index, load_gradients
from bergson.distributed import init_dist
from bergson.hessians.preconditioner import FactoredPreconditioner
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
        preconditioner = FactoredPreconditioner.from_shards(
            self.path,
            rank=self.rank,
            device=self.device,
            inversion_cfg=None if self.apply_fn is not None else self.inversion_cfg,
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

        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        # Load the gradients into memory. They are mmap'd read-only; `from_numpy`
        # uses the same buffer and warns that writes would be unsafe.
        # We never write through `grads` (the preconditioner returns fresh
        # tensors) so we suppress the warning.
        grads: dict[str, torch.Tensor] = {}
        for name in preconditioner.eigen_a:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable",
                    category=UserWarning,
                )
                grads[name] = torch.from_numpy(mmap[name][:]).to(
                    device=self.device, dtype=torch.float32
                )

        transformed = preconditioner.apply(grads)

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")

        # Stage the async D2H copies, synchronize once, then read them into the
        # (pageable) numpy buffer — the collector/builder idiom. `.numpy()` is a
        # host read, so the sync must sit between the copies and it; the earlier
        # bug synchronized *before* issuing the copies, leaving NaNs.
        staged = {
            name: v.to(device="cpu", non_blocking=True)
            for name, v in transformed.items()
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, t in staged.items():
            grad_buffer[name][:] = t.flatten(1).numpy()

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
