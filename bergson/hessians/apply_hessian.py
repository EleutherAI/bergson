import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from simple_parsing import ArgumentParser
from torch import Tensor

from bergson.data import create_index, load_gradients
from bergson.distributed import init_dist
from bergson.hessians.sharded_computation import ShardedMul
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
    this bounds peak memory independently of the query count. Each chunk
    also pays a fixed number of broadcast rounds (modules x ranks), so
    prefer the largest chunk that fits in memory."""


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

    def compute_ivhp_sharded(self):
        eigen_a = load_file(
            self.path + f"/eigen_activation_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )
        eigen_g = load_file(
            self.path + f"/eigen_gradient_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )
        lambda_dir = (
            "eigenvalue_correction_sharded"
            if self.cfg.ev_correction
            else "eigenvalue_sharded"
        )
        lambda_factor = load_file(
            self.path + f"/{lambda_dir}/shard_{self.rank}.safetensors",
            device=self.device,
        )

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

        # Every rank computes the full result (the sharded ops broadcast and
        # accumulate), so only the main rank writes it out.
        grad_buffer = None
        if self.rank == 0:
            grad_buffer = create_index(
                Path(self.cfg.run_path),
                num_grads=info["num_grads"],
                grad_sizes=grad_sizes,
                dtype=np.float32,
            )

        num_grads = info["num_grads"]
        chunk_size = self.cfg.query_chunk_size or num_grads
        self.logger.info(
            f"Loaded gradients for {num_grads} queries and computing IVHP "
            f"in chunks of {chunk_size}..."
        )

        # Process the queries in chunks: every rank holds ~two full fp32
        # model gradients per chunk row, so this bounds peak GPU memory
        # independently of the query count. All ranks iterate chunks and
        # modules in the same order, as the sharded ops broadcast internally.
        for start in range(0, num_grads, chunk_size):
            end = min(start + chunk_size, num_grads)

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
                    grad_buffer[k][start:end] = (
                        v.to(device="cpu", non_blocking=True).flatten(1).numpy()
                    )

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
