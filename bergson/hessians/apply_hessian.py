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
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger


@dataclass
class EkfacConfig:
    hessian_method_path: str
    gradient_path: str
    run_path: str
    debug: bool = False
    lambda_damp_factor: float = 0.1


class EkfacApplicator:
    def __init__(self, cfg: EkfacConfig):
        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = f"cuda:{self.rank}"

        self.sharded_computer = ShardedMul()

    def compute_ivhp_sharded(self):
        eigen_a = load_file(
            self.path + f"/eigen_activation_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/eigen_gradient_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        lambda_factor = load_file(
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
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

        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        # Forward rotation into eigenbasis: Q_S^T @ G @ Q_A
        transformed_gradients: dict[str, Tensor] = {}
        for k, v in eigen_a.items():
            gradients_noi = torch.from_numpy(mmap[k][:]).to(
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

        # Divide by damped eigenvalues in eigenbasis
        for k, v in lambda_factor.items():
            self.sharded_computer._hadamard(
                matrix_noi=transformed_gradients[k],
                lambda_ci=v,
                lambda_damp_factor=self.cfg.lambda_damp_factor,
            )

        self.logger.debug("Finished G' / lambda")
        del lambda_factor
        gc.collect()

        # Rotate back to parameter space: Q_S @ G' @ Q_A^T
        for k, v in eigen_g.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
            ).transpose(-2, -1)

        self.logger.debug("Finished Q_S @ G'")
        del eigen_g
        gc.collect()

        for k, v in eigen_a.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k], matrix_cb=v
            )

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")
        del eigen_a
        gc.collect()

        torch.cuda.synchronize()
        for k, v in transformed_gradients.items():
            grad_buffer[k][:] = v.to(device="cpu", non_blocking=True).flatten(1).numpy()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker function for distributed IVHP computation."""
    from datetime import timedelta

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

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
