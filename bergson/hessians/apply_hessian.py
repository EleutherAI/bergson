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


def damped_inverse(eigenvalues: Tensor, damping: float) -> Tensor:
    """Return a finite damped inverse while retaining float64 inputs."""
    mean = eigenvalues.mean()
    floor = torch.finfo(eigenvalues.dtype).tiny
    denominator = eigenvalues + damping * mean
    denominator = torch.where(
        denominator.abs() < floor, torch.full_like(denominator, floor), denominator
    )
    return denominator.reciprocal()


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
        # safetensors stays CPU resident; only the current module is moved below.
        eigen_a = load_file(
            self.path + f"/eigen_activation_sharded/shard_{self.rank}.safetensors",
        )
        eigen_g = load_file(
            self.path + f"/eigen_gradient_sharded/shard_{self.rank}.safetensors",
        )
        lambda_dir = (
            "eigenvalue_correction_sharded"
            if self.cfg.ev_correction
            else "eigenvalue_sharded"
        )
        lambda_factor = load_file(
            self.path + f"/{lambda_dir}/shard_{self.rank}.safetensors",
        )

        grad_sizes = {
            name: eigen_g[name].shape[1] * eigen_a[name].shape[1] for name in eigen_a
        }

        # A structured row can be tens of GiB for full models and NumPy's
        # structured dtype construction overflows the C int itemsize limit.
        # Keep the on-disk format unchanged but map the flat representation.
        mmap = load_gradients(self.gradient_path, structured=False)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        source_sizes = info["grad_sizes"]
        offsets = {}
        offset = 0
        for name, size in source_sizes.items():
            offsets[name] = (offset, offset + int(size))
            offset += int(size)

        base_dtype = info.get("base_dtype")
        if base_dtype is None:
            if mmap.dtype.fields is None:
                base_dtype = mmap.dtype
            else:
                base_dtype = mmap.dtype.fields[next(iter(mmap.dtype.fields))][0].base
        input_dtype = np.dtype(base_dtype)
        promoted_dtype = (
            np.float64
            if (
                input_dtype == np.float64
                or any(
                    v.dtype == torch.float64
                    for d in (eigen_a, eigen_g, lambda_factor)
                    for v in d.values()
                )
            )
            else np.float32
        )
        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=promoted_dtype,
            with_structure=False,
        )
        output_offsets = {}
        output_offset = 0
        for name, size in grad_sizes.items():
            output_offsets[name] = (output_offset, output_offset + int(size))
            output_offset += int(size)

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        # Process modules in identical sorted order on every rank. Factors and
        # transformed queries are released after each module to bound device RAM.
        output_dtype = torch.float64 if promoted_dtype == np.float64 else torch.float32
        for k in sorted(eigen_a):
            v = eigen_a[k].to(device=self.device, dtype=output_dtype)
            g = eigen_g[k].to(device=self.device, dtype=output_dtype)
            lam = lambda_factor[k].to(device=self.device, dtype=output_dtype)
            if k not in offsets:
                raise KeyError(f"Gradient index is missing module {k!r}")
            start, end = offsets[k]
            gradients_noi = torch.from_numpy(mmap[:, start:end]).to(
                device=self.device, dtype=output_dtype
            )
            gradients_noi = gradients_noi.view(-1, g.shape[1], v.shape[1])
            transformed = self.sharded_computer._matmul(
                vector_nsa=gradients_noi, matrix_cb=v
            )
            transformed = self.sharded_computer._matmul(
                vector_nsa=transformed.transpose(-2, -1), matrix_cb=g
            ).transpose(-2, -1)
            if self.apply_fn is None and not self.sharded_computer.dist:
                transformed.mul_(damped_inverse(lam, self.cfg.lambda_damp_factor))
            elif self.apply_fn is None:
                self.sharded_computer._hadamard(
                    transformed, lam, self.cfg.lambda_damp_factor
                )
            else:
                self.sharded_computer._apply_eigfn(transformed, lam, fn=self.apply_fn)
            transformed = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed.transpose(-2, -1), matrix_cb=g
            ).transpose(-2, -1)
            transformed = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed, matrix_cb=v
            )
            out_start, out_end = output_offsets[k]
            grad_buffer[:, out_start:out_end] = (
                transformed.to(device="cpu").flatten(1).numpy()
            )
            del v, g, lam, gradients_noi, transformed
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

        grad_buffer.flush()

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
