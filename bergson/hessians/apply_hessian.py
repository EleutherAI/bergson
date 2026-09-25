import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from simple_parsing import ArgumentParser

from bergson.collector.collector import create_module_projection_matrix
from bergson.config import InversionConfig
from bergson.data import column_offsets, create_index, load_gradients
from bergson.distributed import init_dist
from bergson.gradients import GradientProcessor
from bergson.hessians.hessian_approximations import partition_modules
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
    apply_batch_size: int = 2
    """Number of query gradients moved on-device and preconditioned at a time."""
    projection_dim: int = 0
    """When set, compress each module's IVHP output to a ``[p, p]`` Kronecker
    random projection (``P_S @ (H^-1 G) @ P_A^T``)."""
    projection_type: Literal["normal", "rademacher"] = "rademacher"
    projection_scale: Literal["jl", "row_norm"] = "jl"
    projection_seed: int | None = None
    """Must match the index being scored. See ``IndexConfig``."""
    preconditioner_path: str = ""
    """Safetensors of a diagonal optimizer preconditioner (module name ->
    [out, in] grid), for the Adam SOURCE variant."""
    module_partitions: int = 1
    """Apply the inverse in this many module groups, one group's factors on the
    device at a time."""
    debug: bool = False


class EkfacApplicator:
    """Distributed runner that applies a factored inverse Hessian to stored
    gradients and writes the result to disk.

    Loads each rank's factor shards into a
    :class:`~bergson.hessians.preconditioner.FactoredPreconditioner`, reads the
    gradients from the mmap into memory, applies the preconditioner, and
    writes the transformed gradients to ``run_path``. The low-level logic lives in the
    preconditioner. Pass ``inversion_cfg`` for a standard inversion, or ``apply_fn``
    for a custom eigenvalue function (e.g. approximate unrolling). Both only with
    ``preconditioner_path``, where the EK-FAC inverse from ``inversion_cfg`` is
    applied after ``apply_fn``'s elementwise multiplier.
    """

    def __init__(
        self,
        cfg: EkfacConfig,
        inversion_cfg: InversionConfig | None = None,
        apply_fn=None,
    ):
        if (
            inversion_cfg is not None
            and apply_fn is not None
            and not cfg.preconditioner_path
        ):
            raise ValueError("Pass either inversion_cfg or apply_fn, not both.")

        if cfg.projection_dim > 0 and cfg.ev_correction:
            raise ValueError(
                "projection_dim compression is not supported with EK-FAC "
                "(ev_correction=True); set ev_correction=False or "
                "projection_dim=0."
            )

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

    def _factor_dims(self) -> tuple[list[str], dict[str, int], dict[str, int]]:
        """Module names and their [O]/[I] sizes from the eigenvector shard headers."""
        shard = f"shard_{self.rank}.safetensors"
        with safe_open(
            os.path.join(self.path, "eigen_activation_sharded", shard), framework="pt"
        ) as f:
            names = list(f.keys())
            i_dims = {n: f.get_slice(n).get_shape()[1] for n in names}
        with safe_open(
            os.path.join(self.path, "eigen_gradient_sharded", shard), framework="pt"
        ) as f:
            o_dims = {n: f.get_slice(n).get_shape()[1] for n in names}
        return names, o_dims, i_dims

    def _build_preconditioner(self, modules: list[str] | None):
        """The preconditioner chain for ``modules`` (all when ``None``)."""
        chain: list = []
        if self.cfg.preconditioner_path:
            diagonal = DiagonalFactoredPreconditioner.from_shards(
                self.path,
                self.cfg.preconditioner_path,
                rank=self.rank,
                device=self.device,
                apply_fn=self.apply_fn,
                ev_correction=self.cfg.ev_correction,
            )
            if self.inversion_cfg is not None:
                # Bae et al. App. D: "use the diagonal Hessian approximation for
                # computing the matrix exponential ... Note that we still use the
                # EK-FAC factors to compute H^-1 g in Equation 43."
                # Eq-43 reads M_mask @ H^-1 @ g_train; applied to the QUERY
                # gradient that is the adjoint H^-1 @ M_mask @ q, so the
                # diagonal mask goes first and the EK-FAC inverse second.
                chain.append(diagonal)
                preconditioner = FactoredPreconditioner.from_shards(
                    self.path,
                    rank=self.rank,
                    device=self.device,
                    inversion_cfg=self.inversion_cfg or InversionConfig(),
                    ev_correction=self.cfg.ev_correction,
                )
            else:
                preconditioner = diagonal
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
                modules=modules,
            )
        return chain, preconditioner

    def compute_ivhp_sharded(self):
        if self.cfg.preconditioner_path:
            if self.apply_fn is None:
                raise ValueError("preconditioner_path requires apply_fn.")
            if self.cfg.module_partitions > 1:
                raise ValueError(
                    "module_partitions > 1 is not supported with preconditioner_path."
                )

        names, o_dims, i_dims = self._factor_dims()

        p = self.cfg.projection_dim
        grad_sizes = {
            name: p * p if p > 0 else o_dims[name] * i_dims[name] for name in names
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

        groups = partition_modules(names, self.cfg.module_partitions)
        for group in groups:
            chain, preconditioner = self._build_preconditioner(
                group if len(groups) > 1 else None
            )
            self._apply_group(
                group,
                chain,
                preconditioner,
                mmap,
                in_offsets,
                grad_buffer,
                out_offsets,
                o_dims,
                i_dims,
            )
            del chain, preconditioner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grad_buffer.flush()
        if self.rank == 0:
            self._save_projection_settings()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")

    def _save_projection_settings(self):
        """Record how the output was projected, so scoring can check it matches
        the index."""
        # The factors include a bias column exactly when the query gradients do,
        # which widens the right projection like the collector's include_bias.
        query_cfg = Path(self.gradient_path, "processor_config.yaml")
        include_bias = (
            query_cfg.exists()
            and GradientProcessor.load_config(self.gradient_path).include_bias
        )
        GradientProcessor(
            projection_dim=self.cfg.projection_dim or None,
            projection_type=self.cfg.projection_type,
            projection_scale=self.cfg.projection_scale,
            include_bias=include_bias,
        ).save(Path(self.cfg.run_path))

    def _apply_group(
        self,
        group,
        chain,
        preconditioner,
        mmap,
        in_offsets,
        grad_buffer,
        out_offsets,
        o_dims,
        i_dims,
    ):
        """Write ``H^-1 G`` for the modules in ``group``, one module at a time."""
        p = self.cfg.projection_dim
        num_queries = mmap.shape[0]
        for start in range(0, num_queries, self.cfg.apply_batch_size):
            end = min(start + self.cfg.apply_batch_size, num_queries)

            for name in group:
                lo, hi = in_offsets[name]
                # The mmap is read-only, which torch warns about; the
                # preconditioner returns fresh tensors.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The given NumPy array is not writable",
                        category=UserWarning,
                    )
                    grads = {
                        name: numpy_to_tensor(mmap[start:end, lo:hi]).to(
                            device=self.device, dtype=torch.float32
                        )
                    }

                for pre in chain:
                    grads = pre.apply(grads)
                transformed = preconditioner.apply(grads)[name]
                del grads

                if p > 0:
                    g = transformed.view(-1, o_dims[name], i_dims[name])
                    P_l = create_module_projection_matrix(
                        name,
                        "left",
                        p,
                        o_dims[name],
                        g.dtype,
                        g.device,
                        self.cfg.projection_type,
                        self.cfg.projection_scale,
                        self.cfg.projection_seed,
                    )
                    P_r = create_module_projection_matrix(
                        name,
                        "right",
                        p,
                        i_dims[name],
                        g.dtype,
                        g.device,
                        self.cfg.projection_type,
                        self.cfg.projection_scale,
                        self.cfg.projection_seed,
                    )
                    transformed = torch.einsum("ps,nsa,ra->npr", P_l, g, P_r)

                lo, hi = out_offsets[name]
                grad_buffer[start:end, lo:hi] = transformed.flatten(1).cpu().numpy()
                del transformed
                self.logger.debug(
                    "%s: wrote H^-1 G for queries %d:%d", name, start, end
                )

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")


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
