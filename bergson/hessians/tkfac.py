import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class TraceCovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for TKFAC (Gao et al. 2021).

    TKFAC approximates the Fisher ``F = E[a aᵀ ⊗ g gᵀ]`` by ``δ Φ ⊗ Ψ`` with,
    over gradient-carrying positions t,

        Φ = Σ_t a_t a_tᵀ |g_t|²,   Ψ = Σ_t g_t g_tᵀ |a_t|²,   δ = 1 / tr(F),

    so that ``tr(δ Φ ⊗ Ψ) = tr(F)``, ``tr(F) = Σ_t |a_t|² |g_t|²``. The traces
    weight each position by its own norms; a batch's traces would mix positions.

    Teardown saves ``Φ / mean_t |g_t|²`` as the activation covariance and
    ``Ψ · Σ_t |g_t|² / tr(F)`` as the gradient covariance, sums whose product
    carries δ and which reduce to K-FAC's ``Σ a aᵀ`` and ``Σ g gᵀ`` when ``|a|``
    and ``|g|`` are independent, so they are normalized like K-FAC's.
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_tcov_dict = {}
        self.S_tcov_dict = {}
        self.shard_computer = ShardedMul()
        # Per module: positions, Σ_t |g_t|² and tr(F) = Σ_t |a_t|² |g_t|².
        self.stats = {
            name: torch.zeros(3, device=self.shard_computer.device, dtype=torch.float64)
            for name in self.target_info
        }
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_tcov_dict,
            gradient_covariance_dict=self.S_tcov_dict,
            dtype=self.dtype,
            target_info=self.target_info,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""

        mask = self.collection_mask(module)
        assert mask is not None, "Collection mask not set for forward hook."

        # a: [N, S, I], collection mask: [N, S] -> select gradient-carrying positions
        a_bi = a[mask]  # [num_valid, I]

        # Augment with a ones column so A matches the [O, I+1] gradient layout
        # produced when the bias gradient is collected.
        if module._collect_bias:
            a_bi = torch.cat(
                [a_bi, a_bi.new_ones(a_bi.shape[0], 1)], dim=1
            )  # [num_valid, I+1]

        module._inputs = a_bi

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        S_tcov_po = self.S_tcov_dict[name]
        A_tcov_ki = self.A_tcov_dict[name]
        mask = self.collection_mask(module)

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = g[mask].to(self.dtype)  # [num_valid, O]
        a_bi = module._inputs.to(self.dtype)
        assert isinstance(a_bi, Tensor)

        # Weight each position's outer products by the other factor's norm
        a_sq = a_bi.pow(2).sum(dim=1)  # [num_valid]
        g_sq = g_bo.pow(2).sum(dim=1)  # [num_valid]
        local_update_oo = (g_bo * a_sq[:, None]).mT @ g_bo
        local_update_ii = (a_bi * g_sq[:, None]).mT @ a_bi
        self.stats[name] += torch.stack(
            [a_sq.new_tensor(len(a_sq)), g_sq.sum(), (a_sq * g_sq).sum()]
        ).to(self.stats[name])

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_oo, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_update_ii, op=dist.ReduceOp.SUM)

        # Extract our shard
        start_row_grad, end_row_grad = self.shard_computer.shard_bounds(
            local_update_oo.shape[0]
        )
        update_slice_po = local_update_oo[start_row_grad:end_row_grad, :]

        start_row_act, end_row_act = self.shard_computer.shard_bounds(
            local_update_ii.shape[0]
        )
        update_slice_ki = local_update_ii[start_row_act:end_row_act, :]

        # Accumulate
        S_tcov_po.add_(update_slice_po)
        A_tcov_ki.add_(update_slice_ki)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        for name, stats in self.stats.items():
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            positions, g_sq, trace = stats.tolist()
            if trace > 0:
                self.A_tcov_dict[name].mul_(positions / g_sq)
                self.S_tcov_dict[name].mul_(g_sq / trace)
        # Save sharded covariance matrices
        save_file(
            self.A_tcov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_tcov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
