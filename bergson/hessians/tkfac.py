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
    Collects activation and gradient covariances for TKFAC.

    Computes:
        A_tcov = sum over batches of (X^T @ X)  for activations * trace(S_cov)
        S_tcov = sum over batches of (G^T @ G)  for gradients * trace(A_cov)
        trace= sum over batches of trace(X^T @ X) * trace(G^T @ G)

    where X is input activations [N*S, I] and G is output gradients [N*S, O].
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_tcov_dict = {}
        self.S_tcov_dict = {}
        self.trace_dict = {}
        self.shard_computer = ShardedMul()
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_tcov_dict,
            gradient_covariance_dict=self.S_tcov_dict,
            dtype=self.dtype,
            target_info=self.target_info,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""

        mask = self._current_valid_mask
        assert mask is not None, "Valid mask not set for forward hook."

        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        a_bi = a[mask]  # [num_valid, I]

        module._inputs = a_bi

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        S_tcov_po = self.S_tcov_dict[name]
        A_tcov_ki = self.A_tcov_dict[name]
        mask = self._current_valid_mask

        # g: [N, S, O], mask: [N, S] -> select valid positions
        g_bo = g[mask].to(self.dtype)  # [num_valid, O]
        a_bi = module._inputs.to(self.dtype)
        assert isinstance(a_bi, Tensor)

        # Compute local covariances
        local_update_oo = g_bo.mT @ g_bo
        local_update_ii = a_bi.mT @ a_bi

        # Compute traces
        trace_oo = (local_update_oo.diagonal()).sum()
        trace_ii = (local_update_ii.diagonal()).sum()

        local_update_oo = local_update_oo * trace_ii
        local_update_ii = local_update_ii * trace_oo

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_oo, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_update_ii, op=dist.ReduceOp.SUM)

        # Extract our shard
        start_row_grad = self.rank * S_tcov_po.shape[0]
        end_row_grad = (self.rank + 1) * S_tcov_po.shape[0]
        update_slice_po = local_update_oo[start_row_grad:end_row_grad, :]

        start_row_act = self.rank * A_tcov_ki.shape[0]
        end_row_act = (self.rank + 1) * A_tcov_ki.shape[0]
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
        # Save sharded covariance matrices
        save_file(
            self.A_tcov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_tcov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
        # TODO: Multiply by trace and save trace_dict
