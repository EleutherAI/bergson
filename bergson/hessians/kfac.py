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
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.shard_computer = ShardedMul()
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_cov_dict,
            gradient_covariance_dict=self.S_cov_dict,
            dtype=self.dtype,
            target_info=self.target_info,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""
        name = assert_type(str, module._name)
        A_cov_ki = self.A_cov_dict[name]
        mask = self._current_valid_mask
        assert mask is not None, "Valid mask not set for forward hook."

        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        a_bi = a[mask]  # [num_valid, I]

        # Compute local covariance
        local_update_ii = a_bi.mT @ a_bi

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_ii, op=dist.ReduceOp.SUM)

        # Extract our shard
        start_row = self.rank * A_cov_ki.shape[0]
        end_row = (self.rank + 1) * A_cov_ki.shape[0]
        update_slice_ki = local_update_ii[start_row:end_row, :]

        # Accumulate
        A_cov_ki.add_(update_slice_ki)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        S_cov_po = self.S_cov_dict[name]
        mask = self._current_valid_mask

        # g: [N, S, O], mask: [N, S] -> select valid positions
        g_bo = g[mask]  # [num_valid, O]

        # Compute local covariance
        local_update_oo = g_bo.mT @ g_bo

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_oo, op=dist.ReduceOp.SUM)

        # Extract our shard
        start_row = self.rank * S_cov_po.shape[0]
        end_row = (self.rank + 1) * S_cov_po.shape[0]
        update_slice_po = local_update_oo[start_row:end_row, :]

        # Accumulate
        S_cov_po.add_(update_slice_po)

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
            self.A_cov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_cov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
