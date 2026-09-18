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
class ShampooCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for TKFAC.

    Computes:
        A_shampoo = sum over batches of Grad @ Grad.T  for activations
        S_shampoo = sum over batches of Grad.T @ Grad for gradients * trace(A_cov)


    where X is input activations [N*S, I] and G is output gradients [N*S, O].
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_shampoo_dict = {}
        self.S_shampoo_dict = {}
        self.shard_computer = ShardedMul()
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_shampoo_dict,
            gradient_covariance_dict=self.S_shampoo_dict,
            dtype=self.dtype,
            target_info=self.target_info,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""

        mask = self.collection_mask(module)
        assert mask is not None, "Collection mask not set for forward hook."

        # a: [N, S, I], collection mask: [N, S] -> select gradient-carrying positions
        a_bi = a[mask]  # [num_valid, I]

        # Augment with a ones column so the [O, I+1] per-batch gradient matches
        # the layout produced when the bias gradient is collected.
        if module._collect_bias:
            a_bi = torch.cat(
                [a_bi, a_bi.new_ones(a_bi.shape[0], 1)], dim=1
            )  # [num_valid, I+1]

        module._inputs = a_bi

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        S_shampoo_po = self.S_shampoo_dict[name]
        A_shampoo_ki = self.A_shampoo_dict[name]
        mask = self.collection_mask(module)

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = g[mask].to(self.dtype)  # [num_valid, O]
        a_bi = module._inputs.to(self.dtype)
        assert isinstance(a_bi, Tensor)

        grad_oi = torch.einsum("bi,bo->oi", a_bi, g_bo)
        local_update_ii = torch.einsum("oi,oj->ij", grad_oi, grad_oi)
        local_update_oo = torch.einsum("oi,pi->op", grad_oi, grad_oi)

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
        S_shampoo_po.add_(update_slice_po)
        A_shampoo_ki.add_(update_slice_ki)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)

        # Normalize activation covariance by trace
        for name, A_shampoo_ki in self.A_shampoo_dict.items():
            # Extract diagonal elements from this shard
            # For row i in shard, the resp. diagonal column is i + shard start
            start_row, _ = self.shard_computer.shard_bounds(A_shampoo_ki.shape[1])
            diag_indices = torch.arange(
                A_shampoo_ki.shape[0], device=A_shampoo_ki.device
            )
            diag_col_indices = diag_indices + start_row
            local_trace = A_shampoo_ki[diag_indices, diag_col_indices].sum()

            # All-reduce to get full trace
            if dist.is_initialized():
                dist.all_reduce(local_trace, op=dist.ReduceOp.SUM)

            # Divide by trace
            A_shampoo_ki.div_(local_trace)

        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        # Save sharded covariance matrices
        save_file(
            self.A_shampoo_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_shampoo_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
