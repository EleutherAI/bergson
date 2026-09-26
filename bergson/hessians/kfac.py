import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import (
    ShardedMul,
    assign_module_owners,
    gather_batch_shapes,
    gather_to_owner,
    owned_to_row_shards,
)
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].

    Distributed, each module's covariances belong to one rank, which receives
    every rank's positions for that module; teardown saves the usual row shards.
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.shard_computer = ShardedMul()
        self.A_shapes, self.S_shapes = {}, {}
        for name, (_, (out_dim, in_dim), collect_bias) in self.target_info.items():
            self.A_shapes[name] = (in_dim + collect_bias,) * 2
            self.S_shapes[name] = (out_dim, out_dim)

        # Each module's owning rank; None in a single process.
        self.owners: dict[str, int] | None = None
        if not dist.is_initialized():
            self.shard_computer._init_covariance_dict(
                activation_covariance_dict=self.A_cov_dict,
                gradient_covariance_dict=self.S_cov_dict,
                dtype=self.dtype,
                target_info=self.target_info,
            )
            return

        self.owners = assign_module_owners(self.target_info, self.world_size)
        self._rows = 0  # set per batch by with_batch
        device = self.shard_computer.device
        for name, owner in self.owners.items():
            if owner == self.rank:
                self.A_cov_dict[name] = torch.zeros(
                    self.A_shapes[name], device=device, dtype=self.dtype
                )
                self.S_cov_dict[name] = torch.zeros(
                    self.S_shapes[name], device=device, dtype=self.dtype
                )

    def with_batch(self, collection_mask: Tensor | None = None):
        super().with_batch(collection_mask)
        if self.owners is not None and collection_mask is not None:
            # Every rank pads its positions to the batch's largest count.
            counts = gather_batch_shapes(
                int(collection_mask.sum()), device=collection_mask.device
            )
            self._rows = max(count for (count,) in counts)
        return self

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""
        name = assert_type(str, module._name)
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

        self._accumulate(self.A_cov_dict, name, a_bi)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        name = assert_type(str, module._name)
        mask = self.collection_mask(module)

        # g: [N, S, O], mask: [N, S] -> select gradient-carrying positions
        g_bo = g[mask]  # [num_valid, O]

        self._accumulate(self.S_cov_dict, name, g_bo)

    def _accumulate(self, covariances: dict[str, Tensor], name: str, x: Tensor):
        """Add ``X^T @ X`` to ``name``'s covariance, where ``X`` stacks every
        rank's ``x``, in the accumulation dtype."""
        if self.owners is None:
            x = x.to(self.dtype)
            covariances[name].addmm_(x.mT, x)
            return

        # Sent in the model's dtype and cast on the owner, which moves less data.
        stacked = gather_to_owner(x, self._rows, self.owners[name])
        if stacked is not None:
            # Padding rows are zero, so they add nothing.
            stacked = stacked.to(self.dtype)
            covariances[name].addmm_(stacked.mT, stacked)

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
        for covariances, shapes, path in (
            (self.A_cov_dict, self.A_shapes, activation_path),
            (self.S_cov_dict, self.S_shapes, gradient_path),
        ):
            if self.owners is not None:
                shards = owned_to_row_shards(
                    covariances,
                    shapes,
                    self.owners,
                    self.dtype,
                    self.shard_computer.device,
                )
            else:
                shards = covariances
            save_file(shards, os.path.join(path, f"shard_{self.rank}.safetensors"))
            covariances.clear()
