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
class ActivationCovarianceCollector(HookCollectorBase):
    """
    Collects only `A_cov = E[A^T A]` for FOOF (Benzing 2022).

    FOOF approximates the Fisher as `E[aaᵀ] ⊗ I`, the gradient-side
    factor is the identity, so we don't need `E[ggᵀ]`. Combined with
    `fwd_only_factory`, this lets the collection skip the model's
    backward pass entirely.
    """

    dtype: torch.dtype
    path: str

    def setup(self) -> None:
        """Initialise the activation covariance dict only."""
        self.A_cov_dict = {}
        self.shard_computer = ShardedMul()
        for name, (_, weight_shape, _) in self.target_info.items():
            in_dim = weight_shape[1]
            shard_in_dim = (
                in_dim
                if not self.shard_computer.dist
                else in_dim // self.shard_computer.world_size
            )
            self.A_cov_dict[name] = torch.zeros(
                (shard_in_dim, in_dim),
                device=self.shard_computer.device,
                dtype=self.dtype,
            )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A. Same as CovarianceCollector."""
        name = assert_type(str, module._name)
        A_cov_ki = self.A_cov_dict[name]
        mask = self._current_valid_mask
        assert mask is not None, "Valid mask not set for forward hook."

        a_bi = a[mask].to(self.dtype)
        local_update_ii = a_bi.mT @ a_bi

        if dist.is_initialized():
            dist.all_reduce(local_update_ii, op=dist.ReduceOp.SUM)

        start_row = self.rank * A_cov_ki.shape[0]
        end_row = (self.rank + 1) * A_cov_ki.shape[0]
        update_slice_ki = local_update_ii[start_row:end_row, :]
        A_cov_ki.add_(update_slice_ki)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """No-op: FOOF doesn't need gradient covariance."""
        pass

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass

    def teardown(self) -> None:
        """Save the activation covariance shard."""
        activation_path = os.path.join(self.path, "activation_sharded")
        os.makedirs(activation_path, exist_ok=True)
        self.logger.info(f"Saving sharded activation covariance to {activation_path}")
        save_file(
            self.A_cov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
