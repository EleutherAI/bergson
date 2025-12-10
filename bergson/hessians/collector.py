import os
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils import assert_type, get_device


@dataclass
class HookCollectorBase(ContextDecorator, ABC):
    """
    Abstract base class for collectors that attach forward and backward hooks to model layers.

    Automatically discovers nn.Linear layers in the model, registers hooks during context entry,
    and provides lifecycle methods (setup/teardown) for subclasses to implement custom logic.

    Assumes model input shape is [N, S, I] where N=batch size, S=sequence length, I=input dimension.

    Subclasses must implement:
        - setup(): Initialize state (buffers, dicts, etc.)
        - teardown(): Clean up and save results
        - forward_hook(): Process activations during forward pass
        - backward_hook(): Process gradients during backward pass
    """

    model: nn.Module
    target_modules: set[str] | None = None
    """
    Set of module names to attach hooks to. Should consist only of nn.Linear modules.
    If None, hooks are attached to all Linear layers in the model.
    """
    valid_masks: Tensor = None  # type: ignore[assignment]
    """
    Mask of shape [N, S] indicating which positions are valid.
    Must be set via set_valid_masks() before each batch.
    """

    @staticmethod
    def discover_targets(
        model: nn.Module, target_modules: set[str] | None = None
    ) -> dict[str, tuple[torch.device, torch.Size]]:
        """
        Discover target Linear modules without instantiating a collector.

        This is useful when you need target_info early (e.g., to allocate buffers)
        before creating the actual collector instance.

        Args:
            model: The model to scan for Linear layers
            target_modules: Optional set of module names to filter. If None, all Linear layers are included.

        Returns:
            Dictionary mapping module names to (device, weight_shape) tuples

        Example:
            >>> target_info = HookCollectorBase.discover_targets(model, target_modules)
            >>> allocate_buffers(target_info)  # Use target_info before creating collector
            >>> collector = CovarianceCollector(model=model, ...)
        """
        target_info = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if target_modules is not None and name not in target_modules:
                continue

            target_info[name] = layer.weight.device, layer.weight.shape

        return target_info

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # Discover target Linear modules using the static method
        self.target_info = self.discover_targets(self.model, self.target_modules)

        # Allow subclasses to perform custom initialization
        self.setup()

    def set_valid_masks(self, masks: Tensor) -> None:
        """
        Set the valid_masks for the current batch.
        """
        self.valid_masks = masks

    def __enter__(self):
        """Register forward and backward hooks on all target modules."""
        for name in self.target_info:
            layer = self.model.get_submodule(name)

            # Store module name for use in hook callbacks
            layer._name = name  # type: ignore[attr-defined]

            # Register hooks
            fwd_hook = layer.register_forward_hook(self._save_input)
            self._fwd_hooks.append(fwd_hook)

            bwd_hook = layer.register_full_backward_hook(self._process_grad)
            self._bwd_hooks.append(bwd_hook)

        return self

    def _save_input(self, module: nn.Module, inp: tuple, _):
        """Internal forward hook that extracts input and delegates to subclass."""
        name = assert_type(str, module._name)
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        self.forward_hook(name, x)

    def _process_grad(self, module: nn.Module, _, grad_out):
        """Internal backward hook that extracts gradient and delegates to subclass."""
        assert isinstance(module, nn.Linear), "Expected a Linear module"
        name = assert_type(str, module._name)
        g = grad_out[0].detach()  # [N, S, O]

        self.backward_hook(name, g)

    def __exit__(self, exc_type, exc, tb):
        """Clean up hooks and allow subclass cleanup."""
        # Allow subclasses to save results, flush buffers, etc.
        self.teardown()

        # Clean up temporary attributes
        for layer in self.model.modules():
            if hasattr(layer, "_name"):
                del layer._name

        # Remove all registered hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()

        return False

    @abstractmethod
    def setup(self) -> None:
        """
        Called at the end of __post_init__.

        Override to perform custom initialization such as:
        - Allocating buffers or dictionaries
        - Loading pretrained weights or data
        - Initializing accumulators
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Called at the start of __exit__, before hooks are removed.

        Override to perform custom cleanup such as:
        - Saving results to disk
        - Flushing buffers
        - Computing final statistics
        - Freeing resources
        """
        pass

    @abstractmethod
    def forward_hook(self, name: str, a: torch.Tensor) -> None:
        """
        Process activations during the forward pass.

        Args:
            name: Name of the module
            a: Input activations of shape [N, S, I]
        """
        pass

    @abstractmethod
    def backward_hook(self, name: str, g: torch.Tensor) -> None:
        """
        Process gradients during the backward pass.

        Args:
            name: Name of the module
            g: Gradient with respect to module output, shape [N, S, O]
        """
        pass


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].
    """

    shard_computer: ShardedMul
    dtype: torch.dtype
    rank: int
    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}

        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_cov_dict,
            gradient_covariance_dict=self.S_cov_dict,
            dtype=self.dtype,
        )

    def forward_hook(self, name: str, a: Tensor) -> None:
        """Compute activation covariance: A^T @ A."""
        A_cov_ki = self.A_cov_dict[name]

        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        a_bi = a[self.valid_masks]  # [num_valid, I]

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

    def backward_hook(self, name: str, g: Tensor) -> None:
        """Compute gradient covariance: G^T @ G."""
        S_cov_po = self.S_cov_dict[name]

        # Reshape to [N*S, O]
        g_bo = g.reshape(-1, g.shape[-1])

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

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_covariance_sharded")
        gradient_path = os.path.join(self.path, "gradient_covariance_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)

        # Save sharded covariance matrices
        save_file(
            self.A_cov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_cov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )


@dataclass(kw_only=True)
class LambdaCollector(HookCollectorBase):
    """
    Computes eigenvalue corrections for EKFAC (Eq. 20 from paper).

    Transforms activations and gradients using precomputed eigenvectors,
    then computes outer products for diagonal correction terms.
    """

    shard_computer: ShardedMul
    path: str
    rank: int
    world_size: int
    device: torch.device

    def setup(self) -> None:
        """Load eigenvectors and initialize storage."""
        device = get_device(self.rank)

        # Load precomputed eigenvectors
        self.eigen_a = load_file(
            os.path.join(
                self.path, f"activation_eigen_sharded/shard_{self.rank}.safetensors"
            ),
            device=device,
        )
        self.eigen_g = load_file(
            os.path.join(
                self.path, f"gradient_eigen_sharded/shard_{self.rank}.safetensors"
            ),
            device=device,
        )

        # Initialize accumulators
        self.eigenvalue_corrections = {}
        self.transformed_a_cache = {}

    def forward_hook(self, name: str, a: Tensor) -> None:
        """Transform activations using eigenvectors and cache."""
        # a shape: [N, S, I]

        # Transform: a @ eigen_a
        transformed = self.shard_computer._matmul(
            vector_nsa=a, matrix_cb=self.eigen_a[name]
        )  # shape [N, S, I]

        # Cache for use in backward pass
        self.transformed_a_cache[name] = transformed

    def backward_hook(self, name: str, g: Tensor) -> None:
        """Transform gradients and compute eigenvalue corrections."""
        # g shape: [N, S, O]

        # Transform: g @ eigen_g
        transformed_g = self.shard_computer._matmul(
            vector_nsa=g, matrix_cb=self.eigen_g[name]
        )  # shape [N, S, O]

        # Compute outer product: sum_n (transformed_a_n^T @ transformed_g_n)
        # Einstein notation: [N, S, I] x [N, S, O] -> [N, O, I]
        transformed_grad_shard = torch.einsum(
            "N S I, N S O -> N O I", self.transformed_a_cache[name], transformed_g
        )

        # Square and sum over batch
        transformed_grad_shard = (transformed_grad_shard**2).sum(dim=0).contiguous()

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(transformed_grad_shard, op=dist.ReduceOp.SUM)

        # Extract our shard
        shard_size = transformed_grad_shard.shape[0] // self.world_size
        start_row = self.rank * shard_size
        end_row = (self.rank + 1) * shard_size

        # Accumulate (with CPU offloading for memory efficiency)
        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = transformed_grad_shard[
                start_row:end_row, :
            ].contiguous()
        else:
            self.eigenvalue_corrections[name] = self.eigenvalue_corrections[name].to(
                device=self.device
            )
            self.eigenvalue_corrections[name].add_(
                transformed_grad_shard[start_row:end_row, :].contiguous()
            )
            self.eigenvalue_corrections[name] = self.eigenvalue_corrections[name].to(
                device="cpu", non_blocking=False
            )

    def teardown(self) -> None:
        """Save eigenvalue corrections to disk."""
        output_path = os.path.join(self.path, "eigenvalue_correction_sharded")
        os.makedirs(output_path, exist_ok=True)

        save_file(
            self.eigenvalue_corrections,
            os.path.join(output_path, f"shard_{self.rank}.safetensors"),
        )
