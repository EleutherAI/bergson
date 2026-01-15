"""Ground truth collector for EKFAC testing."""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class GroundTruthCovarianceCollector(HookCollectorBase):
    activation_covariances: MutableMapping[str, Tensor]
    gradient_covariances: MutableMapping[str, Tensor]

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        name = assert_type(str, module._name)
        mask = self._current_valid_mask

        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        if mask is not None:
            a = a[mask]  # [num_valid, I]
        else:
            a = a.reshape(-1, a.shape[-1])  # [N*S, I]

        update = a.mT @ a

        if name not in self.activation_covariances:
            self.activation_covariances[name] = update
        else:
            self.activation_covariances[name].add_(update)

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        name = assert_type(str, module._name)
        mask = self._current_valid_mask

        # g: [N, S, O], valid_masks: [N, S] -> select valid positions
        if mask is not None:
            g = g[mask]  # [num_valid, O]
        else:
            g = g.reshape(-1, g.shape[-1])  # [N*S, O]

        update = g.mT @ g

        if name not in self.gradient_covariances:
            self.gradient_covariances[name] = update
        else:
            self.gradient_covariances[name].add_(update)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass


@dataclass(kw_only=True)
class GroundTruthNonAmortizedLambdaCollector(HookCollectorBase):
    eigenvalue_corrections: MutableMapping[str, Tensor]
    eigenvectors_activations: Mapping[str, Tensor]
    eigenvectors_gradients: Mapping[str, Tensor]
    device: torch.device

    def setup(self) -> None:
        self.activation_cache: dict[str, Tensor] = {}

    def teardown(self) -> None:
        self.activation_cache.clear()

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        name = assert_type(str, module._name)
        self.activation_cache[name] = a

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        name = assert_type(str, module._name)
        eigenvector_a = self.eigenvectors_activations[name].to(device=self.device)
        eigenvector_g = self.eigenvectors_gradients[name].to(device=self.device)

        activation = self.activation_cache[name]  # [N, S, I]
        gradient = g  # [N, S, O]

        gradient = torch.einsum("N S O, N S I -> N S O I", gradient, activation)

        gradient = torch.einsum("N S O I, I J -> N S O J", gradient, eigenvector_a)
        gradient = torch.einsum("O P, N S O J -> N S P J", eigenvector_g, gradient)

        gradient = gradient.sum(dim=1)  # sum over sequence length

        gradient = gradient**2
        correction = gradient.sum(dim=0)

        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = correction
        else:
            self.eigenvalue_corrections[name].add_(correction)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass


@dataclass(kw_only=True)
class GroundTruthAmortizedLambdaCollector(HookCollectorBase):
    eigenvalue_corrections: MutableMapping[str, Tensor]
    eigenvectors_activations: Mapping[str, Tensor]
    eigenvectors_gradients: Mapping[str, Tensor]
    device: torch.device

    def setup(self) -> None:
        self.activation_cache: dict[str, Tensor] = {}

    def teardown(self) -> None:
        self.activation_cache.clear()

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        name = assert_type(str, module._name)
        self.activation_cache[name] = a

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        name = assert_type(str, module._name)
        eigenvector_a = self.eigenvectors_activations[name].to(device=self.device)
        eigenvector_g = self.eigenvectors_gradients[name].to(device=self.device)

        activation = self.activation_cache[name]  # [N, S, I]

        transformed_a = torch.einsum("N S I, I J -> N S J", activation, eigenvector_a)
        transformed_g = torch.einsum("O P, N S O -> N S P", eigenvector_g, g)

        correction = (
            (torch.einsum("N S O, N S I -> N O I", transformed_g, transformed_a) ** 2)
            .sum(dim=0)
            .contiguous()
        )

        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = correction
        else:
            self.eigenvalue_corrections[name].add_(correction)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass
