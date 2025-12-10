"""Ground truth collector for EKFAC testing.

This module provides a collector that computes activation and gradient covariances
for ground truth comparison, mimicking the old EkfacCollector (removed in commit 8232b77).
"""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import torch
from torch import Tensor

from bergson.hessians.collector import HookCollectorBase


@dataclass(kw_only=True)
class GroundTruthCovarianceCollector(HookCollectorBase):
    activation_covariances: MutableMapping[str, Tensor]
    gradient_covariances: MutableMapping[str, Tensor]

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def forward_hook(self, name: str, a: Tensor) -> None:
        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        a = a[self.valid_masks]  # [num_valid, I]

        update = a.mT @ a

        if name not in self.activation_covariances:
            self.activation_covariances[name] = update
        else:
            self.activation_covariances[name].add_(update)

    def backward_hook(self, name: str, g: Tensor) -> None:
        g = g.reshape(-1, g.shape[-1])  # [N*S, O]

        update = g.mT @ g

        if name not in self.gradient_covariances:
            self.gradient_covariances[name] = update
        else:
            self.gradient_covariances[name].add_(update)


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

    def forward_hook(self, name: str, a: Tensor) -> None:
        self.activation_cache[name] = a

    def backward_hook(self, name: str, g: Tensor) -> None:
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

    def forward_hook(self, name: str, a: Tensor) -> None:
        self.activation_cache[name] = a

    def backward_hook(self, name: str, g: Tensor) -> None:
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
