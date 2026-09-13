import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.process_autocorrelation import process_autocorrelation_matrices


@dataclass(kw_only=True)
class AutocorrelationCollector(HookCollectorBase):
    """Fit a per-module autocorrelation Hessian approximation.

    For each target module this accumulates the per-example gradient Gram
    ``H = Σ_n vec(g_n)ᵀ vec(g_n)`` and eigendecomposes it in ``teardown`` via
    :func:`process_autocorrelation_matrices`.
    """

    data: Dataset
    """The dataset the Hessian is fit on (its row count — documents, or
    tokens under ``attribute_tokens`` — normalizes the Gram)."""

    path: str
    """Directory the fitted GradientProcessor is saved to."""

    def setup(self) -> None:
        assert self.processor.projection_target != "global", (
            "Autocorrelation Hessian fitting requires per-module projection; "
            "projection_target='global' sums all modules into a single key and "
            "has no per-module Hessian."
        )

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """Accumulate the per-module per-example gradient Gram ``PᵀP``."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g).float()
        if name in self.processor.hessians:
            self.processor.hessians[name].addmm_(P.mT, P)
        else:
            self.processor.hessians[name] = P.mT @ P

    def process_batch(self, indices: list[int], **kwargs):
        """No per-batch output; the Gram accumulates directly on the processor."""
        return

    def teardown(self):
        """Reduce/eigendecompose the accumulated Grams and save the processor."""
        grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
        if self.processor.hessians:
            process_autocorrelation_matrices(
                self.processor,
                self.processor.hessians,
                self.num_rows(self.data),
                grad_sizes,
                self.rank,
            )
        if self.rank == 0:
            self.processor.save(Path(self.path))


@dataclass(kw_only=True)
class JointAutocorrelationCollector(AutocorrelationCollector):
    """The ``structure="joint"`` autocorrelation Hessian: one Gram over the
    concatenation of every module's projected gradient — the TRAK kernel
    ``Phi^T Phi`` — instead of a Gram per module.

    Per-module projected gradients are concatenated in ``shapes()`` order (the
    order the index stores them in) and the ``[D, D]`` Gram accumulates under the
    key ``"joint"``. With ``projection_target="global"`` there is a single key
    and the joint Gram is just the ``[k, k]`` Gram of the global sketch.
    """

    def setup(self) -> None:
        """A joint Gram is fit over the global sketch too, so skip the
        per-module check in the base class."""

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """Keep this module's projected per-example gradient until the batch ends."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)
        if self.accumulate_global_projection(name, P):
            return
        self.mod_grads[name] = P.float()

    def process_batch(self, indices: list[int], **kwargs):
        """Concatenate the batch's module gradients and add their Gram."""
        G = torch.cat([self.mod_grads[name].float() for name in self.shapes()], dim=1)
        if "joint" in self.processor.hessians:
            self.processor.hessians["joint"].addmm_(G.mT, G)
        else:
            self.processor.hessians["joint"] = G.mT @ G
        self.mod_grads.clear()

    def teardown(self):
        """Reduce/eigendecompose the joint Gram and save it."""
        dim = sum(math.prod(s) for s in self.shapes().values())
        if self.processor.hessians:
            process_autocorrelation_matrices(
                self.processor,
                self.processor.hessians,
                self.num_rows(self.data),
                {"joint": dim},
                self.rank,
            )
        if self.rank == 0:
            self.processor.save(Path(self.path))
