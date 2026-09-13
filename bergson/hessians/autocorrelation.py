import json
import math
from dataclasses import dataclass, field
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


JOINT_LAYOUT_FILE = "joint_layout.json"
"""Module order and sizes of the concatenated gradient a joint Gram was fit on."""

MAX_JOINT_DIM = 20_000
"""Largest concatenated projected-gradient size the joint Gram accepts; the
eigendecomposition is dense in fp64."""


@dataclass(kw_only=True)
class JointAutocorrelationCollector(AutocorrelationCollector):
    """The ``structure="joint"`` autocorrelation Hessian: one Gram over the
    concatenation of every module's projected gradient — the TRAK kernel
    ``Phi^T Phi`` — instead of a Gram per module.

    Per-module projected gradients are concatenated in ``shapes()`` order (the
    order the index stores them in) and the ``[D, D]`` Gram accumulates under the
    key ``"joint"``; ``teardown`` saves it with the processor plus the layout
    that :class:`JointDensePreconditioner` uses to split gradients back into
    modules. With ``projection_target="global"`` there is a single key and the
    joint Gram is just the ``[k, k]`` Gram of the global sketch.
    """

    _layout: list[tuple[str, int]] = field(default_factory=list, init=False)

    def setup(self) -> None:
        self._layout = [(name, math.prod(s)) for name, s in self.shapes().items()]
        dim = sum(size for _, size in self._layout)
        if dim > MAX_JOINT_DIM:
            raise ValueError(
                f"The joint Gram would be {dim} x {dim} (sum of projected module "
                f"sizes); reduce projection_dim so the total is at most "
                f"{MAX_JOINT_DIM}, or use structure='per_module'."
            )

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
        G = torch.cat([self.mod_grads[name].float() for name, _ in self._layout], dim=1)
        if "joint" in self.processor.hessians:
            self.processor.hessians["joint"].addmm_(G.mT, G)
        else:
            self.processor.hessians["joint"] = G.mT @ G
        self.mod_grads.clear()

    def teardown(self):
        """Reduce/eigendecompose the joint Gram and save it with its layout."""
        dim = sum(size for _, size in self._layout)
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
            (Path(self.path) / JOINT_LAYOUT_FILE).write_text(
                json.dumps(
                    {
                        "names": [n for n, _ in self._layout],
                        "sizes": [s for _, s in self._layout],
                    }
                )
            )
