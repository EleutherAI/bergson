from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dc_field

import torch
import torch.distributed as dist
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor, nn

from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig
from bergson.gradients import AdafactorNormalizer, AdamNormalizer, LayerAdapter
from bergson.process_preconditioners import process_preconditioners
from bergson.utils.utils import assert_type, get_gradient_dtype


@dataclass
class InMemoryCollector(HookCollectorBase):
    """Simple collector that accumulates gradients in memory for benchmarking."""

    gradients: dict[str, list[torch.Tensor]] = dc_field(
        default_factory=lambda: defaultdict(list)
    )
    """Accumulated gradients keyed by module name."""

    def __init__(self, *args, **kwargs):
        self.data = assert_type(Dataset, kwargs["data"])
        self.cfg = assert_type(IndexConfig, kwargs["cfg"])

        self.reduce_cfg = kwargs.get("reduce_cfg", None)
        self.builder = kwargs.get("builder", None)
        self.scorer = kwargs.get("scorer", None)

        # Extract parent class arguments
        parent_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "model",
                "filter_modules",
                "target_modules",
                "processor",
                "attention_cfgs",
            }
        }
        parent_kwargs["filter_modules"] = self.cfg.filter_modules

        super().__init__(*args, **parent_kwargs)

    def setup(self) -> None:
        """Initialize gradient storage."""
        assert isinstance(
            self.model.device, torch.device
        ), "Model device is not set correctly"
        if self.cfg.include_bias and self.processor.normalizers is not None:
            raise NotImplementedError(
                "Bias with normalizers not supported yet, "
                "consider disabling bias inclusion for now."
            )

        self.save_dtype = get_gradient_dtype(self.model)
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

        self.per_doc_losses = torch.full(
            (len(self.data),),
            device=self.model.device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        self.gradients = defaultdict(list)
        self._gpu_gradients: dict[str, torch.Tensor] = {}

    def teardown(self) -> None:
        """No cleanup needed for in-memory storage."""
        assert isinstance(
            self.cfg, IndexConfig
        ), "cfg is required for GradientCollector"  # pleasing type checker
        if dist.is_initialized():
            dist.reduce(self.per_doc_losses, dst=0)

        grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
        if self.processor.preconditioners:
            process_preconditioners(
                self.processor,
                self.processor.preconditioners,
                len(self.data),
                grad_sizes,
                self.rank,
            )

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """Store activations for gradient computation."""
        p = self.processor.projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))
        normalizer = self.processor.normalizers.get(name)

        if isinstance(normalizer, AdamNormalizer):
            module._inputs = a
            return
        if isinstance(normalizer, AdafactorNormalizer):
            a_factor = normalizer.col.add(1e-30)
            a_factor = a_factor.rsqrt()
            a = a * a_factor.type_as(a)  # [N, S, I] * [I] → [N, S, I]

        if module._has_bias:
            # Append ones to activation for bias term
            ones = torch.ones(a.size(0), a.size(1), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)
        if p is not None:
            a_projection = self.projection(name, p, i, "right", a.device, a.dtype).T
            a = a @ a_projection  # type: ignore
        # set module._inputs to a
        module._inputs = a

    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]) -> None:
        """Compute and store per-sample gradient."""
        a = module._inputs  # [N, S, I/q]

        assert isinstance(a, torch.Tensor), "Activation cache missing for module"
        name = assert_type(str, module._name)
        p = self.processor.projection_dim
        i = getattr(module, LayerAdapter.in_attr(module))
        o = getattr(module, LayerAdapter.out_attr(module))
        normalizer = self.processor.normalizers.get(name)

        if isinstance(normalizer, AdamNormalizer):
            full_gradient = g.mT @ a  # [N, O, S] @ [N, S, I] → [N, O, I]
            P = normalizer.normalize_(full_gradient)
            if p is not None:
                g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
                a_projection = self.projection(name, p, i, "right", g.device, g.dtype).T
                P = g_projection @ P @ a_projection
        else:
            if isinstance(normalizer, AdafactorNormalizer):
                g_factor = normalizer.row.add(1e-30)
                g_factor = g_factor.mean().sqrt() * g_factor.rsqrt()
                g = g * g_factor.type_as(g)  # [N, S, O] * [O] → [N, S, O]

            if p is not None:
                g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
                g = g @ g_projection.T  # [N, S, p]

            P = g.mT @ a  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]

        P = P.flatten(1).clamp_(self.lo, self.hi)

        if not self.cfg.skip_preconditioners:
            P = P.float()
            if name in self.processor.preconditioners:
                self.processor.preconditioners[name].addmm_(P.mT, P)
            else:
                self.processor.preconditioners[name] = P.mT @ P

        # Store GPU tensor for scoring to avoid GPU→CPU→GPU round-trip
        if self.scorer is not None:
            self._gpu_gradients[name] = P

        # Store on CPU for later use (index building, etc.)
        self.gradients[name].append(P.cpu())

        del module._inputs  # type: ignore

    def process_batch(self, indices: list[int], **kwargs) -> None:
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        if self.scorer is not None:
            self.scorer(indices, self._gpu_gradients)
            self._gpu_gradients = {}

        self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)
