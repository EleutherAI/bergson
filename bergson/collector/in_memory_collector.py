from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import torch
import torch.distributed as dist
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor, nn

from bergson.builders import Builder, create_builder
from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig, PreprocessConfig
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    LayerAdapter,
)
from bergson.process_preconditioners import (
    process_preconditioners,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import assert_type, get_gradient_dtype, numpy_to_tensor


@dataclass(kw_only=True)
class InMemoryCollector(HookCollectorBase):
    """Collector that accumulates gradients in memory.

    Supports both per-example and per-token gradient collection
    via ``cfg.attribute_tokens``.  Uses in-memory builders
    (:class:`InMemorySequenceBuilder` /
    :class:`InMemoryTokenBuilder`) for flat gradient storage and
    an optional :class:`Scorer` for on-the-fly scoring.

    After collection, ``self.gradients`` is populated from the
    builder's buffer in :meth:`teardown`, providing per-module
    gradient tensors for downstream use.
    """

    data: Dataset
    """The dataset being processed."""

    cfg: IndexConfig
    """Configuration for gradient collection."""

    mod_grads: dict = dc_field(default_factory=dict)
    """Temporary per-batch gradients keyed by module name."""

    preprocess_cfg: PreprocessConfig = dc_field(default_factory=PreprocessConfig)
    """Configuration for gradient preprocessing."""

    builder: Builder | None = None
    """Handles writing gradients. Created in setup()."""

    scorer: Scorer | None = None
    """Optional scorer for on-the-fly scoring."""

    gradients: dict[str, torch.Tensor] = dc_field(default_factory=dict)
    """Per-module gradients, populated from builder
    in teardown."""

    scores: list | torch.Tensor | None = None
    """Scores populated from scorer's writer in teardown."""

    def __post_init__(self):
        if self.filter_modules is None:
            self.filter_modules = self.cfg.filter_modules
        super().__post_init__()

    def setup(self) -> None:
        """Initialize collector state and create builders."""
        assert isinstance(
            self.model.device, torch.device
        ), "Model device is not set correctly"
        if self.cfg.include_bias and self.processor.normalizers is not None:
            raise NotImplementedError(
                "Bias with normalizers not supported yet, "
                "consider disabling bias inclusion."
            )

        if self.cfg.attribute_tokens:
            assert self.preprocess_cfg.aggregation == "none", (
                "attribute_tokens is incompatible" " with reduce mode."
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

        self.gradients = {}
        self.scores = None

        # Create in-memory builder when not scoring
        if self.builder is None and self.scorer is None:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            self.builder = create_builder(
                self.data,
                grad_sizes,
                self.save_dtype,
                self.preprocess_cfg,
                attribute_tokens=self.cfg.attribute_tokens,
            )

    def teardown(self) -> None:
        assert isinstance(self.cfg, IndexConfig)
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

        if self.builder is not None:
            self.builder.teardown()

            # Populate self.gradients from builder buffer
            buf = self.builder.grad_buffer
            offset = 0
            for name, dim in grad_sizes.items():
                self.gradients[name] = numpy_to_tensor(buf[:, offset : offset + dim])
                offset += dim

        if self.scorer is not None:
            self.scores = self.scorer.writer.scores

    def forward_hook(
        self,
        module: nn.Module,
        a: Float[Tensor, "N S I"],
    ) -> None:
        """Cache activations for gradient computation."""
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
            a = a * a_factor.type_as(a)

        if module._has_bias:
            ones = torch.ones(
                a.size(0),
                a.size(1),
                1,
                device=a.device,
                dtype=a.dtype,
            )
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)
        if p is not None:
            a_proj = self.projection(name, p, i, "right", a.device, a.dtype).T
            a = a @ a_proj
        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(
        self,
        module: nn.Module,
        g: Float[Tensor, "N S O"],
    ) -> None:
        """Compute per-sample gradient.

        When ``self.cfg.attribute_tokens`` is True, computes
        per-position gradients filtered to valid positions.
        """
        a = module._inputs  # [N, S, I/q]

        assert isinstance(a, torch.Tensor), "Activation cache missing for module"
        name = assert_type(str, module._name)
        p = self.processor.projection_dim
        i = getattr(module, LayerAdapter.in_attr(module))
        o = getattr(module, LayerAdapter.out_attr(module))
        normalizer = self.processor.normalizers.get(name)

        if isinstance(normalizer, AdamNormalizer):
            if self.cfg.attribute_tokens:
                # Per-position outer product: [N,S,O,1]*[N,S,1,I] → [N,S,O,I]
                P = g.unsqueeze(-1) * a.unsqueeze(-2)
                P = normalizer.normalize_(P)  # broadcasts [O,I] over [N,S,O,I]
                if p is not None:
                    g_proj = self.projection(name, p, o, "left", g.device, g.dtype)
                    a_proj = self.projection(name, p, i, "right", g.device, g.dtype).T
                    P = g_proj @ P @ a_proj  # [N, S, p, q]
                P = P.flatten(2)  # [N, S, grad_dim]
                P = P[self._current_valid_mask]  # [total_valid, grad_dim]
            else:
                full_gradient = g.mT @ a
                P = normalizer.normalize_(full_gradient)
                if p is not None:
                    g_proj = self.projection(name, p, o, "left", g.device, g.dtype)
                    a_proj = self.projection(name, p, i, "right", g.device, g.dtype).T
                    P = g_proj @ P @ a_proj
        else:
            if isinstance(normalizer, AdafactorNormalizer):
                g_factor = normalizer.row.add(1e-30)
                g_factor = g_factor.mean().sqrt() * g_factor.rsqrt()
                g = g * g_factor.type_as(g)

            if p is not None:
                g_proj = self.projection(name, p, o, "left", g.device, g.dtype)
                g = g @ g_proj.T

            if self.cfg.attribute_tokens:
                # Per-position outer product
                # [N,S,O/p,1]*[N,S,1,I/q] -> [N,S,O/p,I/q]
                P = g.unsqueeze(-1) * a.unsqueeze(-2)
                P = P.flatten(2)

                # Filter to valid positions
                P = P[self._current_valid_mask]
            else:
                P = g.mT @ a

        P = P.flatten(1).clamp_(self.lo, self.hi)

        if not self.cfg.skip_preconditioners:
            P = P.float()
            if name in self.processor.preconditioners:
                self.processor.preconditioners[name].addmm_(P.mT, P)
            else:
                self.processor.preconditioners[name] = P.mT @ P

        # GPU for scorer/reduce, CPU for builder
        if self.scorer is not None or self.preprocess_cfg.aggregation != "none":
            self.mod_grads[name] = P.to(dtype=self.save_dtype)
        else:
            self.mod_grads[name] = P.to(
                device="cpu",
                dtype=self.save_dtype,
                non_blocking=True,
            )

    def process_batch(self, indices: list[int], **kwargs) -> None:
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        if self.builder is not None:
            self.builder(indices, self.mod_grads)
        if self.scorer is not None:
            self.scorer(indices, self.mod_grads)

        self.mod_grads.clear()

        self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)
