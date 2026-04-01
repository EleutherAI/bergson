from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import torch
import torch.distributed as dist
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor, nn

from bergson.builder import Builder
from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig, PreprocessConfig
from bergson.process_preconditioners import (
    process_preconditioners,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import get_gradient_dtype, numpy_to_tensor


@dataclass(kw_only=True)
class InMemoryCollector(HookCollectorBase):
    """Collector that accumulates gradients in memory.

    Supports both per-example and per-token gradient collection
    via ``cfg.attribute_tokens``.  Uses in-memory builder
    (:class:`Builder`) for flat gradient storage and
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
        """Initialize collector state and create builder."""
        assert isinstance(
            self.model.device, torch.device
        ), "Model device is not set correctly"
        self.attribute_tokens = self.cfg.attribute_tokens
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
            self.builder = Builder(
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

    @HookCollectorBase.split_attention_heads
    def backward_hook(
        self,
        module: nn.Module,
        g: Float[Tensor, "N S O"],
    ) -> None:
        """Compute per-sample gradient, accumulate preconditioner, and store."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)

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
