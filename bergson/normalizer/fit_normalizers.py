import math
import random
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedModel

from bergson.collector.collector import CollectorComputer, HookCollectorBase
from bergson.config import IndexConfig
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    LayerAdapter,
    Normalizer,
)
from bergson.process_preconditioners import process_preconditioners
from bergson.utils.utils import assert_type, get_gradient_dtype


@dataclass(kw_only=True)
class NormalizerCollector(HookCollectorBase):
    """
    Collects per-sample gradients from model layers and writes them to disk.

    - For each forward/backward hook, we compute the the gradient or a low-rank
    approximation via random projections, if cfg.projection_dim is set.
    - Supports also normalization via Adam or Adafactor normalizers.
    - Uses Builder for index construction and gradient saving.
    - Also supports Scorer for on-the-fly scoring of gradients.
    """

    data: Dataset
    """The dataset being processed."""

    cfg: IndexConfig
    """Configuration for gradient index."""

    normalizers: dict[str, Normalizer] = field(default_factory=dict)

    mod_grads: dict = field(default_factory=dict)
    """Temporary storage for gradients during a batch, keyed by module name."""

    def adafactor_update(self, name: str, g: torch.Tensor):
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.float().square_().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if (normalizer := self.normalizers.get(name)) is None:
            # initialize accumulators at zero
            self.normalizers[name] = normalizer = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )
        else:
            assert isinstance(normalizer, AdafactorNormalizer)

        # in‐place accumulate
        normalizer.row.add_(row_acc)
        normalizer.col.add_(col_acc)

    def adam_update(self, name: str, g: torch.Tensor):
        sq = g.square_().float().sum(0)

        # initialize accumulators at zero
        if (normalizer := self.normalizers.get(name)) is None:
            self.normalizers[name] = normalizer = AdamNormalizer(torch.zeros_like(sq))
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.avg_sq.add_(sq)

    def setup(self) -> None:
        """
        Initialize collector state.

        Sets up a Builder for gradient storage if not using a Scorer.
        """
        self.callback = (
            self.adafactor_update
            if self.cfg.normalizer == "adafactor"
            else self.adam_update
        )
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

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """
        Cache activations for gradient computation with normalizer preprocessing
        and compress via random projection if configured.
        Stores result in module._inputs for use in backward_hook.
        """
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

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """
        Compute per-sample gradient and store in mod_grads.

        Computes gradient as outer product g.T @ a (again with optional projection and
        normalization).
        """
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

        # self.mod_grads[name] = P.to(dtype=self.save_dtype)
        self.callback(name, P)

        del module._inputs

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch."""
        self.mod_grads.clear()

    def teardown(self):
        """
        Finalize normalizer collection.
        """
        grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
        if self.processor.preconditioners:
            process_preconditioners(
                self.processor,
                self.processor.preconditioners,
                len(self.data),
                grad_sizes,
                self.rank,
            )

        if self.rank == 0:
            self.processor.save(self.cfg.partial_run_path)


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset,
    cfg: IndexConfig,
    batches: list[list[int]],
    *,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    # Just to make the pbar more accurate
    rng = random.Random(0)
    rng.shuffle(batches)

    collector = NormalizerCollector(
        model=model.base_model,  # type: ignore
        data=data,
        cfg=cfg,
        target_modules=target_modules,
        filter_modules=cfg.filter_modules,
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="Estimating normalizers")

    normalizers = collector.normalizers

    # Divide by the number of documents processed and average across all ranks
    for normalizer in normalizers.values():
        if isinstance(normalizer, AdamNormalizer):
            normalizer.avg_sq.div_(len(data))

            if dist.is_initialized():
                dist.all_reduce(normalizer.avg_sq, op=dist.ReduceOp.AVG)

        elif isinstance(normalizer, AdafactorNormalizer):
            normalizer.row.div_(len(data))
            normalizer.col.div_(len(data))

            if dist.is_initialized():
                dist.all_reduce(normalizer.row, op=dist.ReduceOp.AVG)
                dist.all_reduce(normalizer.col, op=dist.ReduceOp.AVG)

    return normalizers
