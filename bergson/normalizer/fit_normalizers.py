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
    GradientProcessor,
    Normalizer,
)
from bergson.utils.utils import assert_type, get_gradient_dtype


@dataclass(kw_only=True)
class NormalizerCollector(HookCollectorBase):
    """
    Collects per-sample gradients from model layers and writes them to disk.

    - For each forward/backward hook, we compute the the gradient or a low-rank
    approximation via random projections, if cfg.projection_dim is set.
    - Supports normalization via Adam or Adafactor normalizers.
    """

    data: Dataset
    """The dataset being processed."""

    cfg: IndexConfig
    """Configuration for gradient index."""

    normalizers: dict[str, Normalizer] = field(default_factory=dict)

    def adafactor_update(
        self,
        name: str,
        weight_grad: torch.Tensor,
        bias_grad: torch.Tensor | None = None,
    ):
        # [N, O, I] -> [O, I]
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        sq = weight_grad.float().square_().sum(0)
        # row: mean over columns, shape [O]
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        # bias_avg_sq = E[bias_grad^2] where bias_grad = g.sum(dim=seq)
        bias_sq = None
        if bias_grad is not None:
            # [N, S, O] -> [N, O] -> [O]
            bias_sq = bias_grad.float().sum(dim=1).square().sum(0)

        if (normalizer := self.normalizers.get(name)) is None:
            # initialize accumulators at zero
            self.normalizers[name] = normalizer = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
                bias_avg_sq=torch.zeros_like(bias_sq) if bias_sq is not None else None,
            )
        else:
            assert isinstance(normalizer, AdafactorNormalizer)

        # in‐place accumulate
        normalizer.row.add_(row_acc)
        normalizer.col.add_(col_acc)
        if bias_sq is not None:
            assert normalizer.bias_avg_sq is not None
            normalizer.bias_avg_sq.add_(bias_sq)

    def adam_update(
        self,
        name: str,
        weight_grad: torch.Tensor,
        bias_grad: torch.Tensor | None = None,
    ):
        # [N, O, I] -> [O, I]
        sq = weight_grad.float().square_().sum(0)

        # bias_avg_sq = E[bias_grad^2] where bias_grad = g.sum(dim=seq)
        bias_sq = None
        if bias_grad is not None:
            bias_sq = bias_grad.float().sum(dim=1).square().sum(0)

        # initialize accumulators at zero
        if (normalizer := self.normalizers.get(name)) is None:
            self.normalizers[name] = normalizer = AdamNormalizer(
                torch.zeros_like(sq),
                bias_avg_sq=torch.zeros_like(bias_sq) if bias_sq is not None else None,
            )
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.weight_avg_sq.add_(sq)
        if bias_sq is not None:
            assert normalizer.bias_avg_sq is not None
            normalizer.bias_avg_sq.add_(bias_sq)

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
        self.save_dtype = get_gradient_dtype(self.model)
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """
        Cache activations for gradient computation.
        Bias second moments are computed directly from g in backward_hook,
        so we don't append ones here.
        """
        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """
        Compute per-sample gradient and accumulate second-moment statistics.

        Weight stats are computed from the outer product g.T @ a.
        Bias stats (when present) are computed directly from g.
        """
        a = module._inputs  # [N, S, I]

        assert isinstance(a, torch.Tensor), "Activation cache missing for module"
        name = assert_type(str, module._name)

        P = g.mT @ a  # [N, O, S] @ [N, S, I] → [N, O, I]

        self.callback(name, P, g if module._collect_bias else None)

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch."""

    def teardown(self):
        """Finalize normalizer collection: average across samples and ranks."""
        # n is the total number of samples across all ranks
        n = len(self.data)
        for normalizer in self.normalizers.values():
            if isinstance(normalizer, AdamNormalizer):
                normalizer.weight_avg_sq.div_(n)
                if dist.is_initialized():
                    dist.all_reduce(normalizer.weight_avg_sq, op=dist.ReduceOp.SUM)
                if normalizer.bias_avg_sq is not None:
                    normalizer.bias_avg_sq.div_(n)
                    if dist.is_initialized():
                        dist.all_reduce(normalizer.bias_avg_sq, op=dist.ReduceOp.SUM)
            elif isinstance(normalizer, AdafactorNormalizer):
                normalizer.row.div_(n)
                normalizer.col.div_(n)
                if dist.is_initialized():
                    dist.all_reduce(normalizer.row, op=dist.ReduceOp.SUM)
                    dist.all_reduce(normalizer.col, op=dist.ReduceOp.SUM)
                if normalizer.bias_avg_sq is not None:
                    normalizer.bias_avg_sq.div_(n)
                    if dist.is_initialized():
                        dist.all_reduce(normalizer.bias_avg_sq, op=dist.ReduceOp.SUM)

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
        processor=GradientProcessor(include_bias=cfg.include_bias),
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="Estimating normalizers")

    return collector.normalizers
