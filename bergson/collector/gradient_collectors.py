import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, Value
from jaxtyping import Float
from torch import Tensor

from bergson.builder import Builder
from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig, PreprocessConfig
from bergson.process_preconditioners import process_preconditioners
from bergson.score.scorer import Scorer
from bergson.utils.utils import get_gradient_dtype


@dataclass(kw_only=True)
class GradientCollector(HookCollectorBase):
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

    mod_grads: dict = field(default_factory=dict)
    """Temporary storage for gradients during a batch, keyed by module name."""

    preprocess_cfg: PreprocessConfig = field(default_factory=PreprocessConfig)
    """Configuration for gradient preprocessing."""

    builder: Builder | None = None
    """Handles writing gradients to disk. Created in setup() if save_index is True."""

    scorer: Scorer | None = None
    """Optional scorer for computing scores instead of building an index."""

    def setup(self) -> None:
        """
        Initialize collector state.

        Sets up a Builder for gradient storage if not using a Scorer.
        """
        assert isinstance(
            self.model.device, torch.device
        ), "Model device is not set correctly"

        self.attribute_tokens = self.cfg.attribute_tokens

        if self.cfg.attribute_tokens:
            assert (
                self.preprocess_cfg.aggregation == "none"
            ), "attribute_tokens is incompatible with reduce mode."

        self.save_dtype = get_gradient_dtype(self.model)
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

        self.per_doc_losses = torch.full(
            (len(self.data),),
            device=self.model.device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        # Compute whether we need to save the index
        self.save_index = self.scorer is None and not self.cfg.skip_index

        if self.save_index:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            self.builder = Builder(
                self.data,
                grad_sizes,
                self.save_dtype,
                self.preprocess_cfg,
                attribute_tokens=self.cfg.attribute_tokens,
                path=self.cfg.partial_run_path,
            )
        else:
            self.builder = None

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """Compute per-sample gradient, accumulate preconditioner, and store."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)

        if not self.cfg.skip_preconditioners:
            P = P.float()
            if name in self.processor.preconditioners:
                self.processor.preconditioners[name].addmm_(P.mT, P)
            else:
                self.processor.preconditioners[name] = P.mT @ P

        if self.save_index and self.preprocess_cfg.aggregation == "none":
            # Asynchronously move the gradient to CPU and convert to the final
            # dtype
            self.mod_grads[name] = P.to(
                device="cpu", dtype=self.save_dtype, non_blocking=True
            )
        else:
            self.mod_grads[name] = P.to(dtype=self.save_dtype)

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch and update losses."""
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        if self.builder:
            self.builder(indices, self.mod_grads)
        if self.scorer:
            self.scorer(indices, self.mod_grads)
        self.mod_grads.clear()
        self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)

    def teardown(self):
        """
        Finalize gradient collection, save results and flush/reduce the Builder.
        """
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

        if self.builder:
            self.builder.teardown()

        if self.rank == 0:
            if self.preprocess_cfg.aggregation != "none":
                # Create a new dataset with one row for each reduced gradient
                assert self.builder
                self.data = Dataset.from_list(
                    [
                        {"query_index": i}
                        for i in range(self.builder.grad_buffer.shape[0])
                    ]
                )
            else:
                if self.cfg.drop_columns:
                    self.data = self.data.remove_columns(["input_ids"])

                self.data = self.data.add_column(
                    "loss",
                    self.per_doc_losses.cpu().numpy(),
                    feature=Value("float32"),
                    new_fingerprint="loss",
                )

            self.data.save_to_disk(str(self.cfg.partial_run_path / "data.hf"))

            self.processor.save(self.cfg.partial_run_path)


@dataclass(kw_only=True)
class TraceCollector(HookCollectorBase):
    """
    Collects gradient traces for influence function computation.

    Accumulates per-sample gradients across batches in memory (as lists per module).
    Optionally applies preconditioning using eigendecomposition of the gradient
    covariance. Designed for query-time gradient collection rather than index building.
    """

    mod_grads: dict = field(default_factory=lambda: defaultdict(list))
    """Accumulated grads per module. Maps module name to list of gradient tensors."""

    device: torch.device | str
    """Device to store collected gradients on."""

    dtype: torch.dtype
    """Dtype for stored gradients."""

    def setup(self) -> None:
        self.save_dtype = get_gradient_dtype(self.model)
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """Compute per-sample gradient, optionally precondition, and store."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)

        # Store the gradient for later use
        self.mod_grads[name].append(P.to(self.device, self.dtype, non_blocking=True))

    def process_batch(self, indices: list[int], **kwargs):
        return

    def teardown(self):
        return


@dataclass(kw_only=True)
class StreamingGradientCollector(HookCollectorBase):
    """
    Lightweight collector for streaming gradient collection during training.

    Stores per-sample gradients in `mod_grads` dict for external consumers
    (e.g., callbacks) to process. Used for callback in huggingface.py
    """

    mod_grads: dict = field(default_factory=dict)

    dtype: torch.dtype
    """Dtype for stored gradients."""

    def setup(self) -> None:
        self.save_dtype = get_gradient_dtype(self.model)

        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

    def teardown(self) -> None:
        pass

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        """Compute per-sample gradient and store on CPU."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)

        self.mod_grads[name] = P.to(
            device="cpu", dtype=self.save_dtype, non_blocking=True
        )
