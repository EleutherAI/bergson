import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, Value
from jaxtyping import Float
from torch import Tensor

from bergson.builder import Builder
from bergson.collector.collector import HookCollectorBase
from bergson.dpo import combine_pair_gradients
from bergson.config import IndexConfig, PreprocessConfig
from bergson.process_autocorrelation import process_autocorrelation_matrices
from bergson.score.scorer import Scorer
from bergson.utils.projection import make_global_projector
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

    global_projector: Any = None
    """Lazily-built TRAK projector used when
    ``processor.projection_target == 'global'``."""

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

        self.is_dpo = self.cfg.loss_fn == "dpo"
        self.num_pairs = len(self.data) // 2 if self.is_dpo else len(self.data)
        self.per_doc_losses = torch.full(
            (self.num_pairs,),
            device=self.model.device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        # Compute whether we need to save the index
        self.save_index = self.scorer is None and not self.cfg.skip_index

        if self.save_index:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            builder_data = self.data.select(range(self.num_pairs)) if self.is_dpo else self.data
            self.builder = Builder(
                builder_data,
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
        """Compute per-sample gradient, accumulate autocorrelation matrix, and store."""
        name: str = module._name  # type: ignore[assignment]
        P = self._compute_gradient(module, g)

        global_proj = self.processor.projection_target == "global"

        # Collect per-module hessians when projection target is per_module
        if not self.cfg.skip_hessians and not global_proj and not self.is_dpo:
            P = P.float()
            if name in self.processor.hessians:
                self.processor.hessians[name].addmm_(P.mT, P)
            else:
                self.processor.hessians[name] = P.mT @ P

        if self.cfg.offload_gradients:
            if global_proj:
                raise ValueError("CPU gradient offload requires per-module projection")
            self.mod_grads[name] = P.to(device="cpu", dtype=self.save_dtype)
        elif global_proj:
            # Keep on-device until the global projection runs in process_batch.
            self.mod_grads[name] = P
        elif self.save_index and self.preprocess_cfg.aggregation == "none":
            # Asynchronously move the gradient to CPU and convert to the final
            # dtype
            self.mod_grads[name] = P.to(
                device="cpu", dtype=self.save_dtype, non_blocking=True
            )
        else:
            self.mod_grads[name] = P.to(dtype=self.save_dtype)

    def global_project(self) -> None:
        """Concatenate per-module per-example gradients and project.
        Sets ``self.mod_grads`` to ``{"gradients": projected}``.

        Projects in row-chunks. A naive ``flat = torch.cat(..., dim=1)`` of
        all module gradients can need tens of GiB contiguous on rank 0 when
        the bin-packer assigns many short examples to a single batch (e.g.
        flan_v2 with token_batch_size=2048 packs ~80 rows of ~525 MB each).
        The projector is per-row, so chunking is exact; chunk size is sized
        to a fixed GPU-byte budget.
        """
        # backward_hook fires in reverse forward order, so insertion order in
        # mod_grads is deterministic for a given model.
        parts = list(self.mod_grads.values())
        n_rows = parts[0].shape[0]
        total_grad_dim: int = sum(int(math.prod(P.shape[1:])) for P in parts)

        if self.global_projector is None:
            assert self.processor.projection_dim is not None
            self.global_projector = make_global_projector(
                grad_dim=total_grad_dim,
                proj_dim=self.processor.projection_dim,
                device=parts[0].device,
                dtype=parts[0].dtype,
                projection_type=self.processor.projection_type,
            )

        # Cap chunk_flat at ~4 GiB per chunk to leave headroom for fast_jl's
        # internal scratch buffers alongside the per-module tensors that stay
        # alive until the loop exits.
        bytes_per_row = total_grad_dim * parts[0].element_size()
        chunk_rows = max(1, (4 * 1024**3) // max(bytes_per_row, 1))
        chunk_rows = min(chunk_rows, n_rows)

        chunks_cpu: list[torch.Tensor] = []
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk_flat = torch.cat(
                [P[start:end].reshape(end - start, -1) for P in parts], dim=1
            )
            chunk_projected = self.global_projector.project(chunk_flat, model_id=0)
            chunks_cpu.append(
                chunk_projected.to(
                    device="cpu", dtype=self.save_dtype, non_blocking=True
                )
            )
            del chunk_flat, chunk_projected

        # Free per-module GPU tensors now that all chunks are projected.
        self.mod_grads = {}

        self.mod_grads = {"gradients": torch.cat(chunks_cpu, dim=0)}

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch and update losses."""
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        if self.processor.projection_target == "global":
            self.global_project()

        if self.is_dpo:
            n = len(self.mod_grads[next(iter(self.mod_grads))])
            if n % 2 or len(losses) != n // 2:
                raise ValueError("DPO batch gradients/losses must contain paired rows")
            half = n // 2
            self.mod_grads = {
                name: combine_pair_gradients(grad)
                for name, grad in self.mod_grads.items()
            }
            if not self.cfg.skip_hessians and self.processor.projection_target != "global":
                for name, grad in self.mod_grads.items():
                    p = grad.float()
                    if name in self.processor.hessians:
                        self.processor.hessians[name].addmm_(p.mT, p)
                    else:
                        self.processor.hessians[name] = p.mT @ p
            pair_indices = list(indices[:half])
            # Pair IDs are stable even when a caller reorders a batch.
            if "pair_index" in kwargs:
                pair_indices = list(kwargs["pair_index"][:half])
            indices = pair_indices

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
        if self.processor.hessians:
            process_autocorrelation_matrices(
                self.processor,
                self.processor.hessians,
                self.num_pairs,
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
                if self.is_dpo:
                    self.data = self.data.select(range(self.num_pairs))
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
