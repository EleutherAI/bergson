import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset, Value
from jaxtyping import Float
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.config import IndexConfig, ReduceConfig
from bergson.data import Builder
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    LayerAdapter,
)
from bergson.process_preconditioners import process_preconditioners
from bergson.score.scorer import Scorer
from bergson.utils.utils import assert_type, get_gradient_dtype


@dataclass(kw_only=True)
class GradientCollector(HookCollectorBase):
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

    mod_grads: dict = field(default_factory=dict)
    """Temporary storage for gradients during a batch, keyed by module name."""

    reduce_cfg: ReduceConfig | None = None
    """Configuration for in-run gradient reduction."""

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

        # Compute whether we need to save the index
        self.save_index = self.scorer is None and not self.cfg.skip_index

        if self.save_index:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            self.builder = Builder(
                self.cfg.partial_run_path,
                self.data,
                grad_sizes,
                self.save_dtype,
                self.reduce_cfg,
            )
        else:
            self.builder = None

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

        if self.save_index and self.reduce_cfg is None:
            # Asynchronously move the gradient to CPU and convert to the final dtype
            self.mod_grads[name] = P.to(
                device="cpu", dtype=self.save_dtype, non_blocking=True
            )
        else:
            self.mod_grads[name] = P.to(dtype=self.save_dtype)

        del module._inputs

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

        # Flush and reduce builder if it exists
        if self.builder is not None:
            self.builder.flush()
            self.builder.dist_reduce()

        if self.rank == 0:
            if self.reduce_cfg is not None:
                # Create a new dataset with one row for each reduced gradient
                assert self.builder is not None
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

    eps: float = 1e-6
    """Epsilon for numerical stability in preconditioning."""

    precondition: bool = False
    """Whether to apply preconditioning via autocorrelation Hessian approximation."""

    device: torch.device | str
    """Device to store collected gradients on."""

    dtype: torch.dtype
    """Dtype for stored gradients."""

    def setup(self) -> None:
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
        Compute per-sample gradient for the Attributor trace.

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

        # Precondition the gradient using Cholesky solve
        # TODO: Should damp here?
        if self.precondition:
            eigval, eigvec = self.processor.preconditioners_eigen[name]
            eigval_inverse_sqrt = 1.0 / (eigval + self.eps).sqrt()
            prec = eigvec * eigval_inverse_sqrt @ eigvec.mT
            P = P.type_as(prec) @ prec  # <- apply to P

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
        # TODO: handle more elegantly?
        self.save_dtype = (
            torch.float32 if self.model.dtype == torch.float32 else torch.float16
        )

        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

    def teardown(self) -> None:
        pass

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass

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
        Compute per-sample gradient for the hf callback.

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

        self.mod_grads[name] = P.to(
            device="cpu", dtype=self.save_dtype, non_blocking=True
        )

        del module._inputs
