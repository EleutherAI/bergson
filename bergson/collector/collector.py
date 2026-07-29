import functools
import hashlib
import math
import os
import warnings
from abc import ABC, abstractmethod
from contextlib import ContextDecorator, nullcontext
from dataclasses import astuple, dataclass, field
from fnmatch import fnmatchcase
from typing import Callable, Literal, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from jaxtyping import Float
from peft import PeftModel
from torch import Tensor
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.hooks import RemovableHandle
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.config import AttentionConfig, HessianConfig, IndexConfig
from bergson.data import compute_num_token_grads, pad_and_tensor
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    LayerAdapter,
)
from bergson.moe import (
    ExpertLinear,
    expand_moe,
    is_fused_experts,
    restore_moe,
    router_batch_size,
    tracked_router,
)
from bergson.utils.logger import get_logger
from bergson.utils.peft import set_peft_enabled
from bergson.utils.utils import assert_type


@dataclass
class HookCollectorBase(ContextDecorator, ABC):
    """
    Abstract base class for collectors that attach forward and backward hooks to model
    layers.

    Automatically discovers supported modules in the model, registers hooks during
    context entry, and provides lifecycle methods (setup/teardown) for subclasses to
    implement custom logic.

    Assumes module activation shape is [N, S, I] and activation gradient shape [N,S,O]
    where N=batch size, S=sequence length, I=input dimension, O=output dimension.

    Subclasses must implement:
        - setup(): Initialize state (buffers, dicts, etc.)
        - teardown(): Clean up and save results
        - forward_hook(): Process activations during forward pass
        - backward_hook(): Process gradients during backward pass
    """

    model: nn.Module
    """ The model to attach forward and backward hooks to. """

    filter_modules: str | None = None
    """If provided, a glob pattern to filter out modules from gradient collection.
    For example, "transformer.h.*.mlp.*" will exclude all MLP layers in a
    standard transformer architecture."""

    target_modules: set[str] | None = None
    """
    Set of module names to attach hooks to. Should consist only of supported modules
    (see LayerAdapter.supported_modules). If None, hooks are attached to all supported
    layers in the model.
    """

    processor: GradientProcessor = field(default_factory=GradientProcessor)
    """Configuration for processing and compressing gradients."""

    attention_cfgs: dict[str, AttentionConfig] = field(default_factory=dict)
    """
    Optional configuration specifying how to split up the attention module gradients
    into per-head gradients. See also bergson.config.AttentionConfig.
    """

    attribute_tokens: bool = False
    """When True, compute per-position gradients instead of per-example."""

    lo: float = float("-inf")
    """Lower clamp bound for gradients. May be narrowed in subclass ``setup()``."""

    hi: float = float("inf")
    """Upper clamp bound for gradients. May be narrowed in subclass ``setup()``."""

    mod_grads: dict = field(default_factory=dict)
    """Gradients collected for the current batch, keyed by module name."""

    save_dtype: torch.dtype = torch.float32
    """Dtype gradients are cast to on the way out. Set in subclass ``setup()``."""

    track_moe_experts: bool = False
    """Expose fused-parameter MoE experts to collection, via
    :func:`bergson.moe.expand_moe`. Off by default: see ``IndexConfig``."""

    logger = get_logger("HookCollectorBase", level="INFO")

    def __post_init__(
        self,
    ):
        """Init, discover target modules, and call setup()."""
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # Before discovery: expansion is what makes fused experts trackable at
        # all. Restore on the way down too, so the flag wins over an expansion
        # something else already did.
        if self.track_moe_experts:
            expand_moe(self.model)
        else:
            restore_moe(self.model)
            self._warn_if_moe_untracked()

        # Discover target modules using the static method
        self.target_info = self.discover_targets(
            self.model,
            self.target_modules,
            self.processor.include_bias,
            self.filter_modules,
        )

        # Validate that attention_cfgs keys match actual module names
        if unknown := set(self.attention_cfgs) - set(self.target_info):
            raise ValueError(
                f"attention_cfgs contains module names not found in the model: "
                f"{unknown}. Available modules: {set(self.target_info)}"
            )

        # Allow subclasses to perform custom initialization
        self.setup()

    def _warn_if_moe_untracked(self) -> None:
        """Skipping fused experts used to be silent; say so instead."""
        skipped = sum(1 for m in self.model.modules() if is_fused_experts(m))
        if skipped:
            warnings.warn(
                f"{skipped} fused MoE expert modules are not being attributed, "
                f"leaving only attention and lm_head tracked. Pass "
                f"--track_moe_experts to include them, at a cost in collection "
                f"speed; see the README.",
                stacklevel=2,
            )

    @staticmethod
    def discover_targets(
        model: nn.Module,
        target_modules: set[str] | None = None,
        include_bias: bool = False,
        filter_modules: str | None = None,
    ) -> dict[str, tuple[torch.device, torch.Size, bool]]:
        """
        Discover target modules without instantiating a collector.

        This is useful when you need target_info early (e.g., to allocate buffers)
        before creating the actual collector instance.

        Args:
            model: The model to scan for supported layers, see
            LayerAdapter.supported_modules.
            target_modules: Optional set of module names to filter. If None, all
            supported layers are included.
            include_bias: Whether to track bias parameters for modules that have them.
            filter_modules: Optional glob pattern to exclude modules by name
            (e.g., "*.lm_head").

        Returns:
            Dictionary mapping module names to
                (device, weight_shape, collect_bias) tuples.
        """
        target_info = {}
        for name, layer in model.named_modules():
            if not LayerAdapter.is_supported(layer):
                continue

            if target_modules is not None and name not in target_modules:
                continue

            if filter_modules and any(
                fnmatchcase(name, pat.strip()) for pat in filter_modules.split(",")
            ):
                continue

            collect_bias = getattr(layer, "bias", None) is not None and include_bias

            o = getattr(layer, LayerAdapter.out_attr(layer))
            i = getattr(layer, LayerAdapter.in_attr(layer))
            weight_shape = torch.Size([o, i])

            target_info[name] = (
                layer.weight.device,
                weight_shape,
                collect_bias,
            )
        return target_info

    @staticmethod
    def get_head_name(name: str, head_idx: int) -> str:
        """Get the name of an attention head with index `head_idx` in a
        module with name `name`."""
        return f"{name}.head_{head_idx}"

    @staticmethod
    def split_attention_heads(fn):
        """Decorator that splits attention module calls into per-head calls."""

        @functools.wraps(fn)
        def wrapper(self, module, g):
            name = module._name

            if name not in self.attention_cfgs:
                return fn(self, module, g)

            num_heads, head_size, head_dim = astuple(self.attention_cfgs[name])

            # Save state
            orig_name = module._name
            orig_out = getattr(module, LayerAdapter.out_attr(module))

            assert num_heads * head_size == orig_out, (
                f"{name}: num_heads * head_size = {num_heads * head_size} does "
                f"not cover the module's {orig_out} output features"
            )

            setattr(module, LayerAdapter.out_attr(module), head_size)

            for h in range(num_heads):
                module._name = self.get_head_name(name, h)
                try:
                    head_g = torch.narrow(g, head_dim, h * head_size, head_size)
                except Exception as e:
                    print(
                        f"Error processing gradient of shape {g.shape} for head {h}"
                        f" in module {name}. Provided head config may be incorrect. "
                        f"Head config: head dim {head_dim}, head size {head_size},"
                        f" num heads {num_heads}."
                    )
                    raise e
                fn(self, module, head_g)

            # Restore
            module._name = orig_name
            setattr(module, LayerAdapter.out_attr(module), orig_out)

        return wrapper

    @staticmethod
    def split_head_name(name: str) -> tuple[str, int] | None:
        """Split ``"<module>.head_<i>"`` into ``("<module>", i)``, else None."""
        base, sep, suffix = name.rpartition(".head_")
        if not sep or not suffix.isdigit():
            return None
        return base, int(suffix)

    def normalizer_for(self, name: str):
        """The normalizer for module `name`.

        Normalizers are keyed by the unsplit module name, so a split attention
        head looks up its parent and takes the slice of the output dimension it
        covers.
        """
        normalizer = self.processor.normalizers.get(name)
        if normalizer is not None:
            return normalizer

        split = self.split_head_name(name)
        if split is None:
            return None

        base, head_idx = split
        normalizer = self.processor.normalizers.get(base)
        if normalizer is None or base not in self.attention_cfgs:
            return normalizer

        head_size = self.attention_cfgs[base].head_size
        lo, hi = head_idx * head_size, (head_idx + 1) * head_size

        if isinstance(normalizer, AdamNormalizer):
            bias = normalizer.bias_avg_sq
            return AdamNormalizer(
                normalizer.weight_avg_sq[lo:hi],
                bias[lo:hi] if bias is not None else None,
            )
        if isinstance(normalizer, AdafactorNormalizer):
            bias = normalizer.bias_avg_sq
            return AdafactorNormalizer(
                normalizer.row[lo:hi],
                normalizer.col,
                bias[lo:hi] if bias is not None else None,
            )
        return normalizer

    @property
    def per_module_projection_dim(self) -> int | None:
        """Per-module projection dimension, or None if projection is disabled."""
        if self.processor.projection_target == "global":
            return None
        return self.processor.projection_dim or None

    def shapes(self) -> Mapping[str, torch.Size]:
        """Return the shapes of the gradients collected by this collector."""
        if self.processor.projection_target == "global":
            assert self.processor.projection_dim is not None
            return {"gradients": torch.Size((self.processor.projection_dim,))}

        proj_shape = (
            torch.Size((p_dim, p_dim))
            if (p_dim := self.per_module_projection_dim) is not None
            else None
        )

        shapes = {}
        for name, (_, target_shape, collect_bias) in self.target_info.items():
            if name in self.attention_cfgs:
                attention_cfg = self.attention_cfgs[name]
                if proj_shape:
                    head_shape = proj_shape
                else:
                    # Mutate the attention module's shape to get the attention
                    # head shape
                    attention_shape = list(target_shape)
                    # - 2 because we're excluding the batch and sequence activation
                    # dimensions
                    attention_shape[attention_cfg.head_dim - 2] = (
                        attention_cfg.head_size
                    )
                    if collect_bias:
                        attention_shape[-1] += 1
                    head_shape = torch.Size(attention_shape)

                shapes.update(
                    {
                        self.get_head_name(name, h): head_shape
                        for h in range(attention_cfg.num_heads)
                    }
                )
            else:
                if proj_shape:
                    shapes[name] = proj_shape
                else:
                    grad_shape = list(target_shape)
                    if collect_bias:
                        grad_shape[-1] += 1
                    shapes[name] = torch.Size(grad_shape)

        return shapes

    def projection(
        self,
        name: str,
        m: int,
        n: int,
        role: Literal["left", "right", "single"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the `role` projection matrix for parameter `name` of shape [m, n]."""
        key = (name, role, device)
        if key in self.processor._projection_matrices:
            return self.processor._projection_matrices[key]

        identifier = f"{name}/{role}"

        A = create_projection_matrix(
            identifier,
            m,
            n,
            dtype,
            device,
            self.processor.projection_type,
            self.processor.projection_scale,
        )
        self.processor._projection_matrices[key] = A
        return A

    def accumulate_global_projection(self, name: str, P: Tensor) -> bool:
        """Project ``P`` with module ``name``'s block of the global projection
        matrix and accumulate it into the single ``"gradients"`` key of
        ``self.mod_grads``.

        Returns ``False`` and does nothing when
        ``projection_target != "global"``, for use as a ``backward_hook``
        early-out.
        """
        if self.processor.projection_target != "global":
            return False

        assert self.processor.projection_dim is not None
        R = self.projection(
            name,
            self.processor.projection_dim,
            P.shape[1],
            "single",
            P.device,
            P.dtype,
        )
        projected = P @ R.T  # [N, proj_dim]
        if "gradients" in self.mod_grads:
            self.mod_grads["gradients"].add_(projected)
        else:
            self.mod_grads["gradients"] = projected
        return True

    def finalize_global_projection(self) -> None:
        """Move the accumulated global gradient to CPU / the save dtype.

        Called from ``process_batch``; accumulation stays on the compute
        device across the backward pass.
        """
        if self.processor.projection_target != "global":
            return
        self.mod_grads["gradients"] = self.mod_grads["gradients"].to(
            device="cpu", dtype=self.save_dtype, non_blocking=True
        )

    def num_rows(self, data: Dataset) -> int:
        """Number of gradient rows accumulated into an autocorrelation Gram
        over ``data``: one per gradient-carrying token when
        ``attribute_tokens`` is set, else one per document. Used to normalize
        the Gram into a second moment over the matching unit."""
        if self.attribute_tokens:
            return int(compute_num_token_grads(data).sum())
        return len(data)

    def with_batch(self, collection_mask: Tensor | None = None) -> "HookCollectorBase":
        """
        Set the current collection mask before entering the context.

        This allows hooks to access the mask during forward/backward passes.
        Usage:
            with collector.with_batch(collection_mask):
                # forward/backward pass
                # hooks can access self._current_collection_mask

        Args:
            collection_mask: Optional boolean tensor of shape [batch_size, seq_len],
                aligned with input positions, marking which positions the hooks
                collect. This normally spans every real (non-padding) position
                but each sequence's last.

        Returns:
            self, for use as a context manager.
        """
        self._current_collection_mask = collection_mask
        return self

    def collection_mask(self, module: nn.Module) -> Tensor | None:
        """The mask selecting ``module``'s gradient-carrying positions. A fused
        MoE expert sees its own ``[N, L]`` grid of routed tokens, not the batch's
        ``[N, S]`` positions."""
        row_mask = getattr(module, "_row_mask", None)
        return self._current_collection_mask if row_mask is None else row_mask

    def __enter__(self):
        """Register forward and backward hooks on all target modules."""
        if self.attribute_tokens and any(
            isinstance(self.model.get_submodule(n), ExpertLinear)
            for n in self.target_info
        ):
            raise ValueError(
                "attribute_tokens is incompatible with fused MoE experts: under "
                "top-k routing one token feeds several experts, so its gradient "
                "rows cannot line up with token positions."
            )

        for name in self.target_info:
            layer = self.model.get_submodule(name)

            # Store module name for use in hook callbacks
            layer._name = name  # type: ignore[attr-defined]
            layer._collect_bias = self.target_info[name][2]  # type: ignore[attr-defined]

            # Register hooks
            if tracked_router(layer) is not None:
                self._fwd_hooks.append(layer.register_forward_hook(self._tap_router))
            else:
                self._fwd_hooks.append(layer.register_forward_hook(self._process_input))
                self._bwd_hooks.append(
                    layer.register_full_backward_hook(self._process_grad)
                )

        return self

    def _process_input(self, module: nn.Module, inp: tuple, _):
        """Internal forward hook that extracts input and delegates to subclass."""
        x = self._unflatten(module, inp[0].detach())
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        self.forward_hook(module, x)

    def _unflatten(self, module: nn.Module, x: Tensor) -> Tensor:
        """A router sees hidden states already flattened to [N*S, D]."""
        if x.ndim == 2 and tracked_router(module) is not None:
            return x.view(router_batch_size(module), -1, x.shape[-1])
        return x

    def _tap_router(self, module: nn.Module, inp: tuple, out):
        """A router scores its logits inside its own forward, so grad_output on a
        module backward hook is None. Hook the logits tensor instead."""
        self._process_input(module, inp, out)
        o = getattr(module, LayerAdapter.out_attr(module))
        logits = next(
            t
            for t in (out if isinstance(out, tuple) else (out,))
            if isinstance(t, Tensor) and t.requires_grad and t.shape[-1] == o
        )
        logits.register_hook(lambda g: self._dispatch_grad(module, g))

    def _process_grad(self, module: nn.Module, _, grad_out):
        """Internal backward hook that extracts gradient and delegates to subclass."""
        # Sanity checks
        assert LayerAdapter.is_supported(module), (
            f"Expected a module of type {LayerAdapter.supported_modules}, "
            f"got {type(module)}"
        )

        self._dispatch_grad(module, grad_out[0])

    def _dispatch_grad(self, module: nn.Module, grad: Tensor):
        """Hand an output gradient to the subclass as [N, S, O]."""
        g = self._unflatten(module, grad.detach())

        a = getattr(module, "_inputs", None)
        if isinstance(a, Tensor) and a.dtype != g.dtype:
            g = g.to(a.dtype)  # deepseek and glm4 routers upcast before F.linear

        self.backward_hook(module, g)
        if hasattr(module, "_inputs"):
            del module._inputs

    def __exit__(self, exc_type, exc, tb):
        """Clean up hooks and allow subclass cleanup."""
        # Clean up temporary attributes
        for layer in self.model.modules():
            for attr in ("_inputs", "_name", "_row_mask"):
                if hasattr(layer, attr):
                    delattr(layer, attr)

        # Remove all registered hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()
        self._fwd_hooks.clear()
        self._bwd_hooks.clear()

        return False

    @abstractmethod
    def setup(self) -> None:
        """
        Called at the end of __post_init__.

        Override to perform custom initialization such as:
        - Allocating buffers or dictionaries
        - Loading pretrained weights or data
        - Initializing accumulators
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Called at the end.

        Override to perform custom cleanup such as:
        - Saving results to disk
        - Flushing buffers
        - Computing final statistics
        - Freeing resources
        """
        pass

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """
        Cache activations for gradient computation with normalizer preprocessing
        and compress via random projection if configured.
        Stores result in module._inputs for use in backward_hook.
        """
        p = self.per_module_projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))
        normalizer = self.normalizer_for(name)

        if isinstance(normalizer, AdamNormalizer):
            module._inputs = a
            return
        if isinstance(normalizer, AdafactorNormalizer):
            a_factor = normalizer.col.add(1e-30)
            a_factor = a_factor.rsqrt()
            a = a * a_factor.type_as(a)  # [N, S, I] * [I] → [N, S, I]

        # Defer a-projection when bias is included — backward needs full a to
        # compute the outer product before concatenating the bias column.
        if p is not None and not module._collect_bias:
            a_projection = self.projection(name, p, i, "right", a.device, a.dtype).T
            a = a @ a_projection  # [N, S, I(+1)] @ [I(+1), p] → [N, S, p]

        module._inputs = a

    def double_sided_projection(
        self, name: str, P: Tensor, g: Tensor, p: int, o: int, i: int
    ):
        g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
        a_projection = self.projection(name, p, i, "right", g.device, g.dtype).T
        return g_projection @ P @ a_projection

    def double_sided_projection_with_bias(
        self,
        name: str,
        g: Tensor,
        a: Tensor,
        bias_grad: Tensor,
        p: int,
        o: int,
        i: int,
    ) -> Tensor:
        """Double-sided projection of the gradient with the bias column appended,
        without materializing the full [O, I+1] gradient with an outer product.

        Equivalent to forming ``cat([g ⊗ a, bias_grad], -1)`` and calling
        ``double_sided_projection`` with ``i + 1``, because

            ``let L = left side projection matrix``
            ``let R = right side projection matrix``
            ``L @ cat([gᵀa, b], -1) @ Rᵀ = (g @ Lᵀ)ᵀ @ (a @ Rᵀ[:i]) + (b @ Lᵀ) ⊗ Rᵀ[i]``

        Only valid when nothing elementwise (e.g. Adam normalization) is applied
        to the outer product.
        """
        g_projection = self.projection(name, p, o, "left", g.device, g.dtype)
        a_projection = self.projection(name, p, i + 1, "right", g.device, g.dtype).T

        g = g @ g_projection.T  # [N, S, p]
        a = a @ a_projection[:i]  # [N, S, p]
        bias_grad = bias_grad @ g_projection.T  # [N, p] or [N, S, p]

        if self.attribute_tokens:
            # [N, S, p, 1] * [N, S, 1, p] → [N, S, p, p]
            P = g.unsqueeze(-1) * a.unsqueeze(-2)
        else:
            P = g.mT @ a  # [N, p, S] @ [N, S, p] → [N, p, p]

        # Outer product of the projected bias gradient with the bias row of the
        # right projection: [..., p, 1] * [p] → [..., p, p]
        return P + bias_grad.unsqueeze(-1) * a_projection[i]

    def _compute_gradient(self, module: nn.Module, g: Float[Tensor, "N S O"]) -> Tensor:
        """Compute the per-sample (or per-token) module gradient from cached activations
        and the output gradient.

        Handles normalizer preprocessing, bias appending, double-sided random
        projection, and ``attribute_tokens`` per-position paths.  Does not handle
        global (all modules) random projection. Returns the flattened, clamped module
        gradient tensor ``P``.
        """
        a = module._inputs  # [N, S, I/q]
        assert isinstance(a, torch.Tensor), "Activation cache missing for module"
        name = assert_type(str, module._name)
        p = self.per_module_projection_dim
        i = getattr(module, LayerAdapter.in_attr(module))
        o = getattr(module, LayerAdapter.out_attr(module))
        normalizer = self.normalizer_for(name)

        if isinstance(normalizer, AdamNormalizer):
            # Gate on the per-module flag, as shapes(), discover_targets() and
            # the forward hook do: in a mixed-bias model (e.g. Qwen2) the
            # biasless modules have no bias_avg_sq.
            if module._collect_bias:
                if self.attribute_tokens:
                    bias_grad = normalizer.normalize_bias(g)  # [N, S, O]
                else:
                    bias_grad = normalizer.normalize_bias(g).sum(dim=1)  # [N, O]
            else:
                bias_grad = None

            if self.attribute_tokens:
                # Per-position outer product: [N,S,O,1]*[N,S,1,I] → [N,S,O,I]
                P = g.unsqueeze(-1) * a.unsqueeze(-2)

                P = normalizer.normalize_weight(P)  # broadcasts [O,I] over [N,S,O,I]
                if bias_grad is not None:
                    P = torch.cat([P, bias_grad.unsqueeze(-1)], dim=-1)
                    i += 1

                if p is not None:
                    P = self.double_sided_projection(name, P, g, p, o, i)

                P = P.flatten(2)  # [N, S, grad_dim]
                P = P[self.collection_mask(module)]  # [total_valid, grad_dim]
            else:
                P = g.mT @ a  # [N,O,S] @ [N,S,I] → [N,O,I]

                P = normalizer.normalize_weight(P)  # broadcasts [O,I] over [N,O,I]
                if bias_grad is not None:
                    P = torch.cat([P, bias_grad.unsqueeze(2)], dim=2)  # [N,O,I+1]
                    i += 1

                if p is not None:
                    P = self.double_sided_projection(name, P, g, p, o, i)

        elif isinstance(normalizer, AdafactorNormalizer):
            if module._collect_bias:
                if self.attribute_tokens:
                    bias_grad = normalizer.normalize_bias(g)  # [N, S, O]
                else:
                    bias_grad = normalizer.normalize_bias(g).sum(dim=1)  # [N, O]
            else:
                bias_grad = None

            # Apply row normalization to g (for weights)
            g_factor = normalizer.row.add(1e-30)
            g_factor = g_factor.mean().sqrt() * g_factor.rsqrt()
            g = g * g_factor.type_as(g)  # [N, S, O] * [O] → [N, S, O]

            if self.attribute_tokens:
                if bias_grad is not None and p is not None:
                    # a was not projected in forward; project both factors and
                    # add the projected bias column without forming [N,S,O,I+1]
                    P = self.double_sided_projection_with_bias(
                        name, g, a, bias_grad, p, o, i
                    )
                elif bias_grad is not None:
                    # a was not projected in forward
                    # [N, S, O, 1] * [N, S, 1, I] → [N, S, O, I]
                    P = g.unsqueeze(-1) * a.unsqueeze(-2)
                    # [N, S, O, I+1]
                    P = torch.cat([P, bias_grad.unsqueeze(-1)], dim=-1)
                else:
                    # a was already projected in forward; project g individually
                    if p is not None:
                        g_projection = self.projection(
                            name, p, o, "left", g.device, g.dtype
                        )
                        g = g @ g_projection.T
                    # [N, S, O/p, 1] * [N, S, 1, I/q] → [N, S, O/p, I/q]
                    P = g.unsqueeze(-1) * a.unsqueeze(-2)
                P = P.flatten(2)  # [N, S, grad_dim]
                P = P[self.collection_mask(module)]  # [total_valid, grad_dim]
            else:
                if bias_grad is not None and p is not None:
                    P = self.double_sided_projection_with_bias(
                        name, g, a, bias_grad, p, o, i
                    )
                elif bias_grad is not None:
                    P = g.mT @ a  # [N, O, I]
                    P = torch.cat([P, bias_grad.unsqueeze(2)], dim=2)  # [N, O, I+1]
                else:
                    # a was already projected in forward; project g individually
                    if p is not None:
                        g_projection = self.projection(
                            name, p, o, "left", g.device, g.dtype
                        )
                        g = g @ g_projection.T  # [N, S, p]

                    P = g.mT @ a  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]
        else:
            # No normalizer
            if module._collect_bias:
                if self.attribute_tokens:
                    bias_grad = g  # [N, S, O]
                else:
                    bias_grad = g.sum(dim=1)  # [N, O]
            else:
                bias_grad = None

            if p is not None and module._collect_bias:
                # a was not projected in forward; project both factors and add
                # the projected bias column without forming [N, (S,) O, I+1]
                assert bias_grad is not None
                P = self.double_sided_projection_with_bias(
                    name, g, a, bias_grad, p, o, i
                )
                if self.attribute_tokens:
                    P = P.flatten(2)  # [N, S, grad_dim]
                    P = P[self.collection_mask(module)]  # [total_valid, grad_dim]
            else:
                # a was already projected in forward if p is set;
                # project g individually
                if p is not None:
                    g_projection = self.projection(
                        name, p, o, "left", g.device, g.dtype
                    )
                    g = g @ g_projection.T  # [N, S, p]

                if self.attribute_tokens:
                    # [N, S, O/p, 1] * [N, S, 1, I/q] → [N, S, O/p, I/q]
                    P = g.unsqueeze(-1) * a.unsqueeze(-2)
                    if bias_grad is not None:
                        P = torch.cat([P, bias_grad.unsqueeze(-1)], dim=-1)
                    P = P.flatten(2)  # [N, S, grad_dim]
                    P = P[self.collection_mask(module)]  # [total_valid, grad_dim]
                else:
                    P = g.mT @ a  # [N, O/p, I/p]
                    if bias_grad is not None:
                        P = torch.cat([P, bias_grad.unsqueeze(2)], dim=2)  # [N,O,I+1]

        P = P.flatten(1).clamp_(self.lo, self.hi)
        return P

    @abstractmethod
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]) -> None:
        """
        Process gradients during the backward pass.

        Args:
            module: The module whose backward pass triggered this hook. The module name
                is available via module._name.
            g: Gradient with respect to module output, shape [N, S, O] where N=batch
                size, S=sequence length, O=output dimension.
        """
        pass

    @abstractmethod
    def process_batch(self, indices: list[int], **kwargs) -> None:
        """
        Process collected data for a batch. This is called after each
        forward/backward pass. See also CollectorComputer.run_with_collector_hooks.

        Args:
            indices: List of data indices in the current batch
            **kwargs: Additional batch-specific data (e.g., losses)
        """
        pass


class CollectorComputer:
    """
    Orchestrates gradient collection by running forward/backward passes over a dataset.

    Iterates through batches of data, computes losses, triggers backpropagation, and
    delegates gradient processing to the provided collector. Supports distributed
    training and optional profiling via PyTorch profiler via cfg.profile flag.
    """

    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        data: Dataset,
        *,
        collector: HookCollectorBase,
        batches: list[list[int]] | None = None,
        cfg: IndexConfig,
    ):
        """
        Initialize the CollectorComputer.

        Args:
            model: The model to collect gradients from.
            data: HuggingFace Dataset containing input_ids and optionally labels.
            collector: A HookCollectorBase instance that will process the gradients
            via hooks.
            batches: List of index lists defining how to batch the data. If None,
                defaults to batch size 1 (each sample processed individually).
            cfg: IndexConfig controlling all other hyperparameters.
        """
        # Model
        self.model = model
        self.device = torch.device(model.device)  # type: ignore[attr-defined]

        # Data
        self.data = data
        # Batch size one by default
        if batches is None:
            batches = [[idx] for idx in range(len(data))]
        self.batches = batches

        self.forward_backward = fwd_bwd_factory(cfg)

        # Collector
        self.collector = collector

        # Other
        self.cfg = cfg
        self.logger = get_logger(
            "CollectorComputer", level="DEBUG" if cfg.debug else "INFO"
        )

        # Distributed related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.logger.info(
            "Tracking %d modules: %s",
            len(collector.target_info),
            list(collector.target_info.keys()),
        )

    def _setup_profiler(self):
        """Set up profiler if profiling is enabled."""
        if not self.cfg.profile:
            return nullcontext()

        trace_handler = tensorboard_trace_handler(
            dir_name="profiler_logs", worker_name=f"rank_{self.rank}", use_gzip=True
        )
        my_schedule = schedule(wait=0, warmup=0, active=4, repeat=1)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=trace_handler,
            schedule=my_schedule,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,
        )

        log_dir = "profiler_logs"
        os.makedirs(log_dir, exist_ok=True)

        return prof

    def run_with_collector_hooks(
        self,
        desc: Optional[str] = None,
    ):
        """
        Run the main computation loop over all batches.

        For each batch: computes forward pass, calculates loss, triggers backward pass
        (which invokes collector hooks), then calls collector.process_batch(). After
        all batches are processed, calls collector.teardown().

        Args:
            desc: Optional description string for the tqdm progress bar.
        """
        total_processed = torch.tensor(0, device=self.device)
        prof = self._setup_profiler()
        step = 0
        with prof:
            for indices in tqdm(
                self.batches,
                desc=f"Computing {desc}",
            ):
                batch = self.data[indices]

                # Local padding only: bin-packer enforces a per-rank
                # ``max_len × batch_size ≤ N`` budget that a global-max
                # all-reduce would silently violate.
                x, y, _, collection_mask = pad_and_tensor(
                    batch["input_ids"],
                    labels=batch.get("labels"),
                    device=self.device,
                    sync_max_len=False,
                )
                # Must count the same positions the collectors ingest.
                total_processed += collection_mask.sum()

                with (
                    self.collector.with_batch(collection_mask),
                    (
                        record_function(f"step_{step}")
                        if self.cfg.profile
                        else nullcontext()
                    ),
                ):
                    losses = self.forward_backward(self.model, x, y, batch)

                    # TODO: currently builder also calls torch.cuda.synchronize
                    torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.cfg.profile:
                    assert isinstance(prof, profile), "Profiler is not set up correctly"
                    prof.step()
                step += 1

                self.collector.process_batch(indices, losses=losses)

        self.collector.teardown()

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            torch.save(
                total_processed,
                os.path.join(self.cfg.partial_run_path, "total_processed.pt"),
            )
        self.logger.info(f"Total processed: {total_processed.item()}")


def fwd_bwd_factory(cfg: IndexConfig) -> Callable:
    """
    Create a forward/backward function based on the configuration.

    Args:
        cfg: IndexConfig that specifies:
            - cfg.loss_fn: Either "kl" for KL divergence (requires PEFT model) or
              any other value for cross-entropy loss.
            - cfg.loss_reduction: Either "mean" to average over tokens, or "sum" for
              summed loss.

    Returns:
        A callable fwd_bwd(model, x, y, batch) -> Tensor that performs a forward pass
        and backward pass, returning the per-sample losses.
        Args:
            model: The model to run forward/backward on.
            x: Padded input token ids tensor of shape [batch_size, seq_len].
            y: Padded label tensor of shape [batch_size, seq_len] with -100 for padding.
            batch: Original batch dict, used only for "advantage" if present.
        Returns a tensor of shape [batch_size] with one loss value per sample.
    """

    def fwd_bwd(model, x: Tensor, y: Tensor, batch: dict):
        logits = model(x).logits[:, :-1]
        masks = y[:, 1:] != -100
        denoms = (
            masks.sum(dim=1, dtype=model.dtype) if cfg.loss_reduction == "mean" else 1.0
        )

        if cfg.loss_fn == "kl":
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            ft_lps = torch.log_softmax(logits, dim=-1)

            # Compute average KL across all unmasked tokens
            kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
            losses = torch.sum(kls * masks, dim=-1) / denoms
            if "advantage" in batch:
                losses *= torch.tensor(batch["advantage"], device=losses.device)

        else:
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
                label_smoothing=cfg.label_smoothing,
            ).reshape_as(y[:, 1:])
            losses = losses.sum(1) / denoms
            if "advantage" in batch:
                losses *= torch.tensor(batch["advantage"], device=losses.device)

        losses.sum().backward()
        model.zero_grad()

        return losses

    return fwd_bwd


def fwd_bwd_hessian_factory(
    index_cfg: IndexConfig, hessian_cfg: HessianConfig
) -> Callable:
    def fwd_bwd_hessian(model, x: Tensor, y: Tensor, batch: dict):
        logits = model(x).logits[:, :-1]
        masks = y[:, 1:] != -100
        denoms = (
            masks.sum(dim=1, dtype=model.dtype)
            if index_cfg.loss_reduction == "mean"
            else 1.0
        )
        if hessian_cfg.use_dataset_labels:
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])
            losses = losses.sum(1) / denoms
        else:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                sampled_tokens = torch.multinomial(
                    probs.reshape(-1, probs.size(-1)),
                    num_samples=1,
                    replacement=True,
                ).reshape_as(y[:, 1:])
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                sampled_tokens.flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])
            losses = losses.sum(1) / denoms

        losses.sum().backward()
        model.zero_grad()

        return losses

    return fwd_bwd_hessian


def create_projection_matrix(
    identifier: str,
    m: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    projection_type: Literal["normal", "rademacher"] = "normal",
    projection_scale: Literal["jl", "row_norm"] = "jl",
) -> Tensor:
    """Create a projection matrix deterministically based on identifier."""
    # Seed the PRNG deterministically from the identifier string
    message = bytes(identifier, "utf-8")
    digest = hashlib.md5(message).digest()
    seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)

    if projection_type == "normal":
        prng = torch.Generator(device).manual_seed(seed)
        A = torch.randn(m, n, device=device, dtype=dtype, generator=prng)
    elif projection_type == "rademacher":
        numpy_rng = np.random.Generator(np.random.PCG64(seed))
        random_bytes = numpy_rng.bytes((m * n + 7) // 8)
        random_bytes = np.frombuffer(random_bytes, dtype=np.uint8)
        A = np.unpackbits(random_bytes)[: m * n].reshape((m, n))
        A = torch.from_numpy(A).to(device, dtype=dtype)
        A = A.add_(-0.5).mul_(2)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    if projection_scale == "row_norm":
        A /= A.norm(dim=1, keepdim=True)
    else:
        # divide by sqrt(m) to achieve variance of 1/m:
        # https://arxiv.org/html/2410.17413v1#A1.SS1.SSS2
        A /= math.sqrt(m)
    return A
