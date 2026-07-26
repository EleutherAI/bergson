"""Gradient tracking for MoE layers whose experts and router are fused bare
``nn.Parameter``s rather than ``nn.Linear`` layers.

``transformers`` 5.x stores every MoE family's experts as 3D parameters
(``[num_experts, ...]``) behind the ``use_experts_implementation`` decorator, and
its router as a 2D ``[num_experts, hidden]`` parameter applied with ``F.linear``.
Neither the experts nor the router is an ``nn.Linear``, so the hook collectors
silently skip the experts and the router alike — on gpt-oss that leaves only
attention and ``lm_head`` tracked.

:func:`expand_moe` closes the gap by attaching one :class:`ExpertLinear`
submodule per expert projection and swapping in :func:`bergson_experts_forward`,
which routes each expert's tokens through those ``ExpertLinear`` submodules in a
zero-padded ``[N, L, ·]`` layout (``L`` = the most rows any one example sends to
that expert). The collector's existing ``[N, S, ·]`` hooks then yield per-example
gradients with no changes to gradient computation: padding rows carry zero
activations and receive zero output gradient, so the padding rows contribute
nothing to the weight gradient, the bias gradient, or any Hessian factor.

Detection is a capability check rather than a class allowlist, so the 47 model
families sharing the decorator — and families added later — are covered on
arrival. ``llama4`` and ``longcat_flash`` hold fused expert parameters without
adopting the decorator's contract, and would each need a separate adapter.
"""

import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from bergson.utils.utils import assert_type

# Set on the experts class by transformers' ``use_experts_implementation``.
_EXPERTS_ATTRS = ("num_experts", "has_gate", "has_bias", "is_transposed")

EXPERT_PREFIX = "expert_"
"""Name of the per-expert submodule container, e.g. ``experts.expert_3``."""

_LAYOUT_ATTR = "_bergson_layout"
_BARE_LINEAR_ATTR = "_bergson_bare_linear"
_RESTORE_ATTR = "_bergson_moe_restore"


@dataclass
class BatchLayout:
    """Number of examples in the batch currently flowing through a MoE block.

    A fused experts module only ever sees ``[N*S, hidden]``, so the example a
    token belongs to cannot be recovered from the experts module's own
    arguments. A forward pre-hook on the enclosing MoE block — whose input is
    still ``[N, S, hidden]`` — records the batch size here, and the block's
    experts and router both read the batch size from this object.
    """

    num_examples: int = 1


class ExpertLinear(nn.Module):
    """One expert's slice of a fused MoE parameter, presented as a linear layer.

    ``weight`` and ``bias`` are *views* of the parent's fused parameter and are
    deliberately not registered, so ``named_parameters()``, ``state_dict()`` and
    parameter counts are unchanged by expansion — only ``named_modules()`` gains
    entries. ``weight`` is returned in the parent's storage orientation and
    described by :meth:`LayerAdapter.weight_transposed`, as for HF ``Conv1D``.
    """

    def __init__(self, experts: nn.Module, weight_name: str, expert_idx: int):
        super().__init__()
        # Bypass nn.Module.__setattr__ so the parent is not registered as a
        # child, which would make named_modules() recurse forever.
        self.__dict__["_experts"] = experts
        self.weight_name = weight_name
        self.expert_idx = expert_idx
        self.transposed = bool(experts.is_transposed)  # type: ignore[attr-defined]

        rows, cols = self.weight.shape
        self.in_features, self.out_features = (
            (rows, cols) if self.transposed else (cols, rows)
        )

    @property
    def weight(self) -> Tensor:
        return getattr(self._experts, self.weight_name)[self.expert_idx]

    @property
    def bias(self) -> Tensor | None:
        bias = getattr(self._experts, f"{self.weight_name}_bias", None)
        return None if bias is None else bias[self.expert_idx]

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        out = x @ weight if self.transposed else x @ weight.mT
        bias = self.bias
        return out if bias is None else out + bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"expert={self.expert_idx}, param={self.weight_name!r}"
        )


def is_bare_linear(module: nn.Module) -> bool:
    """Whether ``module`` is a bare-``nn.Parameter`` linear op (a MoE router)
    that :func:`expand_moe` has annotated for in-place tracking."""
    return getattr(module, _BARE_LINEAR_ATTR, False)


def batch_layout(module: nn.Module) -> BatchLayout | None:
    """The MoE batch layout attached to ``module``, if any."""
    return getattr(module, _LAYOUT_ATTR, None)


def is_fused_experts(module: nn.Module) -> bool:
    """Whether ``module`` is a fused-parameter MoE experts module."""
    down = getattr(module, "down_proj", None)
    return (
        all(hasattr(module, attr) for attr in _EXPERTS_ATTRS)
        and isinstance(down, nn.Parameter)
        and down.ndim == 3
    )


def is_fused_router(module: nn.Module, num_experts: int) -> bool:
    """Whether ``module`` is a bare-``nn.Parameter`` router over ``num_experts``.

    Keyed on the weight's shape rather than on attribute names, which differ
    across families (``num_experts`` vs ``n_routed_experts``). A router that
    already presents linear-layer metadata is a real linear module (Llama4's
    router subclasses ``nn.Linear``), so such a router is left alone: a real
    linear module is already discoverable, and annotating one would clobber the
    module's own ``in_features`` / ``out_features``.
    """
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, nn.Parameter)
        and weight.ndim == 2
        and weight.shape[0] == num_experts
        and not hasattr(module, "in_features")
    )


def weight_names(experts: nn.Module) -> tuple[str, str]:
    """The up- and down-projection parameter names of a fused experts module."""
    return ("gate_up_proj" if experts.has_gate else "up_proj", "down_proj")  # type: ignore[attr-defined]


def _pad_rows(
    token_idx: Tensor, num_examples: int, seq_len: int
) -> tuple[Tensor, Tensor, int]:
    """Map flat token indices to ``(example, slot)`` coordinates in a padded grid.

    ``token_idx`` is non-decreasing, coming from a row-major ``torch.where``, so
    each row's slot is the offset of that row within the run of rows belonging
    to the same example. Returns the example ids, the slots, and the grid width
    ``L``.
    """
    example = token_idx // seq_len
    counts = torch.bincount(example, minlength=num_examples)
    starts = counts.cumsum(0) - counts
    slot = torch.arange(token_idx.numel(), device=token_idx.device) - starts[example]
    # Keep L >= 1 so the shim's backward hook still fires (and contributes a
    # zero gradient) for an expert that received no tokens this batch: Builder
    # concatenates every module in shapes() and a missing key is a hard failure.
    return example, slot, max(int(counts.max()), 1)


def bergson_experts_forward(
    self: nn.Module,
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
) -> Tensor:
    """Per-expert MoE forward that exposes each expert as a linear submodule.

    Numerically equivalent to transformers' own experts implementations, but each
    expert's tokens are gathered into a zero-padded ``[N, L, ·]`` slab so the
    collector's hooks see the ``[N, S, ·]`` layout the hooks need for per-example
    gradients. Experts with no routed tokens still run (on an all-zero slab).
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_examples = getattr(self, _LAYOUT_ATTR).num_examples
    assert num_tokens % num_examples == 0, (
        f"{num_tokens} tokens do not divide into {num_examples} examples; the "
        f"enclosing MoE block's recorded batch layout is stale."
    )
    seq_len = num_tokens // num_examples

    top_k_index = top_k_index.reshape(num_tokens, -1)
    top_k_weights = top_k_weights.reshape(num_tokens, -1)
    up_name, down_name = weight_names(self)
    out = torch.zeros_like(hidden_states)

    for expert_idx in range(self.num_experts):  # type: ignore[attr-defined]
        shims = getattr(self, f"{EXPERT_PREFIX}{expert_idx}")
        # Expert-parallel sentinels (index == num_experts) never match, which is
        # exactly right: their routing weight is zero.
        token_idx, k_slot = torch.where(top_k_index == expert_idx)
        example, slot, width = _pad_rows(token_idx, num_examples, seq_len)

        a = hidden_states.new_zeros(num_examples, width, hidden_dim)
        a[example, slot] = hidden_states[token_idx]
        weights = top_k_weights.new_zeros(num_examples, width)
        weights[example, slot] = top_k_weights[token_idx, k_slot]

        # Which grid cells hold a real routed token, for collectors that select
        # gradient-carrying positions (the EK-FAC covariance factors).
        row_mask = torch.zeros(
            num_examples, width, dtype=torch.bool, device=hidden_states.device
        )
        row_mask[example, slot] = True

        for shim in shims.children():
            shim._row_mask = row_mask

        h = getattr(shims, up_name)(a)
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)  # type: ignore[attr-defined]
        h = getattr(shims, down_name)(h)
        h = h * weights.unsqueeze(-1)

        out = out.index_add(0, token_idx, h[example, slot].to(out.dtype))

    return out


def _record_layout(layout: BatchLayout):
    """Forward pre-hook recording the batch size of a MoE block's input."""

    def hook(module: nn.Module, args: tuple):
        x = args[0] if args else None
        if not isinstance(x, Tensor) or x.ndim != 3:
            raise RuntimeError(
                f"Expected {type(module).__name__} to receive a [N, S, hidden] "
                f"input so MoE tokens can be attributed to examples, got "
                f"{None if x is None else tuple(x.shape)}. Disable MoE expert "
                f"tracking with --track_moe_experts false."
            )
        layout.num_examples = x.shape[0]

    return hook


def _fused_experts_blocks(model: nn.Module) -> Iterator[tuple[nn.Module, nn.Module]]:
    """Yield ``(block, experts)`` for every fused-parameter MoE layer."""
    for name, module in model.named_modules():
        if not is_fused_experts(module):
            continue
        parent_name = name.rpartition(".")[0]
        yield model.get_submodule(parent_name), module


def expand_moe(model: nn.Module) -> list[str]:
    """Expose fused MoE experts and routers to gradient collection, in place.

    Idempotent, including when called with two different roots over the same
    layers (a model and its ``base_model``). Returns the names of the newly
    trackable modules, relative to ``model``. Reverse with :func:`restore_moe`.
    """
    if getattr(model, _RESTORE_ATTR, None) is not None:
        return []

    restore: list = []
    for block, experts in _fused_experts_blocks(model):
        if batch_layout(experts) is not None:
            continue  # already expanded under another root
        layout = BatchLayout()
        up_name, down_name = weight_names(experts)

        containers = []
        for expert_idx in range(experts.num_experts):  # type: ignore[attr-defined]
            container = nn.Module()
            for weight_name in (up_name, down_name):
                container.add_module(
                    weight_name, ExpertLinear(experts, weight_name, expert_idx)
                )
            child = f"{EXPERT_PREFIX}{expert_idx}"
            experts.add_module(child, container)
            containers.append(child)

        setattr(experts, _LAYOUT_ATTR, layout)
        experts.forward = types.MethodType(bergson_experts_forward, experts)
        handle = block.register_forward_pre_hook(_record_layout(layout))

        routers = [
            router
            for router in block.children()
            if is_fused_router(router, experts.num_experts)  # type: ignore[attr-defined]
        ]
        for router in routers:
            # The router's F.linear stays untouched; it is tracked in place, so
            # it only needs the metadata LayerAdapter reads off a linear layer.
            out_features, in_features = assert_type(Tensor, router.weight).shape
            setattr(router, "out_features", int(out_features))
            setattr(router, "in_features", int(in_features))
            setattr(router, _LAYOUT_ATTR, layout)
            setattr(router, _BARE_LINEAR_ATTR, True)

        restore.append((block, experts, containers, handle, routers))

    setattr(model, _RESTORE_ATTR, restore)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, ExpertLinear) or getattr(module, _BARE_LINEAR_ATTR, False)
    ]


def restore_moe(model: nn.Module) -> None:
    """Undo :func:`expand_moe`, leaving ``model`` exactly as it was found."""
    restore = getattr(model, _RESTORE_ATTR, None)
    if restore is None:
        return

    for _block, experts, containers, handle, routers in restore:
        handle.remove()
        for child in containers:
            delattr(experts, child)
        del experts.forward
        delattr(experts, _LAYOUT_ATTR)
        for router in routers:
            for attr in (
                "out_features",
                "in_features",
                _LAYOUT_ATTR,
                _BARE_LINEAR_ATTR,
            ):
                delattr(router, attr)

    delattr(model, _RESTORE_ATTR)


@contextmanager
def moe_expanded(model: nn.Module, enabled: bool = True):
    """Scope :func:`expand_moe` to a block, restoring the model on exit.

    Only restores what this call expanded, so nesting does not tear an outer
    scope's expansion down early.
    """
    owned = enabled and getattr(model, _RESTORE_ATTR, None) is None
    names = expand_moe(model) if enabled else []
    try:
        yield names
    finally:
        if owned:
            restore_moe(model)
