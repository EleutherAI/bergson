"""Gradient tracking for MoE layers whose experts and router are fused bare
``nn.Parameter``s rather than ``nn.Linear`` layers.

``transformers`` 5.x stores every MoE family's experts as 3D parameters behind
the ``use_experts_implementation`` decorator, and its router as a 2D
``[num_experts, hidden]`` parameter applied with ``F.linear``. Neither the
experts nor the router is an ``nn.Linear``, so the hook collectors skip both —
on gpt-oss that leaves only attention and ``lm_head`` tracked.

:func:`expand_moe` attaches one :class:`ExpertLinear` per expert projection and
swaps in :func:`bergson_experts_forward`, which routes each expert's tokens
through those submodules in a zero-padded ``[N, L, ·]`` grid (``L`` = the most
rows any one example sends to that expert). The collector's existing
``[N, S, ·]`` hooks then produce per-example gradients with no change to
gradient computation, because padding rows carry zero activations and receive
zero output gradient.

Detection is a capability check rather than a class allowlist, so the 47 model
families sharing the decorator are covered, as are families added later.
``llama4`` and ``longcat_flash`` hold fused expert parameters without adopting
the decorator's contract, and would each need a separate adapter.
"""

import types
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from bergson.utils.utils import assert_type

# Set on the experts class by transformers' ``use_experts_implementation``.
_EXPERTS_ATTRS = ("num_experts", "has_gate", "has_bias", "is_transposed")

EXPERT_PREFIX = "expert_"
"""Name of the per-expert submodule container, e.g. ``experts.expert_3``."""

_LAYOUT_ATTR = "_bergson_layout"
_BARE_LINEAR_ATTR = "_bergson_bare_linear"
_RESTORE_ATTR = "_bergson_moe_restore"
_ROUTER_ANNOTATIONS = ("out_features", "in_features", _LAYOUT_ATTR, _BARE_LINEAR_ATTR)


@dataclass
class BatchLayout:
    """Batch size of the block currently flowing through a MoE layer.

    A fused experts module only ever sees ``[N*S, hidden]``, so which example a
    token came from is not recoverable from the experts module's own arguments.
    A pre-hook on the enclosing block records the batch size here.
    """

    num_examples: int = 1


class ExpertLinear(nn.Module):
    """One expert's slice of a fused MoE parameter, presented as a linear layer.

    ``weight`` and ``bias`` are *views* of the parent's fused parameter and are
    deliberately not registered, so ``named_parameters()``, ``state_dict()`` and
    parameter counts are unchanged by expansion — only ``named_modules()`` gains
    entries. ``weight`` keeps the parent's storage orientation, reported by
    :meth:`LayerAdapter.weight_transposed` as for HF ``Conv1D``.
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


# ── Detection ────────────────────────────────────────────────────────────────


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
    router subclasses ``nn.Linear``) and is left alone: a real linear module is
    already discoverable, and annotating one would clobber its own attributes.
    """
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, nn.Parameter)
        and weight.ndim == 2
        and weight.shape[0] == num_experts
        and not hasattr(module, "in_features")
    )


def is_bare_linear(module: nn.Module) -> bool:
    """Whether ``module`` is a router :func:`expand_moe` annotated for tracking."""
    return getattr(module, _BARE_LINEAR_ATTR, False)


def batch_layout(module: nn.Module) -> BatchLayout | None:
    """The MoE batch layout attached to ``module``, if any."""
    return getattr(module, _LAYOUT_ATTR, None)


def weight_names(experts: nn.Module) -> tuple[str, str]:
    """The up- and down-projection parameter names of a fused experts module."""
    return ("gate_up_proj" if experts.has_gate else "up_proj", "down_proj")  # type: ignore[attr-defined]


# ── Forward ──────────────────────────────────────────────────────────────────


@dataclass
class _Grid:
    """Where one expert's routed tokens sit in a padded ``[N, L]`` grid.

    Rows routed to an expert are ragged across examples; placing them in a
    rectangular grid is what lets the collector's ``[N, S, ·]`` machinery run
    unchanged. Padding cells stay zero and so contribute no gradient.
    """

    example: Tensor
    """Row -> example id, shape ``[num_rows]``."""

    slot: Tensor
    """Row -> column within its example, shape ``[num_rows]``."""

    width: int
    """``L``, the widest example's row count."""

    mask: Tensor
    """``[N, L]``, true where a real routed token sits."""

    @classmethod
    def build(cls, token_idx: Tensor, num_examples: int, seq_len: int) -> "_Grid":
        """Place the rows named by ``token_idx`` (non-decreasing, from a
        row-major ``torch.where``) into a grid."""
        example = token_idx // seq_len
        counts = torch.bincount(example, minlength=num_examples)
        starts = counts.cumsum(0) - counts
        arange = torch.arange(token_idx.numel(), device=token_idx.device)
        # Width >= 1 so the shim's backward hook still fires, contributing a
        # zero gradient, for an expert that received no tokens this batch:
        # Builder concatenates every module in shapes() and a missing key is a
        # hard failure.
        width = max(int(counts.max()), 1)

        slot = arange - starts[example]
        mask = torch.zeros(
            num_examples, width, dtype=torch.bool, device=token_idx.device
        )
        mask[example, slot] = True
        return cls(example=example, slot=slot, width=width, mask=mask)

    def scatter(self, rows: Tensor) -> Tensor:
        """``[num_rows, ...]`` -> a zero-padded ``[N, L, ...]`` grid."""
        grid = rows.new_zeros(*self.mask.shape, *rows.shape[1:])
        grid[self.example, self.slot] = rows
        return grid

    def gather(self, grid: Tensor) -> Tensor:
        """``[N, L, ...]`` -> the ``[num_rows, ...]`` cells holding real tokens."""
        return grid[self.example, self.slot]


def bergson_experts_forward(
    self: nn.Module,
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
) -> Tensor:
    """Per-expert MoE forward that exposes each expert as a linear submodule.

    Numerically equivalent to transformers' own experts implementations, but
    each expert's tokens are gathered into a padded grid so the collector's
    hooks see the ``[N, S, ·]`` layout they need. Experts with no routed tokens
    still run, on an all-zero grid.
    """
    num_tokens = hidden_states.shape[0]
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
        # Expert-parallel sentinels (index == num_experts) never match, which is
        # exactly right: their routing weight is zero.
        token_idx, k_slot = torch.where(top_k_index == expert_idx)
        grid = _Grid.build(token_idx, num_examples, seq_len)

        shims = getattr(self, f"{EXPERT_PREFIX}{expert_idx}")
        for shim in shims.children():
            # Collectors that select gradient-carrying positions (the EK-FAC
            # covariance factors) need this expert's rows, not the batch's.
            shim._row_mask = grid.mask

        h = getattr(shims, up_name)(grid.scatter(hidden_states[token_idx]))
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)  # type: ignore[attr-defined]
        h = getattr(shims, down_name)(h)
        h = h * grid.scatter(top_k_weights[token_idx, k_slot]).unsqueeze(-1)

        out = out.index_add(0, token_idx, grid.gather(h).to(out.dtype))

    return out


# ── Expansion ────────────────────────────────────────────────────────────────


@dataclass
class _Expansion:
    """What :func:`expand_moe` added to one MoE block, so it can be undone."""

    experts: nn.Module
    containers: list[str]
    """Names of the per-expert child modules added to ``experts``."""
    routers: list[nn.Module]
    hook: RemovableHandle
    """The enclosing block's batch-layout pre-hook."""


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
        if is_fused_experts(module):
            yield model.get_submodule(name.rpartition(".")[0]), module


def _attach_expert_shims(experts: nn.Module) -> list[str]:
    """Add one :class:`ExpertLinear` per projection per expert.

    Returns the names of the containers added to ``experts``, which give the
    shims readable paths like ``experts.expert_3.gate_up_proj``.
    """
    names = []
    for expert_idx in range(experts.num_experts):  # type: ignore[attr-defined]
        container = nn.Module()
        for weight_name in weight_names(experts):
            container.add_module(
                weight_name, ExpertLinear(experts, weight_name, expert_idx)
            )
        name = f"{EXPERT_PREFIX}{expert_idx}"
        experts.add_module(name, container)
        names.append(name)
    return names


def _annotate_router(router: nn.Module, layout: BatchLayout) -> None:
    """Give a bare-parameter router the metadata ``LayerAdapter`` reads.

    The router's ``F.linear`` is left untouched; the router is tracked in place.
    """
    out_features, in_features = assert_type(Tensor, router.weight).shape
    setattr(router, "out_features", int(out_features))
    setattr(router, "in_features", int(in_features))
    setattr(router, _LAYOUT_ATTR, layout)
    setattr(router, _BARE_LINEAR_ATTR, True)


def expand_moe(model: nn.Module) -> list[str]:
    """Expose fused MoE experts and routers to gradient collection, in place.

    Idempotent, including when called with two different roots over the same
    layers (a model and its ``base_model``). Returns the names of the newly
    trackable modules, relative to ``model``. Reverse with :func:`restore_moe`.
    """
    if getattr(model, _RESTORE_ATTR, None) is not None:
        return []

    expansions: list[_Expansion] = []
    for block, experts in _fused_experts_blocks(model):
        if batch_layout(experts) is not None:
            continue  # already expanded under another root

        layout = BatchLayout()
        containers = _attach_expert_shims(experts)
        setattr(experts, _LAYOUT_ATTR, layout)
        experts.forward = types.MethodType(bergson_experts_forward, experts)

        routers = [
            child
            for child in block.children()
            if is_fused_router(child, experts.num_experts)  # type: ignore[attr-defined]
        ]
        for router in routers:
            _annotate_router(router, layout)

        expansions.append(
            _Expansion(
                experts=experts,
                containers=containers,
                routers=routers,
                hook=block.register_forward_pre_hook(_record_layout(layout)),
            )
        )

    setattr(model, _RESTORE_ATTR, expansions)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, ExpertLinear) or is_bare_linear(module)
    ]


def restore_moe(model: nn.Module) -> None:
    """Undo :func:`expand_moe`, leaving ``model`` exactly as it was found."""
    expansions: list[_Expansion] | None = getattr(model, _RESTORE_ATTR, None)
    if expansions is None:
        return

    for expansion in expansions:
        expansion.hook.remove()
        for container in expansion.containers:
            delattr(expansion.experts, container)
        del expansion.experts.forward
        delattr(expansion.experts, _LAYOUT_ATTR)
        for router in expansion.routers:
            for attr in _ROUTER_ANNOTATIONS:
                delattr(router, attr)

    delattr(model, _RESTORE_ATTR)
