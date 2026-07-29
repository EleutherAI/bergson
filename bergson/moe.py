"""Gradient tracking for MoE layers with fused expert parameters.

In transformers 5.x an MoE layer stores its experts as one 3D ``nn.Parameter``
and its router as a 2D one, so neither is an ``nn.Linear`` and the collectors
skip both.
:func:`expand_moe` attaches an :class:`ExpertLinear` per expert projection and
swaps in :func:`bergson_experts_forward`, which pads each expert's routed tokens
into ``[N, L, ...]`` so the collectors see their usual layout. Padding rows get
zero gradient, so gradient computation is unchanged.

Detection tests for the attributes ``use_experts_implementation`` sets.
``llama4`` and ``longcat_flash`` fuse experts without it and are not covered.
"""

import types

import torch
import torch.nn as nn
from torch import Tensor

from bergson.utils.utils import assert_type

_EXPERTS_ATTRS = ("num_experts", "has_gate", "has_bias", "is_transposed")

# An experts module only sees [N*S, hidden], so a pre-hook on the enclosing block
# leaves the batch size in _BATCH_ATTR. _EXPANSION_ATTR holds what expand_moe
# added, for restore_moe to undo.
_BATCH_ATTR = "_bergson_num_examples"
_EXPANSION_ATTR = "_bergson_expansion"
# A tracked router points back at its experts module, whose batch size it shares.
_ROUTER_ATTR = "_bergson_router_experts"


class ExpertLinear(nn.Module):
    """One expert's slice of a fused MoE parameter, as a linear layer.

    ``weight`` and ``bias`` are unregistered views, so only ``named_modules()``
    grows. ``weight`` keeps the parent's orientation, which
    ``LayerAdapter.weight_transposed`` reports.
    """

    def __init__(self, experts: nn.Module, weight_name: str, expert_idx: int):
        super().__init__()
        # Not setattr: registering the parent makes named_modules() recurse.
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


def is_fused_experts(module: nn.Module) -> bool:
    """Whether ``module`` is a fused-parameter MoE experts module."""
    down = getattr(module, "down_proj", None)
    return (
        all(hasattr(module, attr) for attr in _EXPERTS_ATTRS)
        and isinstance(down, nn.Parameter)
        and down.ndim == 3
    )


def is_fused_router(module: nn.Module, num_experts: int) -> bool:
    """A bare-parameter router: 2D [num_experts, hidden] weight. Routers that are
    already linear modules (Llama4's subclasses nn.Linear) are left alone, since
    annotating one would overwrite its own attributes."""
    w = getattr(module, "weight", None)
    return (
        isinstance(w, nn.Parameter)
        and w.ndim == 2
        and w.shape[0] == num_experts
        and not hasattr(module, "in_features")
    )


def tracked_router(module: nn.Module) -> nn.Module | None:
    """The experts module a tracked router belongs to, or None."""
    return getattr(module, _ROUTER_ATTR, None)


def router_batch_size(router: nn.Module) -> int:
    """Batch size recorded for a tracked router's block this forward."""
    return getattr(getattr(router, _ROUTER_ATTR), _BATCH_ATTR)


def weight_names(experts: nn.Module) -> tuple[str, str]:
    return ("gate_up_proj" if experts.has_gate else "up_proj", "down_proj")  # type: ignore[attr-defined]


def _grid(token_idx: Tensor, num_examples: int, seq_len: int):
    """Row -> (example, column) in a padded grid, plus the [N, L] mask of real
    rows. Rows arrive grouped by example, from a row-major torch.where."""
    example = token_idx // seq_len
    counts = torch.bincount(example, minlength=num_examples)
    starts = counts.cumsum(0) - counts
    slot = torch.arange(token_idx.numel(), device=token_idx.device) - starts[example]

    # An expert with no tokens still needs one column, or its backward hook never
    # fires and Builder comes up a key short.
    mask = torch.zeros(
        num_examples,
        max(int(counts.max()), 1),
        dtype=torch.bool,
        device=token_idx.device,
    )
    mask[example, slot] = True
    return example, slot, mask


def bergson_experts_forward(
    self: nn.Module,
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
) -> Tensor:
    """Per-expert MoE forward, matching the implementations in transformers but
    padded so the collectors see [N, S, ...]. Experts with no tokens still run."""
    num_tokens = hidden_states.shape[0]
    num_examples = getattr(self, _BATCH_ATTR)
    assert (
        num_tokens % num_examples == 0
    ), f"{num_tokens} tokens do not divide into {num_examples} examples"
    seq_len = num_tokens // num_examples

    top_k_index = top_k_index.reshape(num_tokens, -1)
    top_k_weights = top_k_weights.reshape(num_tokens, -1)
    up_name, down_name = weight_names(self)
    out = torch.zeros_like(hidden_states)

    for expert_idx in range(self.num_experts):  # type: ignore[attr-defined]
        # Expert-parallel sentinels get zero routing weight, so never matching
        # them is what we want.
        token_idx, k_slot = torch.where(top_k_index == expert_idx)
        example, slot, mask = _grid(token_idx, num_examples, seq_len)

        shims = getattr(self, f"expert_{expert_idx}")
        for shim in shims.children():
            shim._row_mask = mask  # collectors mask on this, not [N, S]

        a = hidden_states.new_zeros(*mask.shape, hidden_states.shape[-1])
        a[example, slot] = hidden_states[token_idx]
        weights = top_k_weights.new_zeros(mask.shape)
        weights[example, slot] = top_k_weights[token_idx, k_slot]

        h = getattr(shims, up_name)(a)
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)  # type: ignore[attr-defined]
        h = getattr(shims, down_name)(h)
        h = h * weights.unsqueeze(-1)

        out = out.index_add(0, token_idx, h[example, slot].to(out.dtype))

    return out


def _record_batch_size(experts: nn.Module):
    def hook(module: nn.Module, args: tuple):
        x = args[0]
        assert x.ndim == 3, f"{type(module).__name__} input is not [N, S, hidden]"
        setattr(experts, _BATCH_ATTR, x.shape[0])

    return hook


def expand_moe(model: nn.Module) -> list[str]:
    """Expose fused MoE experts to collection, in place. Undo with restore_moe.

    Safe to call twice, and from either a model or its base_model: the undo state
    hangs off the experts module, not off whichever root was passed.
    """
    for name, experts in list(model.named_modules()):
        if not is_fused_experts(experts) or hasattr(experts, _EXPANSION_ATTR):
            continue

        containers = []
        for expert_idx in range(experts.num_experts):  # type: ignore[attr-defined]
            container = nn.Module()
            for weight_name in weight_names(experts):
                container.add_module(
                    weight_name, ExpertLinear(experts, weight_name, expert_idx)
                )
            child = f"expert_{expert_idx}"
            experts.add_module(child, container)
            containers.append(child)

        setattr(experts, _BATCH_ATTR, 1)
        experts.forward = types.MethodType(bergson_experts_forward, experts)
        block = model.get_submodule(name.rpartition(".")[0])

        # The router keeps its own F.linear; it only needs the metadata
        # LayerAdapter reads, and a way to reach the batch size.
        num_experts = int(experts.num_experts)  # type: ignore[arg-type]
        routers = [c for c in block.children() if is_fused_router(c, num_experts)]
        for router in routers:
            out_features, in_features = assert_type(Tensor, router.weight).shape
            setattr(router, "out_features", int(out_features))
            setattr(router, "in_features", int(in_features))
            # Via __dict__: setattr would register the experts module as a child
            # of the router, dragging its parameters into the router's path.
            router.__dict__[_ROUTER_ATTR] = experts

        handle = block.register_forward_pre_hook(_record_batch_size(experts))
        setattr(experts, _EXPANSION_ATTR, (containers, routers, handle))

    return [
        n
        for n, m in model.named_modules()
        if isinstance(m, ExpertLinear) or tracked_router(m) is not None
    ]


def restore_moe(model: nn.Module) -> None:
    """Undo expand_moe, from any root containing the experts."""
    for experts in list(model.modules()):
        if not hasattr(experts, _EXPANSION_ATTR):
            continue

        containers, routers, handle = getattr(experts, _EXPANSION_ATTR)
        handle.remove()
        for container in containers:
            delattr(experts, container)
        for router in routers:
            for attr in ("out_features", "in_features", _ROUTER_ATTR):
                delattr(router, attr)
        del experts.forward
        delattr(experts, _BATCH_ATTR)
        delattr(experts, _EXPANSION_ATTR)
