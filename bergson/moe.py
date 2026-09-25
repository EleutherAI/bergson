"""Expose an MoE layer's fused experts as one module per expert projection.

``transformers`` 5.x stores every expert of a layer in one 3D ``nn.Parameter``,
which gradient collection skips. :func:`expand_moe` attaches an
:class:`ExpertLinear` per expert projection and runs the experts in a loop.
"""

import types
from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch
import torch.nn as nn
from torch import Tensor

# The experts module only sees hidden states flattened to [N*S, hidden], so a
# pre-hook on its parent records the batch size here for the forward to read.
NUM_EXAMPLES_ATTR = "_bergson_num_examples"


@dataclass
class ExpertRows:
    """Where an expert's routed rows came from, and where they pack to.

    ``example`` and ``column`` place each row in a grid of one row per example,
    which is the layout the gradient reduction needs; ``position`` and ``valid``
    map a row back to the token it was routed from, for the collection mask.
    """

    example: Tensor
    """Which example each row was routed from, [T]."""

    position: Tensor
    """Which sequence position each row was routed from, [T]."""

    valid: Tensor
    """False for the placeholder row of an expert that was routed nothing, [T]."""

    column: Tensor
    """Each row's slot within its example, [T]."""

    num_examples: int
    width: int
    """Grid shape: the number of examples and the widest example's row count."""

    def to_grid(self, x: Tensor) -> Tensor:
        """Pack rows ``[T, C]`` into one row per example, ``[N, width, C]``.

        Slots no row landed on stay zero and drop out of the sum over the grid.
        """
        grid = x.new_zeros(self.num_examples, self.width, x.shape[-1])
        grid[self.example, self.column] = x
        return grid


class ExpertLinear(nn.Module):
    """Present one expert's slice of a fused MoE parameter as a linear layer.

    ``weight`` and ``bias`` are unregistered views in the parent's orientation,
    so ``state_dict`` is unchanged.
    """

    def __init__(self, experts: nn.Module, weight_name: str, expert_idx: int):
        super().__init__()

        # Via __dict__: registering the parent as a child would make
        # named_modules() recurse into it.
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
        out = x @ self.weight if self.transposed else x @ self.weight.mT
        bias = self.bias
        return out if bias is None else out + bias


def projection_names(experts: nn.Module) -> tuple[str, str]:
    """Name the fused parameters holding each expert's projections.

    A fused gate shows up in the name of the up projection, which
    ``transformers`` reports as ``has_gate`` from 5.3 on.
    """
    up = "gate_up_proj" if hasattr(experts, "gate_up_proj") else "up_proj"
    return (up, "down_proj")


def moe_forward(
    self: nn.Module,
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
) -> Tensor:
    """Run each expert over the tokens routed to it, one ExpertLinear at a time.

    Each expert sees only its routed rows, and records an :class:`ExpertRows`
    telling the collector which example and position each row came from.
    """
    assert hidden_states.ndim == 2, f"Expected [N*S, hidden], got {hidden_states.shape}"
    num_tokens = hidden_states.shape[0]
    num_examples = getattr(self, NUM_EXAMPLES_ATTR)
    assert (
        num_tokens % num_examples == 0
    ), f"{num_tokens} tokens do not divide into {num_examples} examples"
    seq_len = num_tokens // num_examples

    top_k_index = top_k_index.reshape(num_tokens, -1)
    top_k_weights = top_k_weights.reshape(num_tokens, -1)
    up_name, down_name = projection_names(self)
    gated = up_name == "gate_up_proj"
    out = torch.zeros_like(hidden_states)

    for expert_idx in range(self.num_experts):  # type: ignore[attr-defined]
        # Expert-parallel layouts route padding past the last expert, which
        # this comparison drops.
        picked = top_k_index == expert_idx  # [N*S, top_k]
        weights = (top_k_weights * picked).sum(-1)  # [N*S]
        routed = picked.any(-1).view(num_examples, seq_len)  # [N, S]

        example, pos = routed.nonzero(as_tuple=True)
        col = (routed.cumsum(1) - 1)[example, pos]
        valid = torch.ones_like(pos, dtype=torch.bool)
        if not len(pos):
            # An expert with no tokens still runs, or its backward hook never
            # fires and the index comes up a module short. Its routing weight is
            # zero, so the row it borrows contributes nothing.
            example = col = pos = pos.new_zeros(1)
            valid = valid.new_zeros(1)

        expert = getattr(self, f"expert_{expert_idx}")
        width = max(int(routed.sum(1).max()), 1)
        rows = ExpertRows(example, pos, valid, col, num_examples, width)
        for projection in expert.children():
            projection._rows = rows

        token = example * seq_len + pos
        h = getattr(expert, up_name)(hidden_states[token])  # [T, up]
        h = self._apply_gate(h) if gated else self.act_fn(h)  # type: ignore[attr-defined]
        h = getattr(expert, down_name)(h)

        h = h * weights[token].unsqueeze(-1)
        out = out.index_add(0, token, h.to(out.dtype))

    return out


def _record_num_examples(experts: nn.Module):
    """Build a pre-hook recording its module's batch size on ``experts``."""

    def hook(module: nn.Module, args: tuple):
        assert (
            args and args[0].ndim == 3
        ), f"Expected [N, S, hidden] into {type(module).__name__}"
        setattr(experts, NUM_EXAMPLES_ATTR, args[0].shape[0])

    return hook


def _check_fused(name: str, experts: nn.Module) -> None:
    """Fail on a pattern that matched something other than a fused MoE layer."""
    down = getattr(experts, "down_proj", None)
    up = getattr(experts, projection_names(experts)[0], None)
    attrs = ("num_experts", "is_transposed")
    if (
        not isinstance(down, nn.Parameter)
        or not isinstance(up, nn.Parameter)
        or down.ndim != 3
        or up.ndim != 3
        or not all(hasattr(experts, attr) for attr in attrs)
    ):
        raise ValueError(
            f"moe_experts matched '{name}' ({type(experts).__name__}), which has "
            f"no 3D expert parameters; expected e.g. 'model.layers.*.mlp.experts'."
        )


def expand_moe(model: nn.Module, patterns: str) -> list[str]:
    """Give every expert projection of the matching MoE layers its own module.

    ``patterns`` is a comma-separated list of globs over ``named_modules()``.
    Returns the names of the modules it exposes.
    """
    names = []
    matched = []

    for name, experts in list(model.named_modules()):
        if not any(fnmatchcase(name, p.strip()) for p in patterns.split(",")):
            continue

        matched.append(name)
        _check_fused(name, experts)

        projections = projection_names(experts)
        num_experts = int(experts.num_experts)  # type: ignore[arg-type]
        names += [
            f"{name}.expert_{i}.{w}" for i in range(num_experts) for w in projections
        ]
        if hasattr(experts, "expert_0"):
            continue

        for expert_idx in range(num_experts):
            expert = nn.Module()
            for weight_name in projections:
                expert.add_module(
                    weight_name, ExpertLinear(experts, weight_name, expert_idx)
                )
            experts.add_module(f"expert_{expert_idx}", expert)

        setattr(experts, NUM_EXAMPLES_ATTR, 1)
        experts.forward = types.MethodType(moe_forward, experts)
        parent = model.get_submodule(name.rpartition(".")[0])
        parent.register_forward_pre_hook(_record_num_examples(experts))

    if not matched:
        raise ValueError(
            f"moe_experts={patterns!r} matched no module, "
            f"e.g. 'model.layers.*.mlp.experts'."
        )
    return names
