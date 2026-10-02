"""Tensor-parallel linears for MAGIC: compute on the weight shard.

``simple_fsdp`` shards storage but replicates computation, so every weight is
all-gathered to full size before use. Under MAGIC's double backward those
gathered copies cannot be freed -- the second-order pass differentiates through
the forward values again -- so the whole model stays resident on every rank.
Gathering inside the op does not help; the weight has to never be full-size.

Set ``BERGSON_TP=<width>`` (with ``fsdp: false``) to shard computation instead.
The world becomes a ``dp x tp`` mesh: ranks in a tp group jointly hold one
model and process the same rows; data-parallel reductions run over the dp dim.

* ``ColumnParallelLinear`` holds its own rows of the weight, computes
  ``x @ W_local.T`` and all-gathers the *output*.
* MLP up-projections leave their output un-gathered and the down-projection
  is a ``RowParallelLinear`` (own columns, all-reduced output), so the wide
  MLP intermediate stays at 1/tp size in the forward and both backward graphs.
* Llama-style attention is head-sharded the same way (q/k/v un-gathered,
  ``o_proj`` row-parallel), which shrinks the per-head ``seq x seq`` tensors.
* With ``BERGSON_TP_NARROW=1`` frozen weights the checkpoint stores narrower
  than the run's precision (bf16/fp16) are kept as stored and upcast per use
  through ``_FrozenLinear``.

Every collective is an ``autograd.Function`` whose backward is its dual
(all-reduce <-> copy, all-gather <-> slice), so the graph is correct to any
order. A bare slice or a self-dual all-reduce gives matching first-order
gradients and wrong second-order ones.

Measured at 512 tokens, fp32, one sequence per double backward on 95GiB cards:
Llama-3.1-70B went from OOM at tp=32 (91.5GiB) to 77.8GiB at tp=4, with scores
r = 0.99999986 between tp=4 and tp=8; OLMo-3 32B runs at tp=2 (84.7GiB).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor


class _AllReduceSum(torch.autograd.Function):
    """All-reduce (sum) forward, identity backward. Dual of _CopyToRanks.

    This is the Megatron pairing and it is the correct one; the alternative was
    measured. Making this self-dual instead (all-reduce on the way back too)
    broke the second-order VJP toward an upstream scalar outright -- relative
    error 4.03 against an exact 0.00 -- and inflated the replicated-parameter
    VJP by 29.7x. The two halves map between different kinds of quantity, a
    replicated one and a per-rank partial one, so neither is its own adjoint.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        y = x.clone().contiguous()
        dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _CopyToRanks.apply(grad_out, ctx.group), None


class _CopyToRanks(torch.autograd.Function):
    """Identity forward, all-reduce (sum) backward. Dual of _AllReduceSum.

    A column-parallel linear splits the output, so each rank's backward gives
    only its own term of ``dL/dx = sum_o grad_out_o . W_o``. Without this the
    input gradient is short by the cross-rank sum: measured, the loss and the
    peak were right while the second-order term came out 9.4e-16 against a true
    5.95e+04, i.e. a memory win on a broken graph.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _AllReduceSum.apply(grad_out, ctx.group), None


class _AllGatherLastDim(torch.autograd.Function):
    """All-gather along the last dim. Dual of _SliceLocal."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, group):
        ctx.group = group
        ctx.width = local.shape[-1]
        parts = [
            torch.empty_like(local) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(parts, local.contiguous(), group=group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _SliceLocal.apply(grad_out, ctx.width, ctx.group), None


class _SliceLocal(torch.autograd.Function):
    """This rank's slice along the last dim. Dual of _AllGatherLastDim.

    Needed as a Function rather than a plain slice: differentiating a bare
    slice again yields a local zero-pad, where the correct adjoint gathers the
    other ranks' contributions. Measured with the bare slice, first order
    matched to 7 digits while the second-order term was 1.437234e+05 against a
    true 5.953800e+04.
    """

    @staticmethod
    def forward(ctx, full: torch.Tensor, width: int, group):
        ctx.group = group
        lo = dist.get_rank(group=group) * width
        return full[..., lo : lo + width].contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _AllGatherLastDim.apply(grad_out, ctx.group), None, None


class _FrozenLinear(torch.autograd.Function):
    """``x @ w.T`` for a frozen ``w`` stored narrower than ``x``.

    Llama-3.1-70B and OLMo-3 ship bf16 weights, so holding them in fp32 doubles
    the resident weight set for no extra information. A plain
    ``F.linear(x, w.float())`` does not help: autograd keeps the fp32 temporary
    alive for the backward, and under ``create_graph`` for the double backward
    too, so every upcast weight ends up resident anyway. Here only the narrow
    tensor is kept and each pass re-casts it. The map is linear in ``x`` and
    ``w`` is frozen, so its derivative is the transpose map and vice versa:
    the two Functions are each other's backward to any order.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        ctx.w = w
        return F.linear(x, w.to(x.dtype))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _FrozenLinearT.apply(grad_out, ctx.w), None


class _FrozenLinearT(torch.autograd.Function):
    """``g @ w``: the transpose of _FrozenLinear, and its dual."""

    @staticmethod
    def forward(ctx, g: torch.Tensor, w: torch.Tensor):
        ctx.w = w
        return g @ w.to(g.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _FrozenLinear.apply(grad_out, ctx.w), None


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """``F.linear``, through _FrozenLinear when ``w`` is frozen and narrower."""
    if w.dtype == x.dtype:
        return F.linear(x, w, b)
    assert not w.requires_grad, "narrow storage is only for frozen weights"
    out = _FrozenLinear.apply(x, w)
    return out if b is None else out + b


def column_parallel_linear(
    x: torch.Tensor,
    w_local: torch.Tensor,
    bias_local: torch.Tensor | None,
    group,
) -> torch.Tensor:
    """``F.linear`` whose weight stays sharded along dim 0 across ``group``.

    ``w_local`` is this rank's rows of the weight and ``bias_local`` the
    matching slice of the bias, so the full output is the concatenation of the
    per-rank outputs along the feature dim.
    """
    local_out = _linear(_CopyToRanks.apply(x, group), w_local, bias_local)
    return _AllGatherLastDim.apply(local_out, group)


class ColumnParallelLinear(nn.Linear):
    """``nn.Linear`` holding only its own rows of the weight.

    Installed by class assignment, which is sound only where the forward being
    replaced is exactly ``F.linear(x, self.weight, self.bias)``; callers must
    check ``type(mod) is nn.Linear`` before installing.
    """

    _tp_group = None
    _tp_gather = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The weight is a DTensor sharded over the tp dim (so that Distributed
        # Checkpoint stores it per-rank rather than assuming replication); the
        # matmul wants this rank's rows as a plain tensor. Default
        # grad_placements mirror the DTensor's own, which is what we want: each
        # rank's gradient really is the gradient of its own rows.
        w = self.weight
        b = self.bias
        if isinstance(w, DTensor):
            w = w.to_local()
        if b is not None and isinstance(b, DTensor):
            b = b.to_local()
        if not self._tp_gather:
            # Megatron MLP: the output stays this rank's slice of the feature
            # dim, to be consumed by a RowParallelLinear.
            return _linear(_CopyToRanks.apply(x, self._tp_group), w, b)
        return column_parallel_linear(x, w, b, self._tp_group)


class RowParallelLinear(nn.Linear):
    """``nn.Linear`` holding only its own columns of the weight.

    The input is this rank's slice of the feature dim (the ungathered output of
    a ColumnParallelLinear), so each rank computes one term of the sum over
    input features and the output is their all-reduce. Together the pair keeps
    the MLP intermediate -- 3.5x the residual width at Llama-70B -- at 1/tp
    size on every rank, in the forward and in both backward graphs. The bias
    is replicated and added once, after the reduce.
    """

    _tp_group = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if isinstance(w, DTensor):
            w = w.to_local()
        out = _AllReduceSum.apply(_linear(x, w, None), self._tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


def tp_group_of(mesh: DeviceMesh, dim_name: str = "tp"):
    """Process group for the tensor-parallel dimension of a 2D mesh."""
    return mesh[dim_name].get_group()


# ---------------------------------------------------------------------------
# Which ranks are data-parallel peers
#
# With simple_fsdp there is one flat mesh and every rank is a data-parallel
# peer, so averaging a gradient over the whole world is right. Under tensor
# parallelism it is not: ranks in the same tp group hold different slices of
# each weight and must not average with each other. Four places care, and all
# four have to agree or the scores come out wrong without any error:
#
#   Trainer.apply_update   averages the parameter gradient across peers
#   Trainer.backward       divides a document's weight-grad by the peer count
#   microbatch_step_vjp    the same division on the micro-batched path
#
# ``None`` is torch's own spelling of "the default group", so while no tp mesh
# is configured these accessors return exactly what the call sites used before
# and behaviour is unchanged.
# ---------------------------------------------------------------------------

_DP_GROUP = None


def set_data_parallel_group(group) -> None:
    """Restrict data-parallel reductions to ``group``. ``None`` restores all ranks."""
    global _DP_GROUP
    _DP_GROUP = group


def data_parallel_group():
    return _DP_GROUP


def data_parallel_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_DP_GROUP)


_MESH = None


def parallel_mesh():
    """The dp x tp mesh, built once and cached; ``None`` when BERGSON_TP is unset.

    Building it also registers the data-parallel group, and that has to happen
    before anything reads ``data_parallel_group()``. DataStream does, when it
    decides how to split rows across peers, and it is constructed well before
    the trainer: setting the group inside prepare_trainer left the stream
    splitting rows across the whole world, so each rank in a tp group received
    a DIFFERENT sequence while the group jointly held one model. The
    all-gathered outputs then mixed unrelated rows.

    Measured with the group set too late, pythia-160m at tp=4 on 8 nodes:
    ids_shape (1, 1024) per rank instead of (4, 1024), step-0 loss
    7.9956903458 against 2.7764315605, and final scores uncorrelated with the
    reference (r = -0.0002625186).
    """
    global _MESH
    if _MESH is not None:
        return _MESH

    tp = int(os.environ.get("BERGSON_TP", "1"))
    if tp <= 1 or not dist.is_initialized():
        return None

    world = dist.get_world_size()
    if world % tp:
        raise ValueError(
            f"BERGSON_TP={tp} does not divide world size {world}; the dp x tp "
            "mesh would be ragged."
        )
    _MESH = init_device_mesh("cuda", (world // tp, tp), mesh_dim_names=("dp", "tp"))
    set_data_parallel_group(_MESH["dp"].get_group())
    if dist.get_rank() == 0:
        print(
            f"[tensor_parallel] mesh dp={world // tp} x tp={tp}; "
            "data-parallel reductions and row splitting restricted to dp",
            flush=True,
        )
    return _MESH


def tensor_parallel_active() -> bool:
    """Is a tp mesh configured?

    Weights under tensor parallelism are DTensors, so their gradients are too,
    and the places that hand-roll a data-parallel reduction have to step aside
    and let DTensor placements do it -- exactly as they already do for fsdp.
    Calling ``dist.all_reduce`` on a DTensor fails with "found no DeviceMesh
    from dtensor args for c10d::allreduce_".
    """
    return _MESH is not None
