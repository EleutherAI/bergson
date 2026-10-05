import os
from collections import defaultdict
from typing import TypeVar

import torch
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from .shard_load import ShardReader, materialize_buffers
from .tensor_parallel import ColumnParallelLinear, RowParallelLinear


def fsdp_policy():
    def _fsdp_recomp_policy():
        def _custom_policy(ctx, func, *args, **kwargs):
            to_recompute = func in {
                torch.ops._c10d_functional.all_gather_into_tensor.default,  # type: ignore[attr-defined]
                torch.ops._c10d_functional.wait_tensor.default,  # type: ignore[attr-defined]
            }
            return (
                CheckpointPolicy.MUST_RECOMPUTE
                if to_recompute
                else CheckpointPolicy.MUST_SAVE
            )

        return _custom_policy

    return create_selective_checkpoint_contexts(_fsdp_recomp_policy())


class ReplicateComputation(torch.nn.Module):
    def replicate_compute(self, x):
        return x.redistribute(
            placements=(Replicate(),),
        ).to_local(grad_placements=(Partial(reduce_op="avg"),))

    def forward(self, x):
        return checkpoint(
            self.replicate_compute, x, use_reentrant=False, context_fn=fsdp_policy
        )


_TP_PATCHED: dict[tuple, type] = {}


def _install_column_parallel(mod: torch.nn.Module, group, kind: str = "col") -> None:
    """Give ``mod`` a forward that computes against its own slice of the weight.

    Mixed into the module's current class rather than assigned outright, so it
    composes with a parametrization injected before or after. ``kind`` is
    ``col`` (rows sharded, output gathered), ``col_local`` (rows sharded,
    output left local) or ``row`` (columns sharded, output all-reduced).
    """
    cls = type(mod)
    base = RowParallelLinear if kind == "row" else ColumnParallelLinear
    patched = _TP_PATCHED.get((cls, base))
    if patched is None:
        patched = type(f"{base.__name__}{cls.__name__}", (base, cls), {})
        _TP_PATCHED[(cls, base)] = patched
    mod.__class__ = patched
    mod._tp_group = group
    if kind != "row":
        mod._tp_gather = kind == "col"


# MLP projections whose intermediate can stay sharded: the up-projections
# write the wide dim and the down-projection reads it, with only elementwise
# ops in between (Llama/OLMo: down(act(gate(x)) * up(x)); GPT-NeoX:
# dense_4h_to_h(act(dense_h_to_4h(x)))).
_MLP_UP = {"gate_proj", "up_proj", "dense_h_to_4h"}
_MLP_DOWN = {"down_proj", "dense_4h_to_h"}


def _shardable_attention(model: torch.nn.Module, world: int) -> set[str]:
    """Paths of attention modules whose heads can be split across the tp dim.

    Llama-style attention reshapes the q/k/v projections with ``view(..., -1,
    head_dim)`` and the output with ``reshape(..., -1)``, so given only a
    rank's slice of the projection rows it computes exactly that rank's heads
    with no code change; the per-head ``seq x seq`` tensors, which dominate the
    double-backward graph, then shrink by the tp width. Sound only when the
    slice is whole heads for q and for k/v (grouped-query attention pairs kv
    head ``j`` with q heads ``j*n_rep..``, which contiguous equal slices
    preserve), and when nothing between projection and reshape mixes heads:
    OLMo-3 normalises the full q and k vectors, so it is excluded.
    """
    if os.environ.get("BERGSON_TP_ATTN", "1") != "1":
        return set()
    out = set()
    for name, mod in model.named_modules():
        if not all(hasattr(mod, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        head_dim = getattr(mod, "head_dim", None)
        if head_dim is None or hasattr(mod, "q_norm") or hasattr(mod, "k_norm"):
            continue
        widths = [
            getattr(mod, a).weight.shape[0] for a in ("q_proj", "k_proj", "v_proj")
        ]
        if all(w % (head_dim * world) == 0 for w in widths):
            out.add(name)
    return out


_ATTN_IN = {"q_proj", "k_proj", "v_proj"}
_ATTN_OUT = {"o_proj"}


def _tp_kind(path: str, attention: set[str] = frozenset()) -> str:
    """How the Linear whose weight is at ``path`` is parallelised.

    A PEFT LoRA layer computes ``base(x) + B(A(x))``, so around a projection
    that writes a sharded dim the base and B both write it (``col_local``),
    and around one that reads it the base and A both read it (``row``). A's
    output is all-reduced, so B then sees a replicated input as usual.
    """
    parts = path.split(".")
    for i, part in enumerate(parts):
        if i == 0:
            continue
        rest = parts[i + 1 :]
        up = down = False
        if parts[i - 1] == "mlp" and os.environ.get("BERGSON_TP_MLP", "1") == "1":
            up, down = part in _MLP_UP, part in _MLP_DOWN
        elif ".".join(parts[:i]) in attention:
            up, down = part in _ATTN_IN, part in _ATTN_OUT
        if up:
            return "col" if "lora_A" in rest else "col_local"
        if down:
            return "col" if "lora_B" in rest else "row"
    return "col"


def _linear_weight_paths(model: torch.nn.Module) -> set[str]:
    """Paths of ``weight`` on modules that are exactly ``nn.Linear``.

    Decided before anything is registered: parametrizing a sibling bias injects
    a subclass and would make ``type(mod) is nn.Linear`` order-dependent. The
    check is strict rather than ``isinstance`` because the substitution is only
    sound where the forward replaced is exactly
    ``F.linear(x, self.weight, self.bias)``.
    """
    out = set()
    for path, _ in model.named_parameters(remove_duplicate=False):
        mod_name, _, p_name = path.rpartition(".")
        if p_name != "weight":
            continue
        if type(model.get_submodule(mod_name)) is torch.nn.Linear:
            out.add(path)
    return out


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


def tensor_parallel_model(
    model: ModuleT,
    tp_group,
    reader: ShardReader,
    device: torch.device | str,
    dp_tp_mesh=None,
) -> ModuleT:
    """Shard every plain ``nn.Linear`` along dim 0 across ``tp_group``.

    Separate from ``simple_fsdp`` rather than an option on it, because the two
    are opposite strategies: ``simple_fsdp`` shards storage and replicates
    computation, gathering each weight to full size before use, which is what
    makes MAGIC's double backward hold the whole model resident. This shards the
    computation, so a full weight is never built. Measured on 32 layers of
    8192x8192 fp32 over 4 ranks: peak 12.28GiB -> 4.28GiB, the saving equal to
    the entire weight set, with all orders of gradient matching.

    Only the linears move. Embeddings and norms stay replicated; at 70B the
    embedding is 4.2GiB fp32 against roughly 254GiB of linears, and ``lm_head``
    is itself an ``nn.Linear`` so it shards with the rest.

    ``model`` is built on the meta device; each rank reads only its own slice
    of every sharded weight from ``reader``.
    """
    world = torch.distributed.get_world_size(group=tp_group)

    targets = _linear_weight_paths(model)
    attention = _shardable_attention(model, world)
    if torch.distributed.get_rank() == 0:
        print(
            f"[tensor_parallel] sharding {len(targets)} Linear weights "
            f"{world} ways across the tp dim; {len(attention)} attention "
            "modules head-sharded",
            flush=True,
        )

    for path in sorted(targets):
        mod_name, _, _ = path.rpartition(".")
        mod = model.get_submodule(mod_name)

        kind = _tp_kind(path, attention)
        dim = 1 if kind == "row" else 0
        out_features = mod.weight.shape[dim]
        if out_features % world:
            if kind != "col":
                raise ValueError(
                    f"{path}: MLP width {out_features} is not divisible by the "
                    f"tensor-parallel width {world}"
                )
            # Uneven sharding would make the output all-gather produce a
            # wrongly ordered feature dim rather than fail, so a layer whose
            # width the tp dim does not divide stays replicated instead
            # (OLMo-3 32B: lm_head has 100278 rows, 2GiB fp32).
            if torch.distributed.get_rank() == 0:
                print(
                    f"[tensor_parallel] {path}: out_features {out_features} "
                    f"not divisible by {world}; left replicated",
                    flush=True,
                )
            continue
        for p_name in ("weight", "bias"):
            param = getattr(mod, p_name, None)
            if param is None:
                continue
            if p_name == "bias" and kind == "row":
                # Replicated, added once after the all-reduce; left for the
                # caller to load in full like any other replicated parameter.
                continue
            local = reader.local(f"{mod_name}.{p_name}", param, dim).to(device)

            if dp_tp_mesh is not None:
                # Register as a DTensor, replicated over dp and sharded over
                # tp, rather than as a plain local slice.
                #
                # This is not cosmetic. TrainerState.save goes through
                # dcp.async_save, and Distributed Checkpoint expresses sharded
                # state as DTensors: a plain tensor carrying the same key and
                # shape on every rank is taken to be REPLICATED, so DCP stores
                # one rank's copy and returns it to all of them. Measured on
                # pythia-70m at tp=2, the shards were correctly distinct at
                # construction (tp_rank 0 norm 2.3216206010, tp_rank 1 norm
                # 2.3079075565) and after one save/load every rank held
                # 2.3216206010 -- tp_rank 1's slice simply gone. Declaring the
                # sharding is what makes the checkpoint round-trip faithful.
                value = DTensor.from_local(
                    local,
                    dp_tp_mesh,
                    (Replicate(), Shard(dim)),
                    shape=param.shape,
                    stride=param.stride(),
                )
            else:
                value = local

            mod.register_parameter(
                p_name,
                torch.nn.Parameter(value, requires_grad=param.requires_grad),
            )

        _install_column_parallel(mod, tp_group, kind)

    return model


def simple_fsdp(
    model: ModuleT, reader: ShardReader, device: torch.device | str
) -> ModuleT:
    """SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile"""
    # For each unique parameter, construct a list of the places in the model where it
    # appears. This is a bit wonky, but it is the best way to handle tied weights.
    param_to_paths = defaultdict(list)
    for path, param in model.named_parameters(remove_duplicate=False):
        param_to_paths[param].append(path)

    # Use a while loop to avoid modifying the dict while iterating over it. We don't
    # want to hold onto both the original and distributed versions of each parameter.
    while param_to_paths:
        param, paths = param_to_paths.popitem()

        # Create a new distributed version of this param
        sharded = DTensor.from_local(
            reader.local(paths[0], param).to(device),
            placements=(Shard(0),),
            shape=param.shape,
            stride=param.stride(),
        )
        dist_param = torch.nn.Parameter(sharded, requires_grad=param.requires_grad)

        # Update all occurrences of this parameter in the model
        for path in paths:
            # Find the module that has a reference to this parameter
            mod_name, _, p_name = path.rpartition(".")
            mod = model.get_submodule(mod_name)

            # Re-register the parameter with sharding and replication
            mod.register_parameter(p_name, dist_param)
            register_parametrization(
                mod,
                p_name,
                ReplicateComputation(),
                unsafe=True,
            )

    return model


def shallow_copy(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create a shallow copy of a dict of tensors, handling tied weights.

    Preserves the original key order. All paths that shared the same tensor
    (tied weights) will point to the same copied tensor in the output.
    """
    seen: dict[int, torch.Tensor] = {}  # id(original) -> copied tensor
    result: dict[str, torch.Tensor] = {}

    for path, t in tensor_dict.items():
        tid = id(t)
        if tid not in seen:
            if isinstance(t, DTensor):
                t2 = DTensor.from_local(
                    t.to_local(),
                    t.device_mesh,
                    t.placements,
                    shape=t.shape,
                    stride=t.stride(),
                )
            else:
                t2 = torch.Tensor(t.data)
            t2.requires_grad_(t.requires_grad)
            seen[tid] = t2

        result[path] = seen[tid]

    return result


def tensor_parallel_from_checkpoint(
    model: ModuleT,
    model_path: str,
    tp_group,
    device,
    config,
    dp_tp_mesh=None,
) -> ModuleT:
    """Build a tensor-parallel model without ever holding a full copy.

    The model arrives on the meta device. Each ``nn.Linear`` is given only its
    own rows of the weight and bias, read straight off the safetensors
    checkpoint; everything still on meta afterwards -- embeddings, norms -- is
    replicated, so it is read in full. No rank materialises the whole model and
    no weight is ever gathered to full size at any point.
    """
    import torch.distributed as dist

    world = dist.get_world_size(group=tp_group)
    tp_rank = dist.get_rank(group=tp_group)
    reader = ShardReader(model_path, world, tp_rank)

    try:
        # Rotary tables and friends live in buffers, which meta leaves empty.
        materialize_buffers(model, config, device)

        model = tensor_parallel_model(model, tp_group, reader, device, dp_tp_mesh)

        replicated = 0
        for path, param in list(model.named_parameters(remove_duplicate=False)):
            if not param.is_meta:
                continue
            mod_name, _, p_name = path.rpartition(".")
            mod = model.get_submodule(mod_name)
            full = reader.full(path, param).to(device)
            mod.register_parameter(
                p_name,
                torch.nn.Parameter(full, requires_grad=param.requires_grad),
            )
            replicated += 1

        left = [
            n for n, p in model.named_parameters(remove_duplicate=False) if p.is_meta
        ]
        if left:
            raise RuntimeError(
                f"{len(left)} parameters still on meta after loading, "
                f"e.g. {left[:3]}. A forward would read empty storage."
            )
        if dist.get_rank() == 0:
            print(
                f"[tensor_parallel] loaded {replicated} replicated parameters "
                f"in full; linears sharded {world} ways",
                flush=True,
            )
    finally:
        reader.close()

    return model
