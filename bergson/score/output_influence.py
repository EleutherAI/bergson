import math

import torch
from peft import PeftModel
from torch import Tensor
from torch.func import functional_call, jvp
from transformers import PreTrainedModel

from bergson.collector.collector import (
    HookCollectorBase,
    create_module_projection_matrix,
    global_projection_blocks,
    token_losses,
)
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.gradients import GradientProcessor, LayerAdapter


def check_output_influence_supported(
    index_cfg: IndexConfig, preprocess_cfg: PreprocessConfig
):
    """Raise if the run needs something output influence scoring can't reproduce."""
    unsupported = {
        "reshape_to_square": index_cfg.reshape_to_square,
        "unit_normalize": preprocess_cfg.unit_normalize,
        "loss_fn='kl'": index_cfg.loss_fn == "kl",
        "optimizer_state": bool(index_cfg.optimizer_state),
        "split_attention_modules": bool(index_cfg.split_attention_modules),
        "fsdp": index_cfg.fsdp,
        "moe_experts": bool(index_cfg.moe_experts),
        f"precision={index_cfg.precision!r}": index_cfg.precision in ("int4", "int8"),
    }
    found = [name for name, bad in unsupported.items() if bad]
    if found:
        raise ValueError(
            f"token_influence='output' doesn't support {', '.join(found)}."
        )


def query_directions(
    model: PreTrainedModel | PeftModel,
    query_grads_t: dict[str, Tensor],
    target_info: dict[str, tuple[torch.device, torch.Size, bool]],
) -> list[tuple[Tensor, str, str | None, bool]]:
    """Match each scored module's query block to the parameters it moves.

    ``query_grads_t`` maps a module name under ``model.base_model`` to its
    ``[grad_dim, num_queries]`` query block, laid out ``[out, in]`` with the
    bias as a trailing column when the module's bias is collected. Only
    modules in both the query and ``target_info`` are scored, as with gradients.
    Returns ``(block, weight name, bias name, transposed)`` per module, where
    ``transposed`` marks layers that store their weight ``[in, out]``.
    """
    param_names = {id(p): name for name, p in model.named_parameters()}
    directions = []
    seen = set()
    for name, block in query_grads_t.items():
        if name not in target_info:
            continue
        module = model.base_model.get_submodule(name)
        if not isinstance(module, LayerAdapter.supported_modules):
            raise ValueError(
                f"token_influence='output' can't score {name}: "
                f"{type(module).__name__} is not a supported layer type."
            )
        out_dim = getattr(module, LayerAdapter.out_attr(module))
        in_dim = getattr(module, LayerAdapter.in_attr(module))
        if module.weight.numel() != out_dim * in_dim:
            raise ValueError(
                f"token_influence='output' can't score {name}: its weight has "
                f"{module.weight.numel()} entries but the gradient block covers "
                f"{out_dim * in_dim}, so no direction maps onto the weight."
            )
        has_bias = target_info[name][2]
        cols = in_dim + int(has_bias)
        if block.shape[0] != out_dim * cols:
            raise ValueError(
                f"The query for {name} has {block.shape[0]} entries, but the "
                f"module has {out_dim * cols} parameters."
            )

        weight_name = param_names[id(module.weight)]
        bias_name = param_names[id(module.bias)] if has_bias else None
        # Moving a shared weight would move every module that uses it.
        for param in (weight_name, bias_name):
            if param in seen:
                raise ValueError(
                    f"token_influence='output' can't score shared parameter {param}."
                )
            if param is not None:
                seen.add(param)
        directions.append(
            (block, weight_name, bias_name, LayerAdapter.weight_transposed(module))
        )
    if not directions:
        raise ValueError("The query has no modules in common with the scored ones.")
    return directions


def unproject_query(
    model: PreTrainedModel | PeftModel,
    query_grads_t: dict[str, Tensor],
    target_info: dict[str, tuple[torch.device, torch.Size, bool]],
    processor: GradientProcessor,
) -> dict[str, Tensor]:
    """Map projected query blocks to each scored module's ``[out, in]`` layout.

    A module's gradient ``G`` projects to ``L G R^T``, or to ``R vec(G)`` under
    the global target, so its dot product with a projected query ``Y`` is its dot
    product with ``L^T Y R``, or with ``R^T Y`` at the same scale.
    """
    p = processor.projection_dim
    assert p is not None
    dims = {}
    for name, (_, _, has_bias) in target_info.items():
        module = model.base_model.get_submodule(name)
        out_dim = getattr(module, LayerAdapter.out_attr(module))
        dims[name] = (out_dim, getattr(module, LayerAdapter.in_attr(module)) + has_bias)

    full = {}
    if processor.projection_target == "global":
        y = query_grads_t["gradients"]  # [p, num_queries]
        for name, (out_dim, cols) in dims.items():
            args = (
                HookCollectorBase.projection_identifier(
                    name, "single", processor.projection_seed
                ),
                p,
                out_dim * cols,
                y.dtype,
                y.device,
                processor.projection_type,
            )
            if processor.projection_scale == "row_norm":
                row_sq = y.new_zeros(p)
                for _, _, block in global_projection_blocks(*args):
                    row_sq.add_(block.pow(2).sum(dim=1))
                scaled = y / row_sq.sqrt()[:, None]
            else:
                scaled = y / math.sqrt(p)
            full[name] = y.new_empty(out_dim * cols, y.shape[1])
            for start, stop, block in global_projection_blocks(*args):
                full[name][start:stop] = block.T @ scaled
        return full

    for name, block in query_grads_t.items():
        if name not in dims:
            continue
        out_dim, cols = dims[name]
        left, right = (
            create_module_projection_matrix(
                name,
                role,
                p,
                size,
                block.dtype,
                block.device,
                processor.projection_type,
                processor.projection_scale,
                processor.projection_seed,
            )
            for role, size in (("left", out_dim), ("right", cols))
        )
        y = block.T.reshape(-1, p, p)
        full[name] = torch.einsum("po,npq,qi->oin", left, y, right).flatten(0, 1)
    return full


def query_direction(
    directions: list[tuple[Tensor, str, str | None, bool]],
    params: dict[str, Tensor],
    q: int,
) -> dict[str, Tensor]:
    """Reshape query column ``q`` into a direction for each parameter it moves."""
    direction = {}
    for block, weight_name, bias_name, transposed in directions:
        weight = params[weight_name]
        out_dim = weight.shape[1] if transposed else weight.shape[0]
        in_dim = weight.numel() // out_dim
        v = block[:, q].reshape(out_dim, -1).to(weight.dtype)
        w = v[:, :in_dim]
        direction[weight_name] = (w.T if transposed else w).reshape(weight.shape)
        if bias_name is not None:
            direction[bias_name] = v[:, in_dim]
    return direction


def output_token_influence(
    model: PreTrainedModel | PeftModel,
    x: Tensor,
    y: Tensor,
    directions: list[tuple[Tensor, str, str | None, bool]],
    cfg: IndexConfig,
    advantage: list[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return how fast each loss term changes as the weights move along each
    query, ``[N, S - 1, num_queries]``, and each document's loss, ``[N]``.

    Entry ``t`` belongs to the loss on token ``t + 1``, weighted as
    ``fwd_bwd_factory`` weights it, so the entries of a document sum to its
    gradient's dot product with the query, and its loss is the one
    ``fwd_bwd_factory`` returns. ``torch.func.jvp`` computes the rates in one
    forward pass per query.
    """
    masks = y[:, 1:] != -100
    weights = torch.ones_like(masks, dtype=torch.float32)
    if cfg.loss_reduction == "mean":
        weights = weights / masks.sum(dim=1, keepdim=True).clamp_min(1)
    if advantage is not None:
        weights = weights * torch.tensor(advantage, device=y.device).unsqueeze(1)

    params = {name: p.detach() for name, p in model.named_parameters()}
    num_queries = directions[0][0].shape[1]

    def losses(moved: dict[str, Tensor]) -> Tensor:
        logits = functional_call(model, moved, (x,)).logits[:, :-1].float()
        return token_losses(cfg.loss_fn, logits, y[:, 1:], cfg.label_smoothing)

    scores = []
    doc_losses = weights.new_zeros(len(weights))
    with torch.no_grad():
        for q in range(num_queries):
            direction = query_direction(directions, params, q)
            moved = {name: params[name] for name in direction}
            result = jvp(losses, (moved,), (direction,))
            scores.append(result[1].float() * weights)
            # The losses themselves are the same for every query column.
            doc_losses = (result[0].float() * weights).sum(dim=1)
    return torch.stack(scores, dim=-1), doc_losses
