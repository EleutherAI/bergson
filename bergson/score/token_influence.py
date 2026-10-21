from collections.abc import Iterator

import torch
from peft import PeftModel
from torch import Tensor
from torch.func import functional_call, grad, jvp
from transformers import PreTrainedModel

from bergson.collector.collector import token_losses
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.gradients import LayerAdapter


def check_token_influence_supported(
    index_cfg: IndexConfig, preprocess_cfg: PreprocessConfig, token_influence: str
):
    """Raise if the run needs something forward-mode scoring can't reproduce."""
    unsupported = {
        "projection_dim != 0": index_cfg.projection_dim != 0,
        "unit_normalize": preprocess_cfg.unit_normalize,
        "loss_fn='kl'": index_cfg.loss_fn == "kl",
        "optimizer_state": bool(index_cfg.optimizer_state),
        "split_attention_modules": bool(index_cfg.split_attention_modules),
        "fsdp": index_cfg.fsdp,
        "moe_experts": bool(index_cfg.moe_experts),
        f"precision={index_cfg.precision!r}": index_cfg.precision in ("int4", "int8"),
        # Input token rows don't sum to the per-example score.
        "attribute_tokens=False": (
            token_influence == "input" and not index_cfg.attribute_tokens
        ),
    }
    found = [name for name, bad in unsupported.items() if bad]
    if found:
        raise ValueError(
            f"token_influence={token_influence!r} doesn't support "
            f"{', '.join(found)}."
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
                f"token_influence can't score {name}: "
                f"{type(module).__name__} is not a supported layer type."
            )
        out_dim = getattr(module, LayerAdapter.out_attr(module))
        in_dim = getattr(module, LayerAdapter.in_attr(module))
        if module.weight.numel() != out_dim * in_dim:
            raise ValueError(
                f"token_influence can't score {name}: its weight has "
                f"{module.weight.numel()} entries but the gradient block covers "
                f"{out_dim * in_dim}, so no direction maps onto the weight."
            )
        has_bias = target_info[name][2]
        cols = in_dim + int(has_bias)
        if block.shape[0] != out_dim * cols:
            raise ValueError(
                f"The query for {name} has {block.shape[0]} entries, but the "
                f"module has {out_dim * cols} parameters. Forward-mode "
                "token_influence needs an unprojected query."
            )

        weight_name = param_names[id(module.weight)]
        bias_name = param_names[id(module.bias)] if has_bias else None
        # Moving a shared weight would move every module that uses it.
        for param in (weight_name, bias_name):
            if param in seen:
                raise ValueError(
                    f"token_influence can't score shared parameter {param}."
                )
            if param is not None:
                seen.add(param)
        directions.append(
            (block, weight_name, bias_name, LayerAdapter.weight_transposed(module))
        )
    if not directions:
        raise ValueError("The query has no modules in common with the scored ones.")
    return directions


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


def query_moves(
    model: PreTrainedModel | PeftModel,
    directions: list[tuple[Tensor, str, str | None, bool]],
) -> Iterator[tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Yield the parameters each query column moves and its direction for them."""
    params = {name: p.detach() for name, p in model.named_parameters()}
    for q in range(directions[0][0].shape[1]):
        direction = query_direction(directions, params, q)
        yield {name: params[name] for name in direction}, direction


def weighted_token_losses(
    model: PreTrainedModel | PeftModel,
    moved: dict[str, Tensor],
    inputs: dict[str, Tensor],
    y: Tensor,
    cfg: IndexConfig,
    advantage: list[float] | None,
) -> Tensor:
    """Each loss term ``[N, S - 1]``, weighted as ``fwd_bwd_factory`` weights it,
    with ``moved`` in place of those parameters."""
    masks = y[:, 1:] != -100
    weights = torch.ones_like(masks, dtype=torch.float32)
    if cfg.loss_reduction == "mean":
        weights = weights / masks.sum(dim=1, keepdim=True).clamp_min(1)
    if advantage is not None:
        weights = weights * torch.tensor(advantage, device=y.device).unsqueeze(1)

    logits = functional_call(model, moved, (), inputs).logits[:, :-1].float()
    losses = token_losses(cfg.loss_fn, logits, y[:, 1:], cfg.label_smoothing)
    return losses.float() * weights


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

    def losses(moved: dict[str, Tensor]) -> Tensor:
        return weighted_token_losses(model, moved, {"input_ids": x}, y, cfg, advantage)

    scores = []
    doc_losses = torch.zeros(len(x), device=x.device)
    with torch.no_grad():
        for moved, direction in query_moves(model, directions):
            result = jvp(losses, (moved,), (direction,))
            scores.append(result[1])
            # The losses themselves are the same for every query column.
            doc_losses = result[0].sum(dim=1)
    return torch.stack(scores, dim=-1), doc_losses


def input_token_influence(
    model: PreTrainedModel | PeftModel,
    x: Tensor,
    y: Tensor,
    directions: list[tuple[Tensor, str, str | None, bool]],
    cfg: IndexConfig,
    advantage: list[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return how fast each document's gradient dot product with each query
    changes as each input token's embedding is scaled up, ``[N, S,
    num_queries]``, and each document's loss, ``[N]``.

    Entry ``t`` is the derivative of the query's rate along the weights with
    respect to scaling position ``t``'s embedding by ``1 + eps``, the input
    token influence of Grosse et al. (2023, Eq. 38), computed exactly with one
    backward pass and one forward-mode pass per query instead of their finite
    differences.
    """
    with torch.no_grad():
        embeds = model.get_input_embeddings()(x)
    # In the model's dtype, every row would be rounded to it.
    scale = torch.zeros(x.shape, device=x.device)

    def loss(scale: Tensor, moved: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        scaled = (embeds * (1 + scale).unsqueeze(-1)).to(embeds.dtype)
        losses = weighted_token_losses(
            model, moved, {"inputs_embeds": scaled}, y, cfg, advantage
        )
        return losses.sum(), losses.sum(dim=1).detach()

    scores = []
    doc_losses = torch.zeros(len(x), device=x.device)
    # Differentiating the forward-mode pass instead is wrong through
    # F.layer_norm. torch.func.grad differentiates inside no_grad, which keeps
    # the scores off the autograd graph of the unmoved parameters.
    with torch.no_grad():
        for moved, direction in query_moves(model, directions):
            result = jvp(
                lambda m: grad(loss, has_aux=True)(scale, m), (moved,), (direction,)
            )
            # Each side is the scale gradient and the per-document losses.
            doc_losses = result[0][1]
            scores.append(result[1][0])
    return torch.stack(scores, dim=-1), doc_losses
