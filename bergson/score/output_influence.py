import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from torch.func import functional_call, jvp
from transformers import PreTrainedModel

from bergson.collector.collector import streams_token_losses, token_losses
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.dpo import dpo_margin
from bergson.gradients import LayerAdapter
from bergson.score.output_model import TOKEN_LOSS_REQUEST


def check_output_influence_supported(
    index_cfg: IndexConfig, preprocess_cfg: PreprocessConfig
):
    """Raise if the run needs something output influence scoring can't reproduce."""
    unsupported = {
        "projection_dim != 0": index_cfg.projection_dim != 0,
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
            f"token_influence='output' doesn't support {', '.join(found)}. Set "
            "score_cfg.token_influence='gradient' to score per-token gradients."
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
                f"module has {out_dim * cols} parameters. token_influence='output' "
                "needs an unprojected query."
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
    *,
    reference_logps: list[float] | None = None,
    lengths: list[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return how fast each loss term changes as the weights move along each
    query, ``[N, S - 1, num_queries]``, and each document's loss, ``[N]``.

    Entry ``t`` belongs to the loss on token ``t + 1``, weighted as
    ``fwd_bwd_factory`` weights it, so the entries of a document sum to its
    gradient's dot product with the query, and its loss is the one
    ``fwd_bwd_factory`` returns. ``torch.func.jvp`` computes the rates in one
    forward pass per query.

    With ``loss_fn='dpo'`` the rows are each pair's chosen row followed by its
    rejected row, as ``fwd_bwd_factory`` takes them, with ``reference_logps``
    for every row. A pair's token entries sum to its gradient's dot product with
    the query, and its loss is returned for both of its rows. ``lengths`` runs
    each row on its own unpadded tokens.
    """
    dpo = cfg.loss_fn == "dpo"
    masks = y[:, 1:] != -100
    weights = torch.ones_like(masks, dtype=torch.float32)
    if cfg.loss_reduction == "mean":
        weights = weights / masks.sum(dim=1, keepdim=True).clamp_min(1)
    if advantage is not None and not dpo:
        weights = weights * torch.tensor(advantage, device=y.device).unsqueeze(1)
    if dpo:
        if len(x) % 2 or reference_logps is None:
            raise ValueError("DPO needs whole pairs and their reference_logps")
        references = torch.tensor(reference_logps, device=y.device, dtype=torch.float32)

    params = {name: p.detach() for name, p in model.named_parameters()}
    num_queries = directions[0][0].shape[1]
    streamed = streams_token_losses(model, "dpo" if dpo else cfg.loss_fn)

    def row_losses(moved: dict[str, Tensor], inputs: Tensor, labels: Tensor) -> Tensor:
        if streamed:
            request = TOKEN_LOSS_REQUEST.set((labels, cfg.label_smoothing))
            try:
                return functional_call(model, moved, (inputs,)).logits.float()
            finally:
                TOKEN_LOSS_REQUEST.reset(request)
        logits = functional_call(model, moved, (inputs,)).logits[:, :-1].float()
        loss_fn = "ce" if dpo else cfg.loss_fn
        return token_losses(loss_fn, logits, labels[:, 1:], cfg.label_smoothing)

    def losses(moved: dict[str, Tensor]) -> Tensor:
        if lengths is None:
            return row_losses(moved, x, y)
        rows = [
            F.pad(
                row_losses(moved, x[i : i + 1, :n], y[i : i + 1, :n]),
                (0, x.shape[1] - n),
            )
            for i, n in enumerate(lengths)
        ]
        return torch.cat(rows)

    scores = []
    doc_losses = weights.new_zeros(len(weights))
    with torch.no_grad():
        for q in range(num_queries):
            direction = query_direction(directions, params, q)
            moved = {name: params[name] for name in direction}
            terms, rates = jvp(losses, (moved,), (direction,))
            rates = rates.float() * weights
            if dpo:
                # The pair loss -log sigmoid(margin) changes by -sigmoid(-margin)
                # times the margin, which moves by beta times the chosen log
                # probability's change minus the rejected one's
                log_probs = -(terms.float() * weights).sum(dim=1)
                margin = dpo_margin(log_probs, references, cfg.dpo_beta)
                factor = cfg.dpo_beta * torch.sigmoid(-margin)
                rates = rates * torch.cat((factor, -factor))[:, None]
                doc_losses = F.softplus(-margin).repeat(2)
            else:
                # The losses themselves are the same for every query column.
                doc_losses = (terms.float() * weights).sum(dim=1)
            scores.append(rates)
    return torch.stack(scores, dim=-1), doc_losses
