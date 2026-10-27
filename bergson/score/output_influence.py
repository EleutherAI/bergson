# MIT: output-mode JVP backport from EleutherAI/bergson, with local DPO/FP64 extensions.
import torch
from peft import PeftModel
from torch import Tensor
from torch.func import functional_call, jvp
from transformers import PreTrainedModel

from torch.nn import functional as F
from transformers.pytorch_utils import Conv1D
from bergson.config import IndexConfig, PreprocessConfig
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
        "moe_experts": bool(getattr(index_cfg, "moe_experts", "")),
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
            (block, weight_name, bias_name, isinstance(module, Conv1D))
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
        if block.dtype != weight.dtype:
            raise ValueError("Output JVP requires matching model/query dtypes; promote the model to preserve FP64 queries")
        v = block[:, q].reshape(out_dim, -1).to(weight.device)
        w = v[:, :in_dim]
        direction[weight_name] = (w.T if transposed else w).reshape(weight.shape)
        if bias_name is not None:
            direction[bias_name] = v[:, in_dim]
    return direction


def output_token_influence(model, x, y, directions, cfg, advantage=None, *, reference_logps=None, sequence_lengths=None):
    """Per-output-loss JVPs [sequence, shifted-token, query], not position hooks.

    DPO rows are chosen sequences followed by their paired rejected sequences.
    Their signed token rates sum to the normalized pair-gradient dot product.
    All prompt/padding terms are zero. FP64 inputs remain FP64 throughout.
    """
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Output attribution requires FP32 minimum')
    if cfg.loss_fn not in ('ce', 'dpo'):
        raise ValueError('Output backport supports CE and paired DPO')
    if cfg.loss_fn == 'dpo' and cfg.label_smoothing:
        raise ValueError('DPO does not use label smoothing')
    masks = y[:, 1:] != -100
    weights = masks.to(dtype)
    if cfg.loss_reduction == 'mean':
        if cfg.loss_fn == 'dpo' and not masks.any(dim=1).all():
            raise ValueError('DPO sequences require supervised response tokens')
        weights = weights / masks.sum(dim=1, keepdim=True).clamp_min(1)
    elif cfg.loss_reduction != 'sum':
        raise ValueError('Expected mean or sum reduction')
    params = {name: parameter.detach() for name, parameter in model.named_parameters()}
    columns = directions[0][0].shape[1]

    def evaluate(moved, inputs, labels):
        token = TOKEN_LOSS_REQUEST.set((labels, cfg.label_smoothing))
        try:
            if getattr(model, '_bergson_output_loss_head', False):
                terms = functional_call(model, moved, (inputs,), {'use_cache': False}).logits
            else:
                logits = functional_call(model, moved, (inputs,)).logits[:, :-1]
                if logits.dtype != dtype:
                    raise ValueError('Model changed scoring precision')
                terms = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels[:, 1:].to(logits.device).reshape(-1),
                    reduction='none', label_smoothing=cfg.label_smoothing,
                ).reshape(labels.shape[0], labels.shape[1]-1)
        finally:
            TOKEN_LOSS_REQUEST.reset(token)
        if terms.dtype != dtype:
            raise ValueError('Token losses changed scoring precision')
        return terms

    def losses(moved):
        if sequence_lengths is None:
            terms = evaluate(moved, x, y)
        else:
            if len(sequence_lengths) != len(x):
                raise ValueError('Sequence lengths must cover every row')
            rows = []
            for row, length in enumerate(sequence_lengths):
                if length < 2 or length > x.shape[1]:
                    raise ValueError('Invalid sequence length')
                value = evaluate(moved, x[row:row+1, :length], y[row:row+1, :length])
                rows.append(F.pad(value, (0, x.shape[1]-length)))
            terms = torch.cat(rows, dim=0)
        return terms * weights.to(terms.device)

    scores = []
    document_losses = None
    with torch.no_grad():
        for column in range(columns):
            direction = query_direction(directions, params, column)
            moved = {name: params[name] for name in direction}
            terms, rates = jvp(losses, (moved,), (direction,))
            if cfg.loss_fn == 'dpo':
                if len(x) % 2 or reference_logps is None:
                    raise ValueError('DPO output attribution requires full pairs and frozen references')
                pairs = len(x)//2
                refs = torch.as_tensor(reference_logps, device=terms.device, dtype=dtype)
                if refs.shape != (len(x),):
                    raise ValueError('References must match chosen/rejected order')
                logps = -terms.sum(dim=1)
                margin = cfg.dpo_beta*((logps[:pairs]-refs[:pairs])-(logps[pairs:]-refs[pairs:]))
                factor = cfg.dpo_beta*torch.sigmoid(-margin)
                document_losses = F.softplus(-margin)
                if advantage is not None:
                    pair_weights = torch.as_tensor(advantage, device=terms.device, dtype=dtype)
                    if pair_weights.shape == (len(x),):
                        if not torch.equal(pair_weights[:pairs], pair_weights[pairs:]):
                            raise ValueError('Chosen/rejected weights must match')
                        pair_weights = pair_weights[:pairs]
                    if pair_weights.shape != (pairs,):
                        raise ValueError('DPO weights must match pair count')
                    factor = factor*pair_weights
                    document_losses = document_losses*pair_weights
                rates = rates*torch.cat((factor, -factor))[:, None]
            else:
                document_losses = terms.sum(dim=1)
                if advantage is not None:
                    example_weights = torch.as_tensor(advantage, device=terms.device, dtype=dtype)
                    rates = rates*example_weights[:, None]
                    document_losses = document_losses*example_weights
            scores.append(rates)
    return torch.stack(scores, dim=-1), document_losses


