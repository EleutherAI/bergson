from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import PreTrainedModel

from bergson.gradients import AdafactorNormalizer, AdamNormalizer, Normalizer


def _extract_second_moments(
    state: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract second moment tensors from a single parameter's optimizer state.

    Supports three optimizer formats:
    - Adam (``exp_avg_sq``): full second moment matrix
    - Adafactor (``exp_avg_sq_row`` + ``exp_avg_sq_col``): factored row/col
    - 8-bit Adam (``__bnb_optimizer_quant_state__["state2"]``): quantized second moments

    Returns:
        ``(exp_avg_sq, exp_avg_sq_row, exp_avg_sq_col)`` — whichever fields are
        present; the rest are ``None``.
    """
    # Standard Adam
    if "exp_avg_sq" in state:
        return state["exp_avg_sq"], None, None

    # Native Adafactor
    if "exp_avg_sq_row" in state:
        return None, state["exp_avg_sq_row"], state.get("exp_avg_sq_col")

    # 8-bit Adam (bitsandbytes)
    bnb_state = state.get("__bnb_optimizer_quant_state__")
    if isinstance(bnb_state, dict) and "state2" in bnb_state:
        return bnb_state["state2"], None, None

    return None, None, None


def load_from_optimizer(
    model: PreTrainedModel | PeftModel,
    optimizer_state_path: str,
    include_bias: bool = False,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """Load optimizer second moments from a checkpoint and create normalizer
    instances for each target linear layer.

    Auto-detects the optimizer format:

    - Adam/AdamW: ``exp_avg_sq`` -> AdamNormalizer
    - Adafactor: ``exp_avg_sq_row``/``exp_avg_sq_col`` -> AdafactorNormalizer
    - 8-bit Adam (bnb): ``state2`` -> AdamNormalizer

    Args:
        model: The model whose parameter names are used to map optimizer
            state indices to layer names.
        optimizer_state_path: Path to either a checkpoint directory containing
            ``optimizer.pt`` or directly to an optimizer state file.
        include_bias: Whether to include bias second moments.
        target_modules: Optional set of module names to include. If ``None``,
            all linear layers are included.

    Returns:
        Dictionary mapping layer names to normalizer instances.
    """
    state_path = Path(optimizer_state_path)
    if state_path.is_dir():
        state_path = state_path / "optimizer.pt"

    optimizer_state = torch.load(state_path, map_location="cpu", weights_only=False)

    # The optimizer state is keyed by position in the trainable parameter list.
    # For PEFT checkpoints, only include PEFT params.
    if isinstance(model, PeftModel):
        st = get_peft_model_state_dict(model)
        params_for_index = list(st.items())
    else:
        params_for_index = list(model.named_parameters())

    target_param_index_to_name: dict[int, str] = {}
    for idx, (name, _param) in enumerate(params_for_index):
        target_param_index_to_name[idx] = name

    # Extract second moments per layer
    normalizers: dict[str, Normalizer] = {}
    for param_idx, state in optimizer_state["state"].items():
        param_idx = int(param_idx)
        if param_idx not in target_param_index_to_name:
            continue

        param_name = target_param_index_to_name[param_idx]

        if not param_name.endswith(".weight"):
            continue

        exp_avg_sq, row, col = _extract_second_moments(state)

        if row is not None and col is not None:
            # Native Adafactor checkpoint
            if row.ndim != 1:
                continue

            layer_name = param_name.removesuffix(".weight")
            module_name = layer_name.removeprefix("base_model.")

            if target_modules is not None and module_name not in target_modules:
                continue

            bias_exp_avg_sq = _get_bias_second_moment(
                layer_name, target_param_index_to_name, optimizer_state, include_bias
            )

            normalizers[module_name] = AdafactorNormalizer(
                row=row,
                col=col,
                bias_avg_sq=bias_exp_avg_sq,
            )
        elif exp_avg_sq is not None:
            # Adam or 8-bit Adam checkpoint
            if exp_avg_sq.ndim != 2:
                continue

            layer_name = param_name.removesuffix(".weight")
            module_name = layer_name.removeprefix("base_model.")

            if target_modules is not None and module_name not in target_modules:
                continue

            bias_exp_avg_sq = _get_bias_second_moment(
                layer_name, target_param_index_to_name, optimizer_state, include_bias
            )

            normalizers[module_name] = AdamNormalizer(
                weight_avg_sq=exp_avg_sq,
                bias_avg_sq=bias_exp_avg_sq,
            )

    assert normalizers, (
        f"No optimizer second moments found in '{optimizer_state_path}'. "
        "Ensure the checkpoint was saved from an Adam-family or Adafactor optimizer."
    )

    # Move normalizer tensors to the model's device
    device = next(model.parameters()).device
    for norm in normalizers.values():
        if isinstance(norm, AdamNormalizer):
            norm.weight_avg_sq = norm.weight_avg_sq.to(device)
            if norm.bias_avg_sq is not None:
                norm.bias_avg_sq = norm.bias_avg_sq.to(device)
        elif isinstance(norm, AdafactorNormalizer):
            norm.row = norm.row.to(device)
            norm.col = norm.col.to(device)
            if norm.bias_avg_sq is not None:
                norm.bias_avg_sq = norm.bias_avg_sq.to(device)

    # Report what we loaded
    types = {type(n).__name__ for n in normalizers.values()}
    print(
        f"Loaded {len(normalizers)} normalizers ({', '.join(types)}) "
        f"from '{optimizer_state_path}'"
    )
    return normalizers


def _get_bias_second_moment(
    layer_name: str,
    param_index_to_name: dict[int, str],
    optimizer_state: dict,
    include_bias: bool,
) -> torch.Tensor | None:
    """Look up bias exp_avg_sq for a layer, if present and requested."""
    if not include_bias:
        return None

    bias_name = layer_name + ".bias"
    for idx, name in param_index_to_name.items():
        if name == bias_name:
            bias_state = optimizer_state["state"].get(idx)
            if bias_state is not None:
                exp_avg_sq, _, _ = _extract_second_moments(bias_state)
                return exp_avg_sq
            break

    return None
