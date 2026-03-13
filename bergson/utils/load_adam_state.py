from pathlib import Path

import torch
from transformers import PreTrainedModel

from bergson.gradients import AdamNormalizer


def load_from_optimizer(
    model: PreTrainedModel,
    adam_state_path: str,
    include_bias: bool = False,
    target_modules: set[str] | None = None,
) -> dict[str, AdamNormalizer]:
    """Load Adam second moments (exp_avg_sq) from a checkpoint and create
    AdamNormalizer instances for each target linear layer.

    This function only supports modules with .weight and/or .bias
    parameters.

    Args:
        model: The model whose parameter names are used to map optimizer
            state indices to layer names.
        adam_state_path: Path to either a checkpoint directory containing
            ``optimizer.pt`` or directly to an optimizer state file.
        target_modules: Optional set of module names to include. If ``None``,
            all linear layers are included.

    Returns:
        Dictionary mapping layer names to ``AdamNormalizer`` instances.
    """
    # Load optimizer state
    state_path = Path(adam_state_path)
    if state_path.is_dir():
        state_path = state_path / "optimizer.pt"

    optimizer_state = torch.load(state_path, map_location="cpu", weights_only=True)

    # The optimizer state is keyed by position in the trainable parameter list.
    # For LoRA checkpoints, only include LoRA params.
    # Otherwise include all params.
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n]
    if lora_params:
        params_for_index = lora_params
    else:
        params_for_index = list(model.named_parameters())

    target_param_index_to_name: dict[int, str] = {}
    for idx, (name, param) in enumerate(params_for_index):
        target_param_index_to_name[idx] = name

        # Safety check
        if idx in optimizer_state["state"]:
            opt_shape = optimizer_state["state"][idx]["exp_avg_sq"].shape
            if opt_shape != param.shape:
                raise ValueError(
                    f"Shape mismatch at index {idx}: param '{name}' has shape "
                    f"{tuple(param.shape)} but optimizer state has {tuple(opt_shape)}. "
                    f"The parameter ordering may have changed between training "
                    f"and loading."
                )

    # Extract second moments per layer
    normalizers: dict[str, AdamNormalizer] = {}
    for param_idx, state in optimizer_state["state"].items():
        param_idx = int(param_idx)
        if param_idx not in target_param_index_to_name:
            continue

        param_name = target_param_index_to_name[param_idx]

        if not param_name.endswith(".weight"):
            continue

        weight_exp_avg_sq = state.get("exp_avg_sq")

        # Skips 1D weights such as LayerNorm
        if weight_exp_avg_sq is None or weight_exp_avg_sq.ndim != 2:
            continue

        # Extract layer name
        layer_name = param_name.removesuffix(".weight")
        # PEFT models prefix param names with "base_model." but target module
        # names don't have it — strip so normalizer keys match collector keys.
        module_name = layer_name.removeprefix("base_model.")

        if target_modules is not None and module_name not in target_modules:
            continue

        bias_exp_avg_sq = None
        if include_bias:
            bias_name = layer_name + ".bias"
            for idx, name in target_param_index_to_name.items():
                if name == bias_name:
                    bias_state = optimizer_state["state"].get(idx)
                    if bias_state is not None:
                        bias_exp_avg_sq = bias_state.get("exp_avg_sq")
                    break

        normalizers[module_name] = AdamNormalizer(
            weight_avg_sq=weight_exp_avg_sq,
            bias_avg_sq=bias_exp_avg_sq,
        )

    assert normalizers, (
        f"No Adam second moments (exp_avg_sq) found in '{adam_state_path}'. "
        "Ensure the checkpoint was saved from an Adam-family optimizer."
    )

    # Move normalizer tensors to the model's device
    device = next(model.parameters()).device
    for norm in normalizers.values():
        norm.weight_avg_sq = norm.weight_avg_sq.to(device)
        if norm.bias_avg_sq is not None:
            norm.bias_avg_sq = norm.bias_avg_sq.to(device)

    print(f"Loaded {len(normalizers)} Adam normalizers from '{adam_state_path}'")
    return normalizers
