import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    IterableDataset,
)
from peft import (
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from torch.distributed.fsdp import fully_shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from bergson.config import AttributionConfig, DataConfig, IndexConfig, ModelConfig
from bergson.data import (
    expand_dpo_dataset,
    load_data_string,
    tokenize,
    tokenize_and_chunk,
)
from bergson.format import apply_format
from bergson.gradients import GradientProcessor, Normalizer
from bergson.utils import assert_type, get_layer_list, weighted_causal_lm_ce
from bergson.utils.utils import get_device, simple_parse_kwargs_string

BIG_NUM = np.iinfo(np.int64).max


def validate_run_path(index_cfg: IndexConfig):
    """Validate the run path."""
    if index_cfg.distributed.rank != 0:
        return

    for path in [Path(index_cfg.run_path), Path(index_cfg.partial_run_path)]:
        if not path.exists():
            continue

        if index_cfg.overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Run path {path} already exists. Use --overwrite to overwrite it."
            )


def create_processor(
    model: PreTrainedModel | PeftModel,
    cfg: IndexConfig,
    target_modules: set[str] | None = None,
) -> GradientProcessor:
    """Handle processor creation and normalizer loading."""
    local_rank = cfg.distributed.local_rank
    rank = cfg.distributed.rank

    processor_path = Path(cfg.processor_path)
    if (processor_path / "processor_config.yaml").exists():
        if local_rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            processor_path,
            map_location=get_device(local_rank),
            skip_hessians=cfg.skip_hessians,
        )
    else:
        normalizers: dict[str, Normalizer] = {}
        if cfg.optimizer_state:
            from bergson.utils.load_from_optimizer import load_from_optimizer

            normalizers = load_from_optimizer(
                model,
                cfg.optimizer_state,
                include_bias=cfg.include_bias,
                target_modules=target_modules,
            )

        processor = GradientProcessor(
            normalizers,
            projection_dim=cfg.projection_dim or None,
            reshape_to_square=cfg.reshape_to_square,
            projection_type=cfg.projection_type,
            projection_target=cfg.projection_target,
            include_bias=cfg.include_bias,
        )
        if rank == 0:
            processor.save(cfg.partial_run_path)

    return processor


def apply_force_math_sdp(cfg: ModelConfig) -> None:
    """Disable flash and memory-efficient SDPA backends when requested.

    Forces the math-only SDPA kernel, which produces consistent gradients
    across different padding lengths and batch compositions.
    """
    if not getattr(cfg, "force_math_sdp", False):
        return

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("force_math_sdp: disabled flash and memory-efficient SDPA backends")


def extract_peft_target_modules(model) -> set[str]:
    """Extract adapter module names from a PeftModel.

    PEFT's ``get_peft_model_state_dict`` removes the adapter name from keys,
    so reconstructing module names from that state dict is fragile across PEFT
    and model wrapper versions. Use the live parameter names instead; those are
    the exact module paths that ``get_submodule`` can resolve.
    """
    target_modules: set[str] = set()
    adapter_names = set(model.peft_config.keys())  # type: ignore
    module_root = getattr(model, "base_model", model)

    def resolve_candidates(param_name: str) -> tuple[str, ...]:
        """Return possible module names for a PEFT parameter path.

        Newer PEFT versions may expose names with or without an adapter namespace
        (for example ``.default``). Try both forms when resolving submodules.
        """
        module_name = param_name.rsplit(".", 1)[0]
        candidates: list[str] = [module_name]

        for adapter in adapter_names:
            adapter_token = f".{adapter}"
            if adapter_token in module_name:
                candidates.append(module_name.replace(adapter_token, "", 1))
            else:
                candidates.append(f"{module_name}{adapter_token}")

        normalized: list[str] = []
        for candidate in candidates:
            normalized.append(candidate)
            normalized.append(candidate.removeprefix("base_model."))

        return tuple(dict.fromkeys(normalized))

    for name, _param in model.named_parameters():
        parts = name.split(".")
        if not any(token.startswith("lora_") for token in parts):
            continue

        for candidate in resolve_candidates(name):
            try:
                module_root.get_submodule(candidate)
                target_modules.add(candidate)
                break
            except AttributeError:
                continue

    if target_modules:
        return target_modules

    # Fallback for non-LoRA PEFT methods or unexpected PEFT naming.
    peft_state_dict = get_peft_model_state_dict(model=model)
    for name in list(peft_state_dict.keys()):
        parts = name.split(".")
        if not any(token.startswith("lora_") for token in parts):
            continue

        prefix = name.removesuffix(".weight")
        seeds = (prefix, prefix.removeprefix("base_model."))
        for seed in seeds:
            for candidate in resolve_candidates(seed + ".weight"):
                try:
                    module_root.get_submodule(candidate)
                    target_modules.add(candidate)
                    break
                except AttributeError:
                    continue

    return target_modules


def setup_model_and_peft(
    cfg: ModelConfig,
    device_map_auto: bool = False,
    apply_fsdp: bool = True,
    **model_kwargs,
) -> tuple[PreTrainedModel | PeftModel, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""
    apply_force_math_sdp(cfg)

    local_rank = cfg.distributed.local_rank

    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case "auto":
            dtype = "auto"
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    # Common configuration
    if device_map_auto:
        device_map = "auto"
    elif cfg.fsdp or not torch.cuda.is_available():
        device_map = "cpu"
    else:
        device_map = {"": get_device(local_rank)}

    quantization_config = None
    if cfg.precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg.precision == "int4",
            load_in_8bit=cfg.precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Determine base model path and whether we're loading a pretrained adapter
    try:
        pretrained_peft_config = PeftConfig.from_pretrained(cfg.model)
    except ValueError:
        pretrained_peft_config = None

    assert not (cfg.peft_init_kwargs and pretrained_peft_config), (
        f"peft_init_args is set but '{cfg.model}' is already a" " PEFT adapter."
    )

    base_model_path = (
        pretrained_peft_config.base_model_name_or_path  # type: ignore
        if pretrained_peft_config
        else cfg.model
    )
    assert base_model_path is not None

    model_kwargs.update(simple_parse_kwargs_string(cfg.model_kwargs))
    if model_kwargs.get("attn_implementation") == "bergson_efficient_sdpa":
        from bergson.score.output_model import register_efficient_attention
        register_efficient_attention()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        dtype=dtype,
        revision=cfg.revision,
        **model_kwargs,
    )
    model.loss_function = weighted_causal_lm_ce
    target_modules = None

    if cfg.peft_init_kwargs:
        # Initialize a fresh PEFT adapter
        peft_kwargs = simple_parse_kwargs_string(cfg.peft_init_kwargs)
        peft_type = PeftType(peft_kwargs.pop("peft_type", "LORA"))
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        model = get_peft_model(model, peft_config_cls(**peft_kwargs))
        target_modules = extract_peft_target_modules(model)
    elif pretrained_peft_config:
        # Load pretrained PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )
        target_modules = extract_peft_target_modules(model)  # type: ignore

    if selected_modules := getattr(cfg, "modules", None):
        requested = set(selected_modules)
        if target_modules is not None and not requested.issubset(target_modules):
            raise ValueError("Requested modules are outside the trainable adapter modules")
        for name in requested:
            model.base_model.get_submodule(name)
        target_modules = requested

    if getattr(cfg, "stream_output_loss", False):
        from bergson.score.output_model import install_output_loss_head
        install_output_loss_head(model, checkpoint_reverse=True)

    if getattr(cfg, "checkpoint_mlp", False):
        from bergson.utils.recompute import checkpoint_mlps
        checkpoint_mlps(model)

    if getattr(cfg, "checkpoint_attention", False):
        from bergson.utils.recompute import checkpoint_attentions
        checkpoint_attentions(model)

    # Configure gradients
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)  # type: ignore

    # Apply FSDP if needed
    if cfg.fsdp and apply_fsdp:
        for layer in get_layer_list(model):  # type: ignore
            fully_shard(layer)
        fully_shard(model)

    return model, target_modules  # type: ignore


def estimate_advantage(ds: Dataset, cfg: DataConfig):
    """Group rollouts by prompt and estimate advantages."""
    df = ds.select_columns([cfg.prompt_column, cfg.reward_column]).to_pandas()
    df = assert_type(pd.DataFrame, df)

    advantages = df[cfg.reward_column] - df.groupby(cfg.prompt_column)[
        cfg.reward_column
    ].transform("mean")

    return advantages.tolist()


def filter_by_max_tokens(ds: Dataset, cfg: AttributionConfig) -> Dataset:
    """Filter the dataset by the max tokens limit. This is an experimental
    benchmarking feature that may be removed in the future.

    If the dataset has fewer tokens than ``max_tokens``, rows are
    repeated (tiled) until the budget is met so that benchmarks
    always process the requested number of tokens regardless of
    the on-disk dataset size.
    """
    if cfg.max_tokens is None:
        return ds

    if isinstance(ds, IterableDataset):
        raise ValueError("max_tokens is not supported for IterableDataset")

    lengths = ds["length"]
    dataset_tokens = sum(lengths)

    if dataset_tokens >= cfg.max_tokens:
        # Dataset is large enough: take a prefix.
        total_tokens = 0
        indices_to_keep: list[int] = []
        for idx, length in enumerate(lengths):
            if total_tokens + length > cfg.max_tokens:
                break
            indices_to_keep.append(idx)
            total_tokens += length

        if indices_to_keep:
            ds = ds.select(indices_to_keep)
            print(
                f"Filtered dataset to "
                f"{len(indices_to_keep)} examples "
                f"({total_tokens} tokens) "
                f"due to max_tokens limit."
            )
        else:
            print("Warning: No examples fit within " "max_tokens limit.")
    else:
        # Dataset is too small: tile rows to fill budget.
        n = len(ds)
        full_repeats = cfg.max_tokens // dataset_tokens
        indices = list(range(n)) * full_repeats
        total_tokens = dataset_tokens * full_repeats

        # Fill the remainder with a partial pass.
        for idx in range(n):
            if total_tokens + lengths[idx] > cfg.max_tokens:
                break
            indices.append(idx)
            total_tokens += lengths[idx]

        ds = ds.select(indices)
        print(
            f"Tiled dataset ~{full_repeats}x to "
            f"{len(indices)} examples "
            f"({total_tokens} tokens) "
            f"to reach max_tokens={cfg.max_tokens}."
        )

    return ds


def max_tokens_for_model(tokenizer, model_str: str, revision: str | None) -> int:
    # You might think model_max_length should always be the same as
    # max_position_embeddings, but some models (e.g. Pythia!) have a smaller
    # max_position_embeddings than model_max_length, so we need to check both.
    # Resolve the base model for config loading (PEFT adapters don't have
    # a full config.yaml, so we need the base model path).
    try:
        peft_cfg = PeftConfig.from_pretrained(model_str)
        if peft_cfg.base_model_name_or_path:
            model_str = peft_cfg.base_model_name_or_path
    except ValueError:
        pass

    model_cfg = AutoConfig.from_pretrained(model_str, revision=revision)
    model_max_length = getattr(tokenizer, "model_max_length", BIG_NUM)
    max_pos_emb = getattr(model_cfg, "max_position_embeddings", BIG_NUM)
    return min(model_max_length, max_pos_emb)


def setup_data_pipeline(
    cfg: AttributionConfig,
    data_cfg: DataConfig | None = None,
) -> tuple[Dataset, int]:
    """Handle data loading and preprocessing"""
    data_cfg = data_cfg or cfg.data

    ds = load_data_string(
        data_cfg.dataset, data_cfg.split, data_cfg.subset, data_cfg.data_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)
    max_model_length = max_tokens_for_model(tokenizer, cfg.model, cfg.revision)

    if data_cfg.chunk_length > 0:
        # Sanity check
        if data_cfg.chunk_length > max_model_length:
            raise ValueError(
                f"chunk_length {data_cfg.chunk_length} exceeds model's maximum context"
                f" length {max_model_length}"
            )

        tokenized = tokenize_and_chunk(
            ds,
            tokenizer,
            chunk_size=data_cfg.chunk_length,
        )
        return tokenized, len(ds)

    max_token_bz = getattr(cfg, "token_batch_size", BIG_NUM)
    if BIG_NUM > max_token_bz > max_model_length:
        raise ValueError(
            f"Token batch size {max_token_bz} exceeds model max length "
            f"({max_model_length}). "
            f"Use --token_batch_size {max_model_length} or smaller."
        )

    if data_cfg.max_length < 0:
        raise ValueError("data.max_length must be nonnegative")
    max_length = min(max_model_length, data_cfg.max_length or max_model_length)
    remove_columns = set(ds.column_names) if cfg.drop_columns else set()
    tokenize_cfg = data_cfg

    if data_cfg.format_template:
        ds = apply_format(ds, data_cfg.format_template)
        tokenize_cfg = DataConfig(
            prompt_column="prompt" if "completion" in ds.column_names else "text",
            completion_column="completion" if "completion" in ds.column_names else "",
            truncation=data_cfg.truncation,
        )

    # Already-tokenized DPO datasets have chosen_input_ids/rejected_input_ids;
    # do not send them through the text tokenizer again.
    if ("input_ids" not in ds.column_names
            and "chosen_input_ids" not in ds.column_names):
        ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(
                args=tokenize_cfg,
                tokenizer=tokenizer,
                max_length=max_length,
            ),
        )

    # DPO: expand chosen/rejected columns into a 2N-row standard dataset
    if isinstance(ds, Dataset) and "chosen_input_ids" in ds.column_names:
        ds = expand_dpo_dataset(ds)

    # Suggest to the user that they turn on truncation
    if not data_cfg.truncation:
        max_doc_len = max(ds["length"])
        if max_model_length is not None and max_doc_len > max_model_length:
            warnings.warn(
                f"Dataset contains a document longer than the model can handle "
                f"({max_doc_len} > {max_model_length}). "
                f"Consider using --truncation."
            )
        elif max_doc_len > max_token_bz:
            warnings.warn(
                f"Dataset contains a document longer than token_batch_size "
                f"({max_doc_len} > {max_token_bz}). "
                f"Consider increasing --token_batch_size or using --truncation."
            )

    if data_cfg.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"

        rewards = np.array(ds[data_cfg.reward_column], dtype=np.float64)
        nan_mask = np.isnan(rewards)
        if nan_mask.any():
            if data_cfg.skip_nan_rewards:
                print(f"Warning: Filtering out {nan_mask.sum()} rows with NaN rewards")
                ds = ds.filter(lambda _, idx: not nan_mask[idx], with_indices=True)
            else:
                raise ValueError(
                    f"Reward column '{data_cfg.reward_column}' contains NaN values"
                )

        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, data_cfg),
            new_fingerprint="advantage",  # type: ignore
        )

    # Experimental benchmarking feature
    if cfg.max_tokens is not None:
        ds = filter_by_max_tokens(ds, cfg)

    # Remove extraneous columns
    keep = {"length", "input_ids", "labels", "pair_index", "dpo_side",
            "reference_logps"}
    if cfg.example_weight_column:
        keep.add(cfg.example_weight_column)
    remove_columns -= keep
    remove_columns &= set(ds.column_names)
    if remove_columns:
        ds = ds.remove_columns(list(remove_columns))

    return ds, len(ds)
