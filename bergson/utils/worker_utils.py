from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    IterableDataset,
)
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from torch.distributed.fsdp import fully_shard
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from bergson.config import DataConfig, IndexConfig
from bergson.data import allocate_batches, load_data_string, tokenize
from bergson.gradients import GradientProcessor, Normalizer
from bergson.normalizer.fit_normalizers import fit_normalizers
from bergson.utils.utils import assert_type, get_layer_list


def create_normalizers(
    model: PreTrainedModel,
    ds: Dataset | IterableDataset,
    cfg: IndexConfig,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """Create normalizers for the model"""
    if cfg.normalizer != "none":
        # Evenly sample `stats_sample_size` examples to compute statistics
        if isinstance(ds, Dataset):
            if cfg.stats_sample_size is not None and cfg.stats_sample_size < len(ds):
                stats_ds = ds.shuffle(seed=0).select(range(cfg.stats_sample_size))
            else:
                stats_ds = ds
        else:
            if cfg.stats_sample_size is None:
                stats_iterable_ds = ds
            else:
                stats_iterable_ds = ds.shuffle(seed=0).take(cfg.stats_sample_size)

            stats_ds = assert_type(
                Dataset, Dataset.from_generator(lambda: iter(stats_iterable_ds))
            )

        return fit_normalizers(
            model,
            stats_ds,
            cfg,
            batches=allocate_batches(stats_ds["length"][:], cfg.token_batch_size),
            target_modules=target_modules,
        )

    return {}


def create_processor(
    model: PreTrainedModel,
    ds: Dataset | IterableDataset,
    cfg: IndexConfig,
    target_modules: set[str] | None = None,
) -> GradientProcessor:
    """Handle processor creation and normalizer fitting"""
    local_rank = cfg.distributed.local_rank
    rank = cfg.distributed.rank

    processor_path = Path(cfg.processor_path)
    if (processor_path / "processor_config.json").exists():
        if local_rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            processor_path,
            map_location=f"cuda:{local_rank}",
        )
    else:
        normalizers = create_normalizers(model, ds, cfg, target_modules)

        processor = GradientProcessor(
            normalizers,
            projection_dim=cfg.projection_dim or None,
            reshape_to_square=cfg.reshape_to_square,
            projection_type=cfg.projection_type,
            include_bias=cfg.include_bias,
        )
        if rank == 0:
            processor.save(cfg.partial_run_path)

    return processor


def setup_model_and_peft(
    cfg: IndexConfig,
    device_map_auto: bool = False,
) -> tuple[PreTrainedModel, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""
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
    elif cfg.fsdp:
        device_map = "cpu"
    else:
        device_map = {"": f"cuda:{local_rank}"}

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

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(cfg.model)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            revision=cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            revision=cfg.revision,
        )

        model = PeftModel.from_pretrained(
            base_model,
            cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )

        # Extract target modules
        target_modules = set()
        peft_state_dict = get_peft_model_state_dict(model=model)
        for adapter in model.peft_config.keys():
            for name in list(peft_state_dict.keys()):
                prefix = name.removesuffix(".weight")
                processed_name = f"{prefix}.{adapter}".removeprefix("base_model.")
                try:
                    model.get_submodule(processed_name)
                    target_modules.add(processed_name)
                except AttributeError:
                    print(
                        f"Adapter parameter '{processed_name}' not found in the model."
                    )

    # Configure gradients
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)  # type: ignore

    # Apply FSDP if needed
    if cfg.fsdp:
        for layer in get_layer_list(model):  # type: ignore
            fully_shard(layer)
        fully_shard(model)

    model = cast(PreTrainedModel, model)

    return model, target_modules  # type: ignore


def estimate_advantage(ds: Dataset, cfg: DataConfig):
    """Group rollouts by prompt and estimate advantages."""
    df = ds.select_columns([cfg.prompt_column, cfg.reward_column]).to_pandas()
    df = assert_type(pd.DataFrame, df)

    advantages = df[cfg.reward_column] - df.groupby(cfg.prompt_column)[
        cfg.reward_column
    ].transform("mean")

    return advantages.tolist()


def filter_by_max_tokens(
    ds: Dataset | IterableDataset, cfg: IndexConfig
) -> Dataset | IterableDataset:
    """Filter the dataset by the max tokens limit. This is an experimental
    benchmarking feature that may be removed in the future."""
    if cfg.max_tokens is None:
        return ds

    if isinstance(ds, IterableDataset):
        raise ValueError("max_tokens is not supported for IterableDataset")

    total_tokens = 0
    indices_to_keep = []
    for idx, length in enumerate(ds["length"]):
        if total_tokens + length > cfg.max_tokens:
            break
        indices_to_keep.append(idx)
        total_tokens += length

    if indices_to_keep:
        ds = ds.select(indices_to_keep)
        print(
            f"Filtered dataset to {len(indices_to_keep)} examples "
            f"({total_tokens} tokens) due to max_tokens limit."
        )
    else:
        print("Warning: No examples fit within max_tokens limit.")

    return ds


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""
    ds = load_data_string(
        cfg.data.dataset, cfg.data.split, cfg.data.subset, cfg.data.data_args
    )

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    remove_columns = ds.column_names if cfg.drop_columns else None

    if not ds.column_names or "input_ids" not in ds.column_names:
        ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        )

    if cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"

        rewards = np.array(ds[cfg.data.reward_column], dtype=np.float64)
        nan_mask = np.isnan(rewards)
        if nan_mask.any():
            if cfg.data.skip_nan_rewards:
                print(f"Warning: Filtering out {nan_mask.sum()} rows with NaN rewards")
                ds = ds.filter(lambda _, idx: not nan_mask[idx], with_indices=True)
            else:
                raise ValueError(
                    f"Reward column '{cfg.data.reward_column}' contains NaN values"
                )

        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    ds = filter_by_max_tokens(ds, cfg)

    # Keep length and input_ids
    if remove_columns is not None:
        columns_to_remove = [
            col for col in remove_columns if col not in {"length", "input_ids"}
        ]
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)

    return ds
