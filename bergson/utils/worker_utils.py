import shutil
import warnings
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
    AutoConfig,
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
            skip_preconditioners=cfg.skip_preconditioners,
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
    elif cfg.fsdp or not torch.cuda.is_available():
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


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""
    ds = load_data_string(
        cfg.data.dataset, cfg.data.split, cfg.data.subset, cfg.data.data_args
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)

    default_model_max_len = getattr(tokenizer, "model_max_length", None)
    if (
        default_model_max_len is not None
        and cfg.token_batch_size > default_model_max_len
    ):
        raise ValueError(
            f"Token batch size {cfg.token_batch_size} exceeds model_max_length "
            f"({default_model_max_len}). "
            f"Use --token_batch_size {default_model_max_len} or smaller."
        )

    # Resolve the base model for config loading (PEFT adapters don't have
    # a full config.json, so we need the base model path).
    config_model: str = cfg.model
    try:
        peft_cfg = PeftConfig.from_pretrained(cfg.model)
        if peft_cfg.base_model_name_or_path:
            config_model = peft_cfg.base_model_name_or_path
    except ValueError:
        pass

    max_pos_emb = getattr(
        AutoConfig.from_pretrained(config_model, revision=cfg.revision),
        "max_position_embeddings",
        None,
    )
    if max_pos_emb is not None:
        max_length = min(max_pos_emb, cfg.token_batch_size)
    else:
        max_length = cfg.token_batch_size

    remove_columns = ds.column_names if cfg.drop_columns else None

    if not ds.column_names or "input_ids" not in ds.column_names:
        ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer, max_length=max_length),
        )

    if not cfg.data.truncation and isinstance(ds, Dataset):
        max_doc_len = max(ds["length"])
        if max_pos_emb is not None and max_doc_len > max_pos_emb:
            warnings.warn(
                f"Dataset contains a document longer than max_position_embeddings "
                f"({max_doc_len} > {max_pos_emb}). "
                f"Consider using --truncation."
            )
        elif max_doc_len > cfg.token_batch_size:
            warnings.warn(
                f"Dataset contains a document longer than token_batch_size "
                f"({max_doc_len} > {cfg.token_batch_size}). "
                f"Consider increasing --token_batch_size or using --truncation."
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

    # Experimental benchmarking feature
    if cfg.max_tokens is not None:
        ds = filter_by_max_tokens(ds, cfg)

    # Remove extraneous columns
    if remove_columns is not None:
        keep = {"length", "input_ids", "labels"}
        columns_to_remove = [col for col in remove_columns if col not in keep]
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)

    return ds
