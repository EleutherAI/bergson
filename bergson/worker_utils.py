from pathlib import Path

import pandas as pd
import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bergson.config import DataConfig, IndexConfig
from bergson.data import tokenize
from bergson.gradients import GradientProcessor
from bergson.utils import assert_type, get_layer_list


def create_processor(
    cfg: IndexConfig,
    rank: int,
) -> GradientProcessor:
    """Handle processor creation and normalizer fitting"""
    processor_path = Path(cfg.processor_path)
    if (processor_path / "processor_config.json").exists():
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            {},
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
    rank: int,
) -> tuple[AutoModelForCausalLM, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""

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
    device_map = {"": f"cuda:{rank}"} if not cfg.fsdp else "cpu"
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
            dtype=dtype,
            revision=cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
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

    return model, target_modules  # type: ignore


def estimate_advantage(ds: Dataset, cfg: DataConfig):
    """Group rollouts by prompt and estimate advantages."""
    df = ds.select_columns([cfg.prompt_column, cfg.reward_column]).to_pandas()
    df = assert_type(pd.DataFrame, df)

    advantages = df[cfg.reward_column] - df.groupby(cfg.prompt_column)[
        cfg.reward_column
    ].transform("mean")

    return advantages.tolist()


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""

    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(
                data_str,
                cfg.data.subset,
                split=cfg.data.split,
                streaming=cfg.data.streaming,
            )

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    remove_columns = ds.column_names if cfg.drop_columns else None

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )

    if cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    return ds
