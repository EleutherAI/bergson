import hashlib
import json
import os
import random
import socket
from dataclasses import asdict
from datetime import timedelta
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bergson.data import IndexConfig, allocate_batches, tokenize
from bergson.gradients import GradientProcessor
from bergson.utils import assert_type, get_layer_list


def setup_reproducibility():
    """Setup reproducibility for distributed training"""
    seed: int = 42
    # Set all random seeds - same across all ranks for model consistency
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Environment variables for determinism
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""

    torch.manual_seed(42)

    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(data_str, split="train")

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

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, model_max_length=cfg.token_batch_size
    )

    ds = ds.map(
        tokenize, batched=True, fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer)
    )

    return ds


def setup_model_and_peft(
    cfg: IndexConfig, rank: int, dtype: torch.dtype
) -> tuple[AutoModelForCausalLM, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Common configuration
    if cfg.fsdp or not torch.cuda.is_available():
        device_map = "cpu"
    else:
        device_map = {"": f"cuda:{rank}"}
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
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
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
        for layer in get_layer_list(model):
            fully_shard(layer)
        fully_shard(model)

    return model, target_modules  # type: ignore


def create_processor(
    cfg: IndexConfig,
    model,
    ds: Dataset | IterableDataset,
    rank: int,
    target_modules: set | None,
) -> GradientProcessor:
    """Handle processor creation and normalizer fitting"""
    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            projection_dim=cfg.projection_dim or None,
        )
        if rank == 0:
            processor.save(cfg.run_path)

    return processor


def worker_wrapper(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    ds: Dataset | IterableDataset,
    worker_fn: Callable,
    setup_model: bool = True,
    setup_processor: bool = True,
):
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        if cfg.debug:
            setup_reproducibility()
            print("DEBUG MODE IS ENABLED: quasi-deterministic training")

        # These should be set by the main process
        if world_size > 1:
            addr = os.environ.get("MASTER_ADDR", "localhost")
            port = os.environ.get("MASTER_PORT", "29500")

            dist.init_process_group(
                "nccl",
                init_method=f"tcp://{addr}:{port}",
                device_id=torch.device(f"cuda:{rank}"),
                rank=rank,
                timeout=timedelta(hours=1),
                world_size=world_size,
            )

        # Initialize defaults for optional components
        model, target_modules, processor = None, None, None

        if setup_model:
            match cfg.precision:
                case "bf16":
                    dtype = torch.bfloat16
                case "fp16":
                    dtype = torch.float16
                case "fp32":
                    dtype = torch.float32
                case "int4" | "int8":
                    dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                case other:
                    raise ValueError(f"Unsupported precision: {other}")

            model, target_modules = setup_model_and_peft(cfg, rank, dtype)

        if setup_processor:
            if model is None:
                raise ValueError(
                    "Cannot create processor without model. Set setup_model=True or provide model externally."
                )
            processor = create_processor(cfg, model, ds, rank, target_modules)

        if setup_model and setup_processor:
            assert isinstance(ds, Dataset)
            batches = allocate_batches(ds["length"], cfg.token_batch_size)
            worker_fn(
                model,
                ds,
                processor,
                batches=batches,
                target_modules=target_modules,
                cfg=cfg,
            )
        else:
            # Simplified setup - for compatibility with ekfac_apply style
            worker_fn(cfg)
    finally:
        if dist.is_initialized():
            try:
                # Add a barrier to ensure all processes reach this point
                dist.barrier()
            except Exception:
                pass  # Ignore barrier failures during cleanup

            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Ignore cleanup failures


def distributed_computing(
    cfg: IndexConfig,
    worker_fn: Callable,
    setup_data: bool = True,
    setup_model: bool = True,
    setup_processor: bool = True,
):
    # save cfg as json
    if cfg.apply_ekfac:
        path_hash = hashlib.md5(cfg.ekfac_path.encode("utf-8")).hexdigest()
        cfg.run_path = os.path.join(cfg.run_path, f"query_{path_hash}")

    os.makedirs(cfg.run_path, exist_ok=True)

    with open(os.path.join(cfg.run_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    # Setup data pipeline if requested
    if setup_data:
        ds = setup_data_pipeline(cfg)
    else:
        # Create empty dataset for compatibility
        ds = assert_type(Dataset, Dataset.from_list([]))

    world_size = torch.cuda.device_count() if cfg.world_size is None else cfg.world_size
    if world_size <= 1:
        worker_wrapper(0, 1, cfg, ds, worker_fn, setup_model, setup_processor)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = None
        try:
            ctx = start_processes(
                "build",
                worker_wrapper,
                args={
                    i: (i, world_size, cfg, ds, worker_fn, setup_model, setup_processor)
                    for i in range(world_size)
                },
                envs={
                    i: {
                        "LOCAL_RANK": str(i),
                        "MASTER_ADDR": "localhost",
                        "MASTER_PORT": str(port),
                    }
                    for i in range(world_size)
                },
                logs_specs=DefaultLogsSpecs(),
            )
            ctx.wait()
        finally:
            if ctx is not None:
                ctx.close()  # Kill any processes that are still running
