import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from peft import LoraConfig
from simple_parsing import parse
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bergson.config import DataConfig, IndexConfig
from bergson.data import load_data_string, tokenize
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.utils.utils import assert_type


def worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    train: Dataset,
    eval: Dataset,
    run_name,
):
    torch.cuda.set_device(rank)

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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype="bfloat16",
        revision=cfg.revision,
        device_map=f"cuda:{rank}",
    )

    callback = GradientCollectorCallback(
        Path(run_name) / "gradients",
        accumulate_grads=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,  # type: ignore
        eval_dataset=eval,
        callbacks=[callback],
        args=SFTConfig(
            max_length=8192,
            output_dir=run_name,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            # gradient_checkpointing=True,
            num_train_epochs=1,
            logging_steps=10,
            eval_steps=20,
            save_total_limit=3,
            group_by_length=True,
            completion_only_loss=True,
            ddp_find_unused_parameters=False,
            seed=0,
        ),
        peft_config=LoraConfig(),
    )
    trainer = prepare_for_gradient_collection(trainer)
    trainer.train()


def dist_worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    train: Dataset,
    eval: Dataset,
    run_name: str,
):
    try:
        worker(rank, world_size, cfg, train, eval, run_name)
    finally:
        dist.destroy_process_group()


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args: IndexConfig):
    seed = 0
    set_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)

    # Create DataConfig for tokenization
    data_config = DataConfig(
        prompt_column=args.data.prompt_column,
        completion_column=args.data.completion_column,
        conversation_column=args.data.conversation_column,
    )
    dataset = load_data_string(
        args.data.dataset,
        args.data.split,
        data_kwargs=args.data.data_kwargs,
    )
    dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )
    dataset = assert_type(Dataset, dataset)

    train, eval = dataset.train_test_split(
        test_size=0.05,
        seed=seed,
    ).values()

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, args, train, eval, args.run_path)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "train",
            dist_worker,
            args={
                i: (i, world_size, args, train, eval, args.run_path)
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


if __name__ == "__main__":
    args = parse(IndexConfig)

    main(args)
