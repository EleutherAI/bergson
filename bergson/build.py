import json
import os
import shutil
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm

from bergson.collection import collect_gradients
from bergson.config import IndexConfig
from bergson.data import allocate_batches
from bergson.utils import assert_type
from bergson.worker_utils import setup_model_and_peft

from .distributed import launch_distributed_run
from .worker_utils import create_processor, setup_data_pipeline


def build_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: IndexConfig,
    ds: Dataset | IterableDataset,
):
    """
    Build worker executed per rank to collect gradients to populate the index.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    world_size : int
        Total number of workers participating in the run.
    cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    """
    torch.cuda.set_device(local_rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(cfg, local_rank)
    processor = create_processor(cfg, local_rank, rank)

    attention_cfgs = {module: cfg.attention for module in cfg.split_attention_modules}

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"], cfg.token_batch_size)
        kwargs["batches"] = batches
        collect_gradients(**kwargs)
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush(kwargs):
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(ds_shard["length"][:], cfg.token_batch_size)
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches
            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(cfg.partial_run_path)


def build(index_cfg: IndexConfig):
    """
    Build a gradient index by distributing work across all available GPUs.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and many other gradient collection settings.
    """
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (index_cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(index_cfg), f, indent=2)

    ds = setup_data_pipeline(index_cfg)

    launch_distributed_run("build", build_worker, [index_cfg, ds])

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
