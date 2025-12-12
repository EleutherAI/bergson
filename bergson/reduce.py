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
from bergson.config import IndexConfig, ReduceConfig
from bergson.data import allocate_batches
from bergson.utils import assert_type
from bergson.worker_utils import setup_model_and_peft

from .distributed import launch_distributed_run
from .worker_utils import create_processor, setup_data_pipeline


def reduce_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    reduce_cfg: ReduceConfig,
    ds: Dataset | IterableDataset,
):
    """
    Distributed worker that aggregates per-document gradients into a single vector.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    world_size : int
        Total number of workers participating in the run.
    index_cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    reduce_cfg : ReduceConfig
        Specifies aggregation strategy (mean/sum, unit normalization).
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
            timeout=timedelta(minutes=30),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(index_cfg, local_rank)
    processor = create_processor(index_cfg, rank)

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": index_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
        "reduce_cfg": reduce_cfg,
    }

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"], index_cfg.token_batch_size)
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
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches
            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(index_cfg.partial_run_path)

    if dist.is_initialized():
        dist.barrier()


def reduce(index_cfg: IndexConfig, reduce_cfg: ReduceConfig):
    """
    Reduce a dataset to a single aggregated gradient vector.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and many other gradient collection settings.
    reduce_cfg : ReduceConfig
        Specifies aggregation strategy (mean/sum, unit normalization).
    """
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (index_cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(index_cfg), f, indent=2)

    ds = setup_data_pipeline(index_cfg)

    launch_distributed_run("reduce", reduce_worker, [index_cfg, reduce_cfg, ds])

    rank = int(os.environ.get("START_RANK", os.environ.get("RANK", 0)))
    if rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
