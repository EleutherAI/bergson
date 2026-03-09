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
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.distributed import launch_distributed_run
from bergson.utils.auto_batch_size import maybe_auto_batch_size
from bergson.utils.utils import assert_type, setup_reproducibility
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def build_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    preprocess_cfg: PreprocessConfig,
    ds: Dataset | IterableDataset,
):
    """
    Build worker executed per rank to collect gradients to populate the
    on-disk index.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    local_rank : int
        Local rank / GPU ID for this worker on the node.
    world_size : int
        Total number of workers participating in the run.
    index_cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    preprocess_cfg : PreprocessConfig
        Specifies preprocessing strategy (preconditioning, unit normalization,
        aggregation).
    ds : Dataset | IterableDataset
        The entire dataset to be processed. A subset is assigned to each worker.
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

    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, ds, index_cfg, target_modules)

    maybe_auto_batch_size(index_cfg, model, ds, processor, target_modules, rank)

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
        "preprocess_cfg": preprocess_cfg,
    }

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
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


def build(
    index_cfg: IndexConfig,
    preprocess_cfg: PreprocessConfig,
):
    """
    Convert a dataset to an on-disk index.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and many other gradient collection settings.
    preprocess_cfg : PreprocessConfig
        Preprocessing configuration for gradient normalization, preconditioning,
        and aggregation.
    """
    if index_cfg.debug:
        setup_reproducibility()

    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (index_cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(index_cfg), f, indent=2)

    with (index_cfg.partial_run_path / "preprocess_config.json").open("w") as f:
        json.dump(asdict(preprocess_cfg), f, indent=2)

    ds = setup_data_pipeline(index_cfg)

    launch_distributed_run(
        "build",
        build_worker,
        [index_cfg, preprocess_cfg, ds],
        index_cfg.distributed,
    )

    if index_cfg.distributed.rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
