import os
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset

from bergson.collection import collect_gradients
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.distributed import (
    cap_world_size_to_dataset,
    launch_distributed_run,
    parent_barrier,
)
from bergson.utils.batch_size import maybe_auto_batch_size
from bergson.utils.utils import (
    dist_backend,
    dist_device_id,
    get_device_index,
    setup_reproducibility,
)
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def build_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    index_cfg: IndexConfig,
    preprocess_cfg: PreprocessConfig,
    ds: Dataset,
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
    ds : Dataset
        The entire dataset to be processed. A subset is assigned to each worker.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(local_rank))

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            dist_backend(),
            init_method=f"tcp://{addr}:{port}",
            device_id=dist_device_id(local_rank),
            rank=rank,
            timeout=timedelta(minutes=30),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, index_cfg, target_modules)

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

    batches = allocate_batches(
        ds["length"][:],
        index_cfg.token_batch_size,
        max_batch_size=index_cfg.max_batch_size,
    )
    kwargs["batches"] = batches
    collect_gradients(**kwargs)


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

    ds, _ = setup_data_pipeline(index_cfg)

    dist_cfg = index_cfg.distributed
    if isinstance(ds, Dataset) and len(ds) < dist_cfg.world_size:
        dist_cfg = cap_world_size_to_dataset(index_cfg.distributed, len(ds))
        print(
            f"reducing to nnode=1 and nproc_per_node={dist_cfg.nproc_per_node} for step"
        )

    launch_distributed_run(
        "build",
        build_worker,
        [index_cfg, preprocess_cfg, ds],
        dist_cfg,
    )

    if dist_cfg.rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)

    if dist_cfg.world_size < index_cfg.distributed.world_size:
        parent_barrier(index_cfg.distributed)
