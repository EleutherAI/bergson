import os

import torch

from bergson.config import DistributedConfig, IndexConfig
from bergson.distributed import launch_distributed_run


def check_no_gpu(rank: int, local_rank: int, world_size: int):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()


def test_empty_cuda_visible_devices_hides_gpus_from_every_rank(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    launch_distributed_run(
        "test", check_no_gpu, [], DistributedConfig(nproc_per_node=2)
    )


def check_tf32(rank: int, local_rank: int, world_size: int):
    assert torch.get_float32_matmul_precision() == "high"


def test_workers_use_the_parents_matmul_precision(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_float32_matmul_precision()
    IndexConfig(run_path="unused", use_tf32_matmuls=True)
    try:
        launch_distributed_run(
            "test", check_tf32, [], DistributedConfig(nproc_per_node=2)
        )
    finally:
        torch.set_float32_matmul_precision(previous)
