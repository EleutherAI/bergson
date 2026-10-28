import os
import subprocess
import sys

import torch

from bergson.config import DistributedConfig, IndexConfig
from bergson.distributed import launch_distributed_run


def check_worker_env(rank: int, local_rank: int, world_size: int):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()
    assert torch.get_float32_matmul_precision() == "high"


def test_workers_inherit_hidden_gpus_and_matmul_precision(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_float32_matmul_precision()
    IndexConfig(run_path="unused", use_tf32_matmuls=True)
    try:
        launch_distributed_run(
            "test", check_worker_env, [], DistributedConfig(nproc_per_node=2)
        )
    finally:
        torch.set_float32_matmul_precision(previous)


# Spawned children import the parent's main module, which takes two seconds here.
SLOW_MAIN = """
import sys
import time
from pathlib import Path

from bergson.config import DistributedConfig
from bergson.distributed import launch_distributed_run

time.sleep(2)


def record_start(rank, local_rank, world_size, payload, out):
    (Path(out) / str(rank)).write_text(str(time.time()))


if __name__ == "__main__":
    launch_distributed_run(
        "test",
        record_start,
        [bytes(1 << 20), sys.argv[1]],
        DistributedConfig(nproc_per_node=4),
    )
"""


def test_workers_with_large_arguments_start_together(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = tmp_path / "slow_main.py"
    script.write_text(SLOW_MAIN)
    out = tmp_path / "starts"
    out.mkdir()
    subprocess.run([sys.executable, script, out], check=True)
    starts = [float(p.read_text()) for p in out.iterdir()]
    assert len(starts) == 4
    assert max(starts) - min(starts) < 2


def check_threads(rank: int, local_rank: int, world_size: int, expected: int):
    assert torch.get_num_threads() == expected


def test_workers_share_the_cores(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    launch_distributed_run(
        "test",
        check_threads,
        [max(1, os.cpu_count() // 4)],
        DistributedConfig(nproc_per_node=4),
    )
