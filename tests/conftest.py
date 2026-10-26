import os

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments


def pytest_configure(config) -> None:
    """Spread pytest-xdist workers across GPUs.

    Everything that doesn't go through launch_distributed_run runs on cuda:0,
    so with parallel workers the entire suite queues on one GPU while the
    rest idle (measured ~10x inflation on GPU-bound tests). Rotating each
    worker's CUDA_VISIBLE_DEVICES keeps every GPU visible — multi-GPU tests
    still see the full set — but maps cuda:0 to a different physical device
    per worker. This must happen before the worker initializes CUDA, which
    is why it lives in pytest_configure rather than a fixture.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not worker.startswith("gw"):
        return

    # Torch's intra-op pool defaults to all cores in every worker (and in
    # every subprocess the tests spawn), so parallel workers oversubscribe
    # the CPU severely — measured load ~2x core count and 10-50x test-time
    # inflation. Give each worker an even share instead.
    worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0") or 0)
    if worker_count > 1:
        threads = max(1, (os.cpu_count() or worker_count) // worker_count)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)

    if not torch.cuda.is_available():
        return

    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [d.strip() for d in devices.split(",") if d.strip()]
        if devices
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    if len(ids) > 1:
        start = int(worker[2:]) % len(ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids[start:] + ids[:start])


# Longest-running tests (and xdist groups), by measured duration. Under
# pytest-xdist, tests are dispatched in collection order, so these must go
# first: alphabetical order starts the ~3-minute distributed_magic group
# mid-run, turning it into an idle-worker tail that erases most of the
# speedup from parallelism. No effect on a serial run beyond ordering.
FRONT_LOADED = [
    "test_distributed_magic.py",
    "test_fim_accuracy.py",
    "test_score.py",
    "test_compute_ekfac.py",  # first of the shared ekfac_ground_truth group
    "test_multinode.py",
    "test_batch_size_invariance.py",
    "test_reduce.py",
    "test_hessian.py",
]


def pytest_collection_modifyitems(items) -> None:
    order = {name: i for i, name in enumerate(FRONT_LOADED)}
    items.sort(key=lambda item: order.get(item.path.name, len(FRONT_LOADED)))


@pytest.fixture(autouse=True)
def single_gpu_hf_trainer(monkeypatch):
    """Cap HF Trainer at one GPU so it never wraps models in DataParallel.
    Setting CUDA_VISIBLE_DEVICES instead leaks into the whole pytest run.
    """
    n_gpu = TrainingArguments.n_gpu.fget
    assert n_gpu is not None
    monkeypatch.setattr(
        TrainingArguments, "n_gpu", property(lambda args: min(n_gpu(args), 1))
    )
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def model():
    """Randomly initialize a small test model."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM-3")
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)


@pytest.fixture
def dataset():
    """Create a small test dataset."""
    data = {
        "input_ids": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        "labels": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
    }
    return Dataset.from_dict(data)
