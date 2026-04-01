import subprocess
from pathlib import Path

import pytest
import torch

from bergson import (
    CollectorComputer,
    DataConfig,
    GradientProcessor,
    IndexConfig,
    InMemoryCollector,
    PreprocessConfig,
    collect_gradients,
)
from bergson.build import build
from bergson.data import load_gradient_dataset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduce_cli(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "reduce",
            "test_reduce_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--aggregation",
            "mean",
            "--unit_normalize",
            "--skip_preconditioners",
            "--token_batch_size",
            "1024",
        ],
        cwd=tmp_path,
        capture_output=True,
        # Get strings instead of bytes
        text=True,
    )

    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"

    # Load the gradient index
    index_cfg = IndexConfig(run_path=str(tmp_path / "test_reduce_e2e"))
    ds = load_gradient_dataset(Path(index_cfg.run_path), structured=False)
    assert len(ds) == 1

    grads = torch.tensor(ds["gradients"][:])
    assert not torch.isnan(grads).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_programmatic_reduce(tmp_path: Path):
    index_cfg = IndexConfig(
        run_path=str(tmp_path / "reduction"),
        data=DataConfig(truncation=True, split="train[:100]"),
        model="EleutherAI/pythia-14m",
        skip_preconditioners=True,
        token_batch_size=1024,
    )

    build(index_cfg, PreprocessConfig(aggregation="mean"))

    # Assert 1-row reduction exists at the tmp_path
    ds = load_gradient_dataset(Path(index_cfg.run_path), structured=False)
    assert len(ds) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduce_with_preconditioning(tmp_path: Path, model, dataset):
    # Step 1: build an index WITH preconditioners
    build_cfg = IndexConfig(run_path=str(tmp_path / "build"), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=build_cfg,
    )

    # Step 2: reduce with preconditioning pointing at the built index
    preprocess_cfg = PreprocessConfig(
        aggregation="mean", preconditioner_path=str(build_cfg.partial_run_path)
    )
    reduce_index_cfg = IndexConfig(
        run_path=str(tmp_path / "reduce_precond"),
        token_batch_size=1024,
        skip_preconditioners=True,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=reduce_index_cfg,
        preprocess_cfg=preprocess_cfg,
    )

    ds_out = load_gradient_dataset(reduce_index_cfg.partial_run_path, structured=False)
    assert len(ds_out) == 1
    grads = torch.tensor(ds_out["gradients"][:])
    assert not torch.isnan(grads).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_in_memory_reduce(tmp_path: Path, model, dataset):
    model.cuda()
    cfg = IndexConfig(
        run_path=str(tmp_path / "reduction"),
        skip_preconditioners=True,
        token_batch_size=1024,
    )
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = PreprocessConfig(
        aggregation="mean",
    )

    collector = InMemoryCollector(
        model=model.base_model,
        cfg=cfg,
        processor=GradientProcessor(),
        data=dataset,
        preprocess_cfg=preprocess_cfg,
        attention_cfgs={},
    )

    CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        cfg=cfg,
    ).run_with_collector_hooks(desc="In-memory reduce")

    assert all(len(collector.gradients[name]) == 1 for name in collector.gradients)
