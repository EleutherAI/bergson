import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import create_index
from bergson.gradients import GradientProcessor
from bergson.process_grads import get_trackstar_preconditioner
from bergson.score.score import _make_split_preconditioner
from bergson.score.score_writer import (
    InMemorySequenceScoreWriter,
    MemmapSequenceScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import (
    get_gradient_dtype,
    tensor_to_numpy,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_gradients_query(tmp_path: Path, dataset):
    # Create index for uncompressed gradients from a large model.
    config = AutoConfig.from_pretrained(
        "EleutherAI/pythia-1.4b", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    collector = GradientCollector(
        model.base_model, data=dataset, cfg=IndexConfig(run_path=str(tmp_path))
    )
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    dataset.save_to_disk(str(tmp_path / "query_ds" / "data.hf"))
    create_index(
        tmp_path / "query_ds",
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
        with_structure=False,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bergson",
            "score",
            "test_score_e2e",
            "--projection_dim",
            "0",
            "--query_path",
            str(tmp_path / "query_ds"),
            "--model",
            "EleutherAI/pythia-1.4b",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:8]",
            "--truncation",
            "--token_batch_size",
            "256",
            "--skip_preconditioners",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_score(tmp_path: Path, model, dataset):
    model = model.cuda()
    processor = GradientProcessor(projection_dim=16)

    # Step 1: Reduce query gradients using InMemoryCollector
    reduce_index_cfg = IndexConfig(
        run_path=str(tmp_path / "reduce"), token_batch_size=1024
    )
    reduce_index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    query_collector = InMemoryCollector(
        model=model.base_model,
        data=dataset,
        cfg=reduce_index_cfg,
        processor=processor,
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )

    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=query_collector,
        cfg=reduce_index_cfg,
    )
    computer.run_with_collector_hooks(desc="Reducing query gradients")

    query_grads = query_collector.gradients
    modules = list(query_collector.shapes().keys())

    # Step 2: Score using InMemoryCollector with scorer
    score_dtype = get_gradient_dtype(model)
    score_writer = InMemorySequenceScoreWriter(len(dataset), 1, dtype=score_dtype)
    scorer = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=score_writer,
        device=torch.device("cuda:0"),
        dtype=score_dtype,
    )

    index_processor = GradientProcessor(projection_dim=16)
    index_cfg = IndexConfig(run_path=str(tmp_path / "index"), token_batch_size=1024)
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    index_collector = InMemoryCollector(
        model=model.base_model,
        data=dataset,
        cfg=index_cfg,
        processor=index_processor,
        scorer=scorer,
    )

    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=index_collector,
        cfg=index_cfg,
    )
    computer.run_with_collector_hooks(desc="Scoring")

    scores = index_collector.scores
    assert scores is not None
    assert scores.shape == (len(dataset), 1)
    assert torch.isfinite(scores).all()
    assert not torch.allclose(scores, torch.zeros_like(scores))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_precondition_ds(tmp_path: Path, model, dataset):
    model = model.cuda()
    preprocess_device = torch.device("cuda:0")

    # Collect gradients and build preconditioners using InMemoryCollector
    processor = GradientProcessor(projection_dim=16)
    build_cfg = IndexConfig(run_path=str(tmp_path / "build"), token_batch_size=1024)
    build_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    collector = InMemoryCollector(
        model=model.base_model,
        data=dataset,
        cfg=build_cfg,
        processor=processor,
    )

    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        cfg=build_cfg,
    )
    computer.run_with_collector_hooks(desc="Building preconditioners")
    processor.save(tmp_path)

    # Produce query gradients dict
    query_grads = {
        module: torch.randn(1, shape.numel())
        for module, shape in collector.shapes().items()
    }

    target_modules = list(collector.shapes().keys())

    # Produce preconditioned query gradients
    h_inv = get_trackstar_preconditioner(
        str(tmp_path), device=preprocess_device, power=-1
    )
    preconditioned = {
        name: (query_grads[name].to(preprocess_device) @ h_inv[name]).cpu()
        for name in target_modules
    }

    # Compare against unpreconditioned — should differ
    for name in target_modules:
        vanilla = query_grads[name].to(preprocess_device).cpu()
        assert not torch.allclose(preconditioned[name], vanilla)


def test_memmap_score_writer_bfloat16(tmp_path: Path):
    """MemmapSequenceScoreWriter writes and reads bfloat16."""
    num_items = 10
    num_scores = 3

    writer = MemmapSequenceScoreWriter(
        tmp_path, num_items, num_scores, dtype=torch.bfloat16
    )

    # Create some test scores in bfloat16
    scores_batch1 = torch.tensor(
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.bfloat16
    )
    scores_batch2 = torch.tensor(
        [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5], [13.5, 14.5, 15.5]],
        dtype=torch.bfloat16,
    )

    # Write scores
    writer([0, 1], scores_batch1)
    writer([5, 6, 7], scores_batch2)
    writer.flush()

    # Verify the files exist
    assert (tmp_path / "scores.bin").exists()
    assert (tmp_path / "info.json").exists()

    # Read back and verify
    with open(tmp_path / "info.json", "r") as f:
        info = json.load(f)

    assert info["num_items"] == num_items
    assert info["num_scores"] == num_scores
    assert "bfloat16" in info["dtype"]["formats"][0]

    # Check written flags
    assert writer.scores["written_0"][0]
    assert writer.scores["written_0"][1]
    assert not writer.scores["written_0"][2]  # Not written
    assert writer.scores["written_0"][5]
    assert writer.scores["written_0"][6]
    assert writer.scores["written_0"][7]

    # Check score values (convert back to compare)
    expected_batch1 = tensor_to_numpy(scores_batch1)
    expected_batch2 = tensor_to_numpy(scores_batch2)

    np.testing.assert_array_equal(
        writer.scores["score_0"][[0, 1]].view(bfloat16), expected_batch1[:, 0]
    )
    np.testing.assert_array_equal(
        writer.scores["score_1"][[0, 1]].view(bfloat16), expected_batch1[:, 1]
    )
    np.testing.assert_array_equal(
        writer.scores["score_2"][[0, 1]].view(bfloat16), expected_batch1[:, 2]
    )

    np.testing.assert_array_equal(
        writer.scores["score_0"][[5, 6, 7]].view(bfloat16), expected_batch2[:, 0]
    )


def test_memmap_score_writer_float32(tmp_path: Path):
    """MemmapSequenceScoreWriter writes float32 scores."""
    num_items = 5
    num_scores = 2

    writer = MemmapSequenceScoreWriter(
        tmp_path, num_items, num_scores, dtype=torch.float32
    )

    scores = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
    writer([0, 1], scores)
    writer.flush()

    # Verify values
    np.testing.assert_array_almost_equal(
        writer.scores["score_0"][[0, 1]], np.array([1.5, 3.5], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        writer.scores["score_1"][[0, 1]], np.array([2.5, 4.5], dtype=np.float32)
    )


def test_compute_preconditioner_h_inv():
    """Test that get_trackstar_preconditioner returns empty dict for None path."""

    # No path → empty dict
    result = get_trackstar_preconditioner(None, device=torch.device("cpu"), power=-1)
    assert result == {}


def test_scorer_preconditioners(tmp_path: Path):
    """Test that Scorer applies preconditioners via index_transform."""

    modules = ["mod_a"]
    query_grads = {"mod_a": torch.randn(1, 4)}

    # Save a processor with H = 2*I, then load H^(-1)
    proc = GradientProcessor(preconditioners={"mod_a": torch.eye(4) * 2.0})
    precond_path = tmp_path / "preconditioner"
    proc.save(precond_path)

    h_inv = get_trackstar_preconditioner(
        str(precond_path), device=torch.device("cpu"), power=-1
    )
    preconditioned_query = {m: query_grads[m] @ h_inv[m] for m in modules}

    writer = MemmapSequenceScoreWriter(
        tmp_path / "scores_with", 2, 1, dtype=torch.float32
    )
    scorer = Scorer(
        query_grads=preconditioned_query,
        modules=modules,
        writer=writer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Score with preconditioned query
    mod_grads = {"mod_a": torch.randn(2, 4)}
    scores_with = scorer.score(mod_grads)

    # Score without preconditioners
    writer_no = MemmapSequenceScoreWriter(
        tmp_path / "scores_without", 2, 1, dtype=torch.float32
    )
    scorer_no_precond = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=writer_no,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    scores_without = scorer_no_precond.score(mod_grads)

    # Preconditioner is 2*I, so scores should differ
    assert not torch.allclose(scores_with, scores_without)


def test_scorer_split_preconditioners(tmp_path: Path):
    """Split preconditioning applies H^(-1/2) to both query and index grads,
    then unit normalizes."""
    torch.manual_seed(0)
    modules = ["mod_a"]
    query_grads = {"mod_a": torch.randn(1, 4)}
    index_grads = {"mod_a": torch.randn(2, 4)}

    # Save a processor with H = 2*I
    proc = GradientProcessor(preconditioners={"mod_a": torch.eye(4) * 2.0})
    precond_path = tmp_path / "preconditioner"
    proc.save(precond_path)

    # Load H^(-1/2) for split preconditioning
    h_inv_sqrt = get_trackstar_preconditioner(
        str(precond_path), device=torch.device("cpu"), power=-0.5
    )

    # Precondition query and build index_transform
    preconditioned_query = {m: query_grads[m] @ h_inv_sqrt[m] for m in modules}

    index_transform = _make_split_preconditioner(
        h_inv_sqrt, modules, torch.device("cpu"), torch.float32
    )

    # Score with split preconditioning (unit_normalize=True)
    scorer_precond_norm = Scorer(
        query_grads=preconditioned_query,
        modules=modules,
        writer=InMemorySequenceScoreWriter(2, 1, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unit_normalize=True,
        index_transform=index_transform,
    )
    scores_precond_norm = scorer_precond_norm.score(index_grads)

    # Score with unit_normalize=True but no preconditioner
    scorer_norm = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=InMemorySequenceScoreWriter(2, 1, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unit_normalize=True,
    )
    scores_norm = scorer_norm.score(index_grads)

    # Score with one-sided preconditioning (query only, no index_transform)
    h_inv = get_trackstar_preconditioner(
        str(precond_path), device=torch.device("cpu"), power=-1
    )
    one_sided_query = {m: query_grads[m] @ h_inv[m] for m in modules}
    scorer_inner_products = Scorer(
        query_grads=one_sided_query,
        modules=modules,
        writer=InMemorySequenceScoreWriter(2, 1, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unit_normalize=False,
    )
    scores_inner_products = scorer_inner_products.score(index_grads)

    # Split preconditioning should differ from both:
    # - unit_normalize without preconditioner (preconditioner changes the space)
    # - one-sided preconditioning (different power and normalization)
    assert not torch.allclose(scores_precond_norm, scores_norm)
    assert not torch.allclose(scores_precond_norm, scores_inner_products)

    # Verify split math: H^(-1/2) applied to both sides + unit normalize
    h = h_inv_sqrt["mod_a"]
    q = query_grads["mod_a"] @ h  # preconditioned query
    g = index_grads["mod_a"] @ h  # preconditioned index
    g = g / g.norm(dim=1, keepdim=True)  # unit normalize
    expected = g @ q.T
    assert torch.allclose(scores_precond_norm, expected, atol=1e-6)
