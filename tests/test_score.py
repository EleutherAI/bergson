import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.cli.commands import Build
from bergson.collection import collect_gradients
from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import (
    IndexConfig,
    InversionConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config_io import save_run_config
from bergson.data import column_offsets, create_index, create_token_index
from bergson.gradients import GradientProcessor
from bergson.hessians.preconditioner import (
    DensePreconditioner,
    FactoredPreconditioner,
    load_preconditioner,
)
from bergson.hessians.sharded_computation import shard_bounds
from bergson.score.score import (
    _make_split_hessian,
    create_scorer,
    get_query_grads,
    score_dataset,
)
from bergson.score.score_writer import (
    InMemorySequenceScoreWriter,
    MemmapSequenceScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import (
    get_gradient_dtype,
    tensor_to_numpy,
)


def _h_inv(path, device, power):
    """The dense inverse-Hessian matrices for a saved processor at ``path``."""
    pre = load_preconditioner(str(path), power=power, device=device)
    assert isinstance(pre, DensePreconditioner)
    return pre.h_inv


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

    # Collect gradients and build hessians using InMemoryCollector
    processor = GradientProcessor(projection_dim=16)
    build_cfg = IndexConfig(run_path=str(tmp_path / "build"), token_batch_size=1024)
    build_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    collector = InMemoryCollector(
        model=model.base_model,
        data=dataset,
        cfg=build_cfg,
        processor=processor,
        skip_hessians=False,
    )

    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        cfg=build_cfg,
    )
    computer.run_with_collector_hooks(desc="Building hessians")
    processor.save(tmp_path)

    # Produce query gradients dict
    query_grads = {
        module: torch.randn(1, shape.numel())
        for module, shape in collector.shapes().items()
    }

    target_modules = list(collector.shapes().keys())

    # Produce preconditioned query gradients
    h_inv = _h_inv(tmp_path, preprocess_device, -1)
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


def test_compute_hessian_h_inv():
    """No hessian path → no preconditioner."""

    assert load_preconditioner(None, power=-1, device=torch.device("cpu")) is None


def test_scorer_hessians(tmp_path: Path):
    """Test that Scorer applies hessians via index_transform."""

    modules = ["mod_a"]
    query_grads = {"mod_a": torch.randn(1, 4)}

    # Save a processor with H = 2*I, then load H^(-1)
    proc = GradientProcessor(hessians={"mod_a": torch.eye(4) * 2.0})
    hess_path = tmp_path / "hessian"
    proc.save(hess_path)

    h_inv = _h_inv(hess_path, torch.device("cpu"), -1)
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

    # Score without hessians
    writer_no = MemmapSequenceScoreWriter(
        tmp_path / "scores_without", 2, 1, dtype=torch.float32
    )
    scorer_no_hess = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=writer_no,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    scores_without = scorer_no_hess.score(mod_grads)

    # Hessian is 2*I, so scores should differ
    assert not torch.allclose(scores_with, scores_without)


def test_scorer_split_hessians(tmp_path: Path):
    """Split preconditioning applies H^(-1/2) to both query and index grads,
    then unit normalizes."""
    torch.manual_seed(0)
    modules = ["mod_a"]
    query_grads = {"mod_a": torch.randn(1, 4)}
    index_grads = {"mod_a": torch.randn(2, 4)}

    # Save a processor with H = 2*I
    proc = GradientProcessor(hessians={"mod_a": torch.eye(4) * 2.0})
    hess_path = tmp_path / "hessian"
    proc.save(hess_path)

    # Load H^(-1/2) for split preconditioning
    h_inv_sqrt = _h_inv(hess_path, torch.device("cpu"), -0.5)

    # Precondition query and build index_transform
    preconditioned_query = {m: query_grads[m] @ h_inv_sqrt[m] for m in modules}

    index_transform = _make_split_hessian(
        h_inv_sqrt, modules, torch.device("cpu"), torch.float32
    )

    # Score with split preconditioning (unit_normalize=True)
    scorer_hess_norm = Scorer(
        query_grads=preconditioned_query,
        modules=modules,
        writer=InMemorySequenceScoreWriter(2, 1, dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unit_normalize=True,
        index_transform=index_transform,
    )
    scores_hess_norm = scorer_hess_norm.score(index_grads)

    # Score with unit_normalize=True but no hessian
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
    h_inv = _h_inv(hess_path, torch.device("cpu"), -1)
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
    # - unit_normalize without hessian (hessian changes the space)
    # - one-sided preconditioning (different power and normalization)
    assert not torch.allclose(scores_hess_norm, scores_norm)
    assert not torch.allclose(scores_hess_norm, scores_inner_products)

    # Verify split math: H^(-1/2) applied to both sides + unit normalize
    h = h_inv_sqrt["mod_a"]
    q = query_grads["mod_a"] @ h  # preconditioned query
    g = index_grads["mod_a"] @ h  # preconditioned index
    g = g / g.norm(dim=1, keepdim=True)  # unit normalize
    expected = g @ q.T
    assert torch.allclose(scores_hess_norm, expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_precondition_at_build_not_double_applied(tmp_path: Path, model, dataset):
    """Preconditioning a query at build, then scoring, must not re-apply it.

    A query preconditioned at build (its saved ``preprocess_cfg`` carries
    ``hessian_path``) and a raw query preconditioned at score time must produce
    identical query gradients in the Scorer. If the build-time preconditioning
    were re-applied at score, the build path would be ``H^-2`` vs ``H^-1`` —
    differing by orders of magnitude, not rounding.
    """
    model = model.cuda()
    device = torch.device("cuda:0")
    dtype = torch.float32

    # Fit an autocorrelation Hessian on the data.
    hess_cfg = IndexConfig(run_path=str(tmp_path / "hessian"), token_batch_size=1024)
    hess_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    hess_proc = GradientProcessor(projection_dim=16)
    CollectorComputer(
        model=model,
        data=dataset,
        collector=InMemoryCollector(
            model=model.base_model,
            data=dataset,
            cfg=hess_cfg,
            processor=hess_proc,
            skip_hessians=False,
        ),
        cfg=hess_cfg,
    ).run_with_collector_hooks(desc="Fit Hessian")
    hess_proc.save(hess_cfg.partial_run_path)
    hessian_path = str(hess_cfg.partial_run_path)

    def build_query(name: str, preprocess_cfg: PreprocessConfig) -> Path:
        cfg = IndexConfig(run_path=str(tmp_path / name), token_batch_size=1024)
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        # Persist preprocess_cfg the way the CLI's Build command does, so score
        # can tell whether the query was already preconditioned at build.
        save_run_config(Build(cfg, preprocess_cfg), cfg.partial_run_path)
        collect_gradients(
            model=model,
            data=dataset,
            processor=GradientProcessor(projection_dim=16),
            cfg=cfg,
            preprocess_cfg=preprocess_cfg,
        )
        return cfg.partial_run_path

    # One-sided (no unit_normalize): H^-1 applied to the query only.
    query_precond = build_query(
        "q_precond", PreprocessConfig(hessian_path=hessian_path)
    )
    query_raw = build_query("q_raw", PreprocessConfig())

    def scored_query_grads(
        query_path: Path, tag: str, score_hessian: str | None
    ) -> torch.Tensor:
        scorer = create_scorer(
            path=tmp_path / f"scores_{tag}",
            data=dataset,
            score_cfg=ScoreConfig(query_path=str(query_path)),
            preprocess_cfg=PreprocessConfig(hessian_path=score_hessian),
            device=device,
            dtype=dtype,
        )
        # Concatenate the per-module dict into [total_dim, n_queries]
        return torch.cat([scorer.query_grads_t[m] for m in scorer.modules], dim=0)

    # Preconditioned once at build (guard skips re-apply) vs once at score
    # (guard applies); plus the un-preconditioned query as a non-triviality check.
    from_build = scored_query_grads(query_precond, "precond", hessian_path)
    from_score = scored_query_grads(query_raw, "raw", hessian_path)
    no_precond = scored_query_grads(query_raw, "vanilla", None)

    # H^-1 must actually change the query — otherwise the test is vacuous.
    assert not torch.allclose(
        from_score, no_precond, atol=1e-3
    ), "preconditioning is a no-op here; the equality check would be vacuous"
    assert torch.allclose(
        from_build, from_score, atol=1e-3, rtol=1e-3
    ), "build-time vs score-time preconditioning differ — query was applied twice"


def _write_factored_hessian(
    path: Path, modules: dict[str, tuple[int, int]], num_shards: int = 1, seed: int = 0
) -> None:
    """Write a synthetic factored (EKFAC) Hessian to ``path``.

    Mirrors the on-disk layout of :mod:`bergson.hessians.eigenvectors` that
    :class:`FactoredPreconditioner` reads: ``eigen_activation_sharded`` (Q_A
    ``[I, I]``), ``eigen_gradient_sharded`` (Q_G ``[O, O]``), and
    ``eigenvalue_sharded`` (λ grid ``[O, I]``), plus the ``factor_eig_a``/
    ``factor_eig_g`` vectors used by factored-Tikhonov.
    """
    g = torch.Generator().manual_seed(seed)
    subdirs = [
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_sharded",
        "factor_eig_a",
        "factor_eig_g",
    ]
    per_shard: dict[str, list[dict[str, torch.Tensor]]] = {
        sub: [{} for _ in range(num_shards)] for sub in subdirs
    }
    for name, (o, i) in modules.items():
        q_a = torch.randn(i, i, generator=g)
        q_g = torch.randn(o, o, generator=g)
        lam_a = torch.rand(i, generator=g) + 0.1
        lam_g = torch.rand(o, generator=g) + 0.1
        grid = torch.outer(lam_g, lam_a)  # [O, I]
        for r in range(num_shards):
            ia, ib = shard_bounds(i, r, num_shards)
            oa, ob = shard_bounds(o, r, num_shards)
            per_shard["eigen_activation_sharded"][r][name] = q_a[ia:ib].contiguous()
            per_shard["eigen_gradient_sharded"][r][name] = q_g[oa:ob].contiguous()
            per_shard["eigenvalue_sharded"][r][name] = grid[oa:ob].contiguous()
            per_shard["factor_eig_g"][r][name] = lam_g[oa:ob].contiguous()
            per_shard["factor_eig_a"][r][name] = lam_a.contiguous()  # replicated

    for sub, shards in per_shard.items():
        d = path / sub
        d.mkdir(parents=True, exist_ok=True)
        for r in range(num_shards):
            save_file(shards[r], str(d / f"shard_{r}.safetensors"))


def _write_query_index(
    path: Path,
    grads: dict[str, torch.Tensor],
    preprocess_cfg: PreprocessConfig,
    num_grads: int,
) -> Path:
    """Write a query gradient index (+ its preprocess_cfg) the way reduce would."""
    path.mkdir(parents=True, exist_ok=True)
    grad_sizes = {name: g.shape[1] for name, g in grads.items()}
    index = create_index(
        path, num_grads=num_grads, grad_sizes=grad_sizes, dtype=np.float32
    )
    for name, (lo, hi) in column_offsets(grad_sizes).items():
        index[:, lo:hi] = grads[name].numpy()
    index.flush()
    # Persist preprocess_cfg like the CLI does, so get_query_grads can tell
    # whether the query was already preconditioned upstream (e.g. at reduce).
    save_run_config(Build(IndexConfig(run_path=str(path)), preprocess_cfg), path)
    return path


def test_get_query_grads_token_index(tmp_path: Path):
    """get_query_grads must derive module ordering from grad_sizes for a
    per-token query index, which has no top-level "dtype" field in
    info.json."""
    grad_sizes = {"mod_a": 4, "mod_b": 3}
    num_token_grads = np.array([2, 1], dtype=np.int64)
    path = tmp_path / "token_query"
    buffer, offsets = create_token_index(
        path,
        num_token_grads=num_token_grads,
        grad_sizes=grad_sizes,
        dtype=np.float32,
    )
    total_tokens = int(offsets[-1])
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((total_tokens, 7)).astype(np.float32)
    buffer[:] = expected
    buffer.flush()
    save_run_config(Build(IndexConfig(run_path=str(path)), PreprocessConfig()), path)

    score_cfg = ScoreConfig(query_path=str(path))
    grads, preprocess_cfg = get_query_grads(score_cfg)

    assert set(grads) == set(grad_sizes)
    np.testing.assert_array_equal(grads["mod_a"].numpy(), expected[:, :4])
    np.testing.assert_array_equal(grads["mod_b"].numpy(), expected[:, 4:])
    assert score_cfg.modules == list(grad_sizes)
    assert preprocess_cfg == PreprocessConfig()


def test_score_factored_hessian_query_preconditioning(tmp_path: Path, dataset):
    """A factored (EKFAC) hessian preconditions the query in create_scorer, once.

    Preconditioning recorded at reduce time (query's saved preprocess_cfg carries
    a hessian_path) and preconditioning applied at score time must yield identical
    query gradients — and both must differ from the raw query.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    modules = {"mod_a": (4, 6), "mod_b": (5, 3)}  # (O, I)
    grad_sizes = {m: o * i for m, (o, i) in modules.items()}
    num_q = 3

    hessian_path = str(tmp_path / "hessian")
    _write_factored_hessian(Path(hessian_path), modules)

    rng = torch.Generator().manual_seed(0)
    raw = {m: torch.randn(num_q, s, generator=rng) for m, s in grad_sizes.items()}

    # One-sided H^-1 applied to the query, as a reduce step would have written.
    pre = FactoredPreconditioner.from_path(
        hessian_path, inversion_cfg=InversionConfig(), power=-1.0, device="cpu"
    )
    precond = pre.apply({k: v.clone() for k, v in raw.items()})

    q_precond = _write_query_index(
        tmp_path / "q_precond",
        precond,
        PreprocessConfig(hessian_path=hessian_path),
        num_q,
    )
    q_raw = _write_query_index(tmp_path / "q_raw", raw, PreprocessConfig(), num_q)

    def scored_query_grads(query_path: Path, tag: str, score_hessian) -> torch.Tensor:
        scorer = create_scorer(
            path=tmp_path / f"scores_{tag}",
            data=dataset,
            score_cfg=ScoreConfig(query_path=str(query_path)),
            preprocess_cfg=PreprocessConfig(hessian_path=score_hessian),
            device=device,
            dtype=dtype,
        )
        # Concatenate the per-module dict into [total_dim, n_queries]
        return torch.cat([scorer.query_grads_t[m] for m in scorer.modules], dim=0)

    from_reduce = scored_query_grads(q_precond, "precond", hessian_path)
    from_score = scored_query_grads(q_raw, "raw", hessian_path)
    no_precond = scored_query_grads(q_raw, "vanilla", None)

    assert not torch.allclose(
        from_score, no_precond, atol=1e-4
    ), "factored preconditioning is a no-op here; the equality check is vacuous"
    assert torch.allclose(
        from_reduce, from_score, atol=1e-4, rtol=1e-4
    ), "reduce-time vs score-time factored preconditioning differ"


def test_score_factored_hessian_index_transform(tmp_path: Path, dataset):
    """In split mode a factored hessian yields the H^-1/2 index transform.

    ``create_scorer`` must route the index-side transform through the factored
    ``apply`` (not identity, not the dense matmul path).
    """
    device = torch.device("cpu")
    dtype = torch.float32
    modules = {"mod_a": (4, 6), "mod_b": (5, 3)}
    grad_sizes = {m: o * i for m, (o, i) in modules.items()}
    num_q = 3

    hessian_path = str(tmp_path / "hessian")
    _write_factored_hessian(Path(hessian_path), modules)

    rng = torch.Generator().manual_seed(1)
    raw = {m: torch.randn(num_q, s, generator=rng) for m, s in grad_sizes.items()}
    q_raw = _write_query_index(tmp_path / "q_raw", raw, PreprocessConfig(), num_q)

    scorer = create_scorer(
        path=tmp_path / "scores",
        data=dataset,
        score_cfg=ScoreConfig(query_path=str(q_raw)),
        preprocess_cfg=PreprocessConfig(hessian_path=hessian_path, unit_normalize=True),
        device=device,
        dtype=dtype,
    )

    # Reference: the split factor H^-1/2 applied directly.
    half = FactoredPreconditioner.from_path(
        hessian_path, inversion_cfg=InversionConfig(), power=-0.5, device="cpu"
    )
    batch = {m: torch.randn(2, s, generator=rng) for m, s in grad_sizes.items()}
    got = scorer.index_transform({k: v.clone() for k, v in batch.items()})
    ref = half.apply({k: v.clone() for k, v in batch.items()})

    for m in grad_sizes:
        assert torch.allclose(
            got[m].cpu(), ref[m].cpu(), atol=1e-5
        ), f"factored index transform disagrees with H^-1/2 on {m}"
    assert not torch.allclose(
        got["mod_a"].cpu(), batch["mod_a"], atol=1e-4
    ), "index transform is the identity — factored branch was not selected"


def test_score_factored_hessian_rejects_projection(tmp_path: Path):
    """Scoring a projected index against a factored hessian fails fast."""
    modules = {"mod_a": (4, 6)}
    hessian_path = str(tmp_path / "hessian")
    _write_factored_hessian(Path(hessian_path), modules)

    index_cfg = IndexConfig(run_path=str(tmp_path / "out"), projection_dim=16)
    with pytest.raises(ValueError, match="projection_dim=0"):
        score_dataset(
            index_cfg,
            ScoreConfig(query_path=str(tmp_path / "q")),
            PreprocessConfig(hessian_path=hessian_path),
        )


def test_score_factored_hessian_rejects_projection_with_unit_normalize(
    tmp_path: Path,
):
    """Kronecker-factored hessian + projection + unit_normalize (cosine
    similarity) together fail fast, with a message naming the actual cause."""
    modules = {"mod_a": (4, 6)}
    hessian_path = str(tmp_path / "hessian")
    _write_factored_hessian(Path(hessian_path), modules)

    index_cfg = IndexConfig(run_path=str(tmp_path / "out"), projection_dim=16)
    with pytest.raises(ValueError, match="unit_normalize=True"):
        score_dataset(
            index_cfg,
            ScoreConfig(query_path=str(tmp_path / "q")),
            PreprocessConfig(hessian_path=hessian_path, unit_normalize=True),
        )
