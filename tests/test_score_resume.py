"""Resume support for the `score` command.

Covers three layers:
  1. The writer-level primitive: ``leading_written_count`` reports the written
     prefix and survives a reopen (the on-disk state a resumed run reads).
  2. The FSDP-safety invariant: ``skip_completed_batches`` skips the *same* number
     of batches on every rank (global MIN), so remaining work stays equal-length
     and per-step collectives can't deadlock.
  3. End-to-end crash/resume parity on GPU: a run killed mid-scoring, resumed,
     produces scores identical to an uninterrupted run.
"""

import os
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

from bergson import GradientProcessor, collect_gradients
from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.distributed import skip_completed_batches
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    MemmapTokenScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.utils import get_gradient_dtype


def test_sequence_writer_leading_written_count(tmp_path):
    """Reports the fully-written prefix; survives reopen (resume reads disk)."""
    writer = MemmapSequenceScoreWriter(tmp_path, 10, 2, dtype=torch.float32)
    writer([0, 1], torch.ones(2, 2))
    writer([2, 3], torch.ones(2, 2))
    writer.flush()

    # First two batches fully written, third not.
    assert writer.leading_written_count([[0, 1], [2, 3], [4, 5]]) == 2
    # A gap stops the prefix even if a later batch is written.
    assert writer.leading_written_count([[0, 1], [8, 9], [2, 3]]) == 1
    # A partially-written batch is not counted (row 4 unwritten).
    assert writer.leading_written_count([[0, 1], [3, 4]]) == 1

    # Reopen at the same path: persisted written flags are visible.
    reopened = MemmapSequenceScoreWriter(tmp_path, 10, 2, dtype=torch.float32)
    assert reopened.leading_written_count([[0, 1], [2, 3], [4, 5]]) == 2


def _token_dataset(n: int, seq_len: int = 4) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[i + 1] * seq_len for i in range(n)],
            "length": [seq_len for _ in range(n)],
        }
    )


def test_token_writer_leading_written_count(tmp_path):
    """Token writer tracks completion via written.bin and survives reopen."""
    data = _token_dataset(6)
    writer = MemmapTokenScoreWriter(tmp_path, data, 2, dtype=torch.float32)

    # Each row has seq_len - 1 = 3 token grads; write rows 0,1 then 2.
    writer([0, 1], torch.ones(6, 2))
    writer([2], torch.ones(3, 2))
    writer.flush()

    assert (tmp_path / "written.bin").exists()
    assert writer.leading_written_count([[0, 1], [2], [3, 4]]) == 2
    assert writer.leading_written_count([[0, 1], [4], [2]]) == 1

    reopened = MemmapTokenScoreWriter(tmp_path, data, 2, dtype=torch.float32)
    assert reopened.leading_written_count([[0, 1], [2], [3, 4]]) == 2


def test_skip_completed_batches_single_process(tmp_path):
    """Without a process group, skip = this writer's own leading count."""
    writer = MemmapSequenceScoreWriter(tmp_path, 10, 1, dtype=torch.float32)
    writer([0, 1], torch.ones(2, 1))
    writer([2, 3], torch.ones(2, 1))
    writer.flush()

    batches = [[0, 1], [2, 3], [4, 5], [6, 7]]
    remaining = skip_completed_batches(batches, writer, device=torch.device("cpu"))
    assert remaining == [[4, 5], [6, 7]]


class _StubWriter:
    """Reports a fixed leading-written count (per-rank), isolating the global
    MIN reduction in ``skip_completed_batches`` from the mmap machinery."""

    def __init__(self, count: int):
        self._count = count

    def leading_written_count(self, batches):
        return self._count


def _fsdp_skip_worker(rank: int, world_size: int, return_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=2)
    )
    try:
        # Ragged progress: rank 0 completed 1 batch, rank 1 completed 3.
        # The global MIN must make BOTH skip exactly 1 → equal remainders.
        writer = _StubWriter(1 if rank == 0 else 3)
        batches = [[2 * b, 2 * b + 1] for b in range(5)]
        remaining = skip_completed_batches(batches, writer, device=torch.device("cpu"))
        return_dict[rank] = len(remaining)
    finally:
        dist.destroy_process_group()


def test_skip_completed_batches_fsdp_safe_equal_skip():
    """Core FSDP-safety invariant: every rank skips the same count (global MIN),
    so remaining batch lists are equal-length and collectives stay in lockstep."""
    world_size = 2
    mgr = mp.Manager()
    return_dict = mgr.dict()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_fsdp_skip_worker, args=(r, world_size, return_dict))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"rank worker exited with {p.exitcode}"

    # Both ranks skip MIN(1, 3) = 1 of 5 → 4 batches remain on each.
    assert return_dict[0] == return_dict[1] == 4


def test_load_scores_legacy_structured_format(tmp_path):
    """``load_scores`` still reads the pre-flat structured layout (interleaved
    ``score_i``/``written_i`` fields). TODO(Lucia, 2026-10): drop with the shim."""
    import json

    from bergson.data import load_scores

    num_items, num_scores = 4, 2
    struct = {
        "names": ["score_0", "written_0", "score_1", "written_1"],
        "formats": ["<f4", "|b1", "<f4", "|b1"],
        "offsets": [0, 4, 8, 12],
        "itemsize": 16,
    }
    mmap = np.memmap(
        str(tmp_path / "scores.bin"),
        dtype=np.dtype(struct),  # type: ignore[arg-type]
        mode="w+",
        shape=(num_items,),
    )
    mmap["score_0"] = [1.0, 2.0, 3.0, 4.0]
    mmap["score_1"] = [5.0, 6.0, 7.0, 8.0]
    mmap.flush()
    with (tmp_path / "info.json").open("w") as f:
        json.dump(
            {"num_items": num_items, "num_scores": num_scores, "dtype": struct}, f
        )

    scores = load_scores(tmp_path)
    assert not scores.flat
    np.testing.assert_array_equal(
        np.asarray(scores[:]),
        np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(scores.get(slice(None), 1)), [5.0, 6.0, 7.0, 8.0]
    )


def _make_ds(n: int, offset: int = 0, seq_len: int = 5) -> Dataset:
    rows = [[1 + (i + offset) % 50] * seq_len for i in range(n)]
    return Dataset.from_dict(
        {
            "input_ids": rows,
            "labels": rows,
            "attention_mask": [[1] * seq_len for _ in range(n)],
            "length": [seq_len for _ in range(n)],
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_score_resume_parity(tmp_path):
    """A scoring run interrupted after a prefix of batches, then resumed on the
    remainder, produces scores identical to an uninterrupted run."""
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda()

    query_ds = _make_ds(2)
    index_ds = _make_ds(8, offset=10)
    device = torch.device("cuda:0")
    dtype = get_gradient_dtype(model)

    # Mean-reduced query gradients (one query vector).
    query_cfg = IndexConfig(run_path=str(tmp_path / "q"), token_batch_size=64)
    query_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    query_collector = InMemoryCollector(
        model=model.base_model,
        data=query_ds,
        cfg=query_cfg,
        processor=GradientProcessor(projection_dim=16),
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )
    CollectorComputer(
        model=model, data=query_ds, collector=query_collector, cfg=query_cfg
    ).run_with_collector_hooks(desc="query")
    query_grads = query_collector.gradients
    modules = list(query_collector.shapes().keys())

    full_batches = allocate_batches(index_ds["length"][:], 12)
    assert len(full_batches) >= 4, "need several batches to split"

    def score(run_dir, batches):
        index_cfg = IndexConfig(run_path=str(run_dir), token_batch_size=12)
        index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        writer = MemmapSequenceScoreWriter(
            run_dir / "scores", len(index_ds), len(query_grads[modules[0]]), dtype=dtype
        )
        scorer = Scorer(
            query_grads=query_grads,
            modules=modules,
            writer=writer,
            device=device,
            dtype=dtype,
        )
        collect_gradients(
            model=model,
            data=index_ds,
            processor=GradientProcessor(projection_dim=16),
            cfg=index_cfg,
            scorer=scorer,
            batches=batches,
        )
        writer.flush()
        return writer

    # Reference: uninterrupted run over all batches.
    ref = score(tmp_path / "ref", full_batches)
    ref_scores = np.array(ref.scores).copy()

    # Crash: only the first half of the batches were processed before dying.
    k = len(full_batches) // 2
    score(tmp_path / "run", full_batches[:k])

    # Resume: reopen the partial scores, skip the written prefix, finish the rest.
    resumed = MemmapSequenceScoreWriter(
        tmp_path / "run" / "scores",
        len(index_ds),
        len(query_grads[modules[0]]),
        dtype=dtype,
    )
    skip = resumed.leading_written_count(full_batches)
    assert skip == k, f"expected {k} written batches, got {skip}"
    remaining = full_batches[skip:]
    scorer = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=resumed,
        device=device,
        dtype=dtype,
    )
    resume_cfg = IndexConfig(run_path=str(tmp_path / "run2"), token_batch_size=12)
    resume_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    collect_gradients(
        model=model,
        data=index_ds,
        processor=GradientProcessor(projection_dim=16),
        cfg=resume_cfg,
        scorer=scorer,
        batches=remaining,
    )
    resumed.flush()

    # Every row scored, and identical to the uninterrupted reference.
    assert bool(resumed.written.all())
    np.testing.assert_allclose(
        np.array(resumed.scores), ref_scores, rtol=1e-4, atol=1e-5
    )
