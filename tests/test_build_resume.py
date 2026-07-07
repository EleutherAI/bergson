"""Resume support for the `build` command (plain per-row gradient index).

1. ``IndexResumeTracker`` reports the written-row prefix from ``written.bin``.
2. ``build()`` raises for the stateful paths (Hessian / aggregation) that
   cannot be resumed by skipping rows.
3. End-to-end parity on GPU: a build interrupted after a prefix of batches,
   then resumed on the remainder, yields a gradient index (and loss column)
   identical to an uninterrupted build.
"""

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM

from bergson import GradientProcessor, collect_gradients
from bergson.build import build
from bergson.builder import IndexResumeTracker
from bergson.config import IndexConfig, PreprocessConfig
from bergson.config.config import HessianConfig
from bergson.data import allocate_batches, load_gradients
from bergson.distributed import skip_completed_batches


def test_index_resume_tracker(tmp_path):
    """Reports the written-row prefix; absent file → nothing written."""
    assert IndexResumeTracker(tmp_path, 8).leading_written_count([[0, 1]]) == 0

    written = np.memmap(
        str(tmp_path / "written.bin"), dtype=np.bool_, mode="w+", shape=(8,)
    )
    written[:] = False
    written[[0, 1, 2, 3]] = True
    written.flush()

    tracker = IndexResumeTracker(tmp_path, 8)
    assert tracker.leading_written_count([[0, 1], [2, 3], [4, 5]]) == 2
    assert tracker.leading_written_count([[0, 1], [4, 5], [2, 3]]) == 1


def test_build_resume_rejects_stateful_paths(tmp_path):
    """Hessian fitting and aggregation accumulate state → resume must raise."""
    model = "hf-internal-testing/tiny-random-gpt2"

    with pytest.raises(AssertionError, match="Hessian"):
        build(
            IndexConfig(run_path=str(tmp_path / "h"), model=model, resume=True),
            PreprocessConfig(),
            HessianConfig(method="autocorrelation"),
        )

    with pytest.raises(AssertionError, match="aggregat"):
        build(
            IndexConfig(run_path=str(tmp_path / "a"), model=model, resume=True),
            PreprocessConfig(aggregation="mean"),
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
def test_build_resume_parity(tmp_path):
    """A build interrupted after a prefix of batches, then resumed on the
    remainder, produces a gradient index identical to an uninterrupted build."""
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda()

    index_ds = _make_ds(8, offset=10)
    full_batches = allocate_batches(index_ds["length"][:], 12)
    assert len(full_batches) >= 4, "need several batches to split"

    def run(run_dir, batches, resume):
        cfg = IndexConfig(
            run_path=str(run_dir), token_batch_size=12, projection_dim=4, resume=resume
        )
        cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        collect_gradients(
            model=model,
            data=index_ds,
            processor=GradientProcessor(projection_dim=4),
            cfg=cfg,
            batches=batches,
        )
        return cfg

    # Reference: uninterrupted build over all batches.
    ref_cfg = run(tmp_path / "ref", full_batches, resume=False)
    ref_grads = np.asarray(load_gradients(ref_cfg.partial_run_path, structured=False))
    ref_loss = np.asarray(
        load_from_disk(str(ref_cfg.partial_run_path / "data.hf"))["loss"]
    )

    # Crash after the first half of the batches, then resume on the remainder.
    run_dir = tmp_path / "run"
    k = len(full_batches) // 2
    run(run_dir, full_batches[:k], resume=False)

    tracker = IndexResumeTracker(IndexConfig(run_path=str(run_dir)).partial_run_path, 8)
    skip = tracker.leading_written_count(full_batches)
    assert skip == k, f"expected {k} written batches, got {skip}"
    remaining = skip_completed_batches(
        full_batches, tracker, device=torch.device("cuda:0")
    )
    resumed_cfg = run(run_dir, remaining, resume=True)

    resumed_grads = np.asarray(
        load_gradients(resumed_cfg.partial_run_path, structured=False)
    )
    resumed_loss = np.asarray(
        load_from_disk(str(resumed_cfg.partial_run_path / "data.hf"))["loss"]
    )

    np.testing.assert_allclose(resumed_grads, ref_grads, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(resumed_loss, ref_loss, rtol=1e-4, atol=1e-5)
