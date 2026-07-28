"""Reuse of method-independent per-subset query losses in ``evaluate_retrained``.

Evaluating a pre-saved leave-k-out bank re-runs every banked model on the query
set to get each subset's query-loss diff. Those losses depend only on the bank
and query set, not the attribution scores, so they are cached under the bank
and reused across every method scored on the same bank/query -- the second run
must not touch a single banked model, yet must produce identical correlations.
"""

import copy
import json

import numpy as np
import pytest
import torch
from datasets import Dataset

import bergson.validate as validate
from bergson.config.config import DataConfig
from bergson.magic.config import MagicConfig
from bergson.validate import bank_loss_cache_key, evaluate_retrained

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"

# Bank drops these train doc ids per subset; scores must cover the max id.
SUBSETS = [[0, 1], [2, 3], [4]]
NUM_DOCS = 5


def _build_bank(tmp_path, model):
    """A tiny bank: a base model plus one distinctly-perturbed model per subset."""
    root = tmp_path / "bank"
    models_root = root / "retrained"
    for i, name in enumerate(["base", *[f"subset_{j}" for j in range(len(SUBSETS))]]):
        m = copy.deepcopy(model)
        with torch.no_grad():
            # Distinct perturbation per model so subset losses differ.
            next(m.parameters()).add_(0.05 * (i + 1))
        m.save_pretrained(models_root / name)
    with open(root / "subsets.json", "w") as f:
        json.dump(SUBSETS, f)
    return root


def _query_dataset(tmp_path):
    """Four short query docs saved to disk for ``load_data_string``."""
    ds = Dataset.from_dict(
        {"text": ["the cat sat", "a dog ran fast", "birds fly high", "fish swim deep"]}
    )
    path = tmp_path / "query"
    ds.save_to_disk(path)
    return path, len(ds)


def _run_cfg(tmp_path, run_name, query_path, batch_size=2):
    return MagicConfig(
        run_path=str(tmp_path / run_name),
        model=MODEL,
        batch_size=batch_size,
        precision="fp32",
        query=DataConfig(
            dataset=str(query_path), split="train", prompt_column="text", chunk_length=0
        ),
    )


def _read_validation(run_path):
    rows = []
    with open(run_path / "validation.csv") as f:
        header = f.readline().strip().split(",")
        for line in f:
            rows.append(dict(zip(header, line.strip().split(","))))
    return rows


def test_bank_loss_cache_key_pins_query_and_load_settings():
    base = MagicConfig(run_path="x", model=MODEL, batch_size=2)
    name0 = bank_loss_cache_key(base, multi_query=True, num_subsets=3)

    # Same identity -> same file (run_path is irrelevant to the losses).
    name_same = bank_loss_cache_key(
        MagicConfig(run_path="y", model=MODEL, batch_size=2),
        multi_query=True,
        num_subsets=3,
    )
    assert name0 == name_same

    # Any identity change -> different file, so a different query/model/batch
    # never reuses these losses.
    changed = [
        dict(run_path="x", model=MODEL, batch_size=4),
        dict(run_path="x", model="other-model", batch_size=2),
    ]
    for kw in changed:
        name = bank_loss_cache_key(MagicConfig(**kw), multi_query=True, num_subsets=3)
        assert name != name0, kw

    # Mode and subset count pin the tensor shape, so they key too.
    name_single = bank_loss_cache_key(base, multi_query=False, num_subsets=3)
    name_n = bank_loss_cache_key(base, multi_query=True, num_subsets=4)
    assert len({name0, name_single, name_n}) == 3


@pytest.mark.parametrize("multi_query", [True, False])
def test_evaluate_retrained_reuses_cached_bank_losses(
    tmp_path, model, monkeypatch, multi_query
):
    root = _build_bank(tmp_path, model)
    query_path, n_query = _query_dataset(tmp_path)

    cols = n_query if multi_query else 1
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((NUM_DOCS, cols)).astype(np.float32)
    score_path = tmp_path / "scores.npy"
    np.save(score_path, scores)

    # First run: cold cache, evaluates the bank and writes the cache.
    cfg1 = _run_cfg(tmp_path, "run1", query_path)
    evaluate_retrained(cfg1, str(root), score_path=str(score_path))

    cache_dir = root / "query_loss_cache"
    cache_files = list(cache_dir.glob("losses_*.pt"))
    assert len(cache_files) == 1, "first run must persist exactly one cache file"

    expected_name = bank_loss_cache_key(cfg1, multi_query, len(SUBSETS))
    assert cache_files[0].name == expected_name

    rows1 = _read_validation(tmp_path / "run1")

    # Second run: any attempt to load a banked model means the cache was missed.
    def _fail_load(*args, **kwargs):
        raise AssertionError("banked model loaded despite a valid loss cache")

    monkeypatch.setattr(validate, "_load_banked_model", _fail_load)

    cfg2 = _run_cfg(tmp_path, "run2", query_path)
    evaluate_retrained(cfg2, str(root), score_path=str(score_path))

    rows2 = _read_validation(tmp_path / "run2")

    # Identical diffs and score sums => identical correlations, cache-independent.
    assert len(rows1) == len(rows2)
    for a, b in zip(rows1, rows2):
        assert a["diff"] == b["diff"]
        assert a["score_sum"] == b["score_sum"]


def test_different_query_does_not_reuse_losses(tmp_path, model):
    """A different query set keys a different cache file, never the first's."""
    root = _build_bank(tmp_path, model)
    query_path, n_query = _query_dataset(tmp_path)
    scores = np.random.default_rng(2).standard_normal((NUM_DOCS, n_query)).astype("f4")
    score_path = tmp_path / "scores.npy"
    np.save(score_path, scores)

    cfg_a = _run_cfg(tmp_path, "run_a", query_path)
    evaluate_retrained(cfg_a, str(root), score_path=str(score_path))

    # Same bank, different query split -> a distinct cache file is written.
    cfg_b = _run_cfg(tmp_path, "run_b", query_path)
    cfg_b.query.split = "train[:3]"
    name_a = bank_loss_cache_key(cfg_a, True, len(SUBSETS))
    name_b = bank_loss_cache_key(cfg_b, True, len(SUBSETS))
    assert name_a != name_b
