import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.data import load_scores
from bergson.score.score import score_dataset

from .test_score import _write_query_index

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _dataset(tmp_path: Path) -> str:
    """Documents with a masked prompt, so some loss terms don't exist."""
    rng = np.random.default_rng(0)
    ids, labels = [], []
    for length in (7, 5, 9, 6):
        doc = rng.integers(2, 100, size=length).tolist()
        ids.append(doc)
        labels.append([-100, -100] + doc[2:])
    path = str(tmp_path / "train.hf")
    Dataset.from_dict(
        {"input_ids": ids, "labels": labels, "advantage": [1.0, -0.5, 2.0, 0.25]}
    ).save_to_disk(path)
    return path


def _query(tmp_path: Path, model, dataset: str, num_queries: int) -> Path:
    shapes = GradientCollector(
        model.base_model,
        data=Dataset.load_from_disk(dataset),
        cfg=IndexConfig(run_path=str(tmp_path)),
    ).shapes()
    rng = torch.Generator().manual_seed(0)
    grads = {
        name: torch.randn(num_queries, math.prod(shape), generator=rng)
        for name, shape in shapes.items()
    }
    return _write_query_index(
        tmp_path / "query", grads, PreprocessConfig(), num_queries
    )


def _score(tmp_path, name, dataset, query_path, forward_mode: bool, **kwargs):
    cfg = IndexConfig(run_path=str(tmp_path / name), model=MODEL, **kwargs)
    cfg.data.dataset = dataset
    cfg.distributed.nproc_per_node = 1
    # Keep the advantage column.
    cfg.drop_columns = False
    score_dataset(
        cfg,
        ScoreConfig(query_path=str(query_path), forward_mode=forward_mode),
        PreprocessConfig(),
    )
    return load_scores(tmp_path / name)


@pytest.mark.parametrize(
    "loss_reduction, filter_modules",
    [("sum", None), ("mean", None), ("sum", "*.mlp.*")],
)
def test_forward_mode_matches_gradient_scores(
    tmp_path: Path, loss_reduction, filter_modules
):
    """Per-document forward-mode scores equal the gradient dot products, and
    per-token scores sum to them. The saved dataset carries the same losses."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    dataset = _dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=2)

    common = dict(loss_reduction=loss_reduction, filter_modules=filter_modules)
    expected = _score(
        tmp_path, "grad", dataset, query_path, forward_mode=False, **common
    )[:]
    docs = _score(tmp_path, "fwd", dataset, query_path, forward_mode=True, **common)
    tokens = _score(
        tmp_path,
        "fwd_tokens",
        dataset,
        query_path,
        forward_mode=True,
        attribute_tokens=True,
        **common,
    )

    np.testing.assert_allclose(docs[:], expected, rtol=1e-4, atol=1e-5)
    assert tokens.offsets is not None
    summed = np.stack(
        [
            tokens[tokens.offsets[d] : tokens.offsets[d + 1]].sum(0)
            for d in range(len(tokens))
        ]
    )
    np.testing.assert_allclose(summed, expected, rtol=1e-4, atol=1e-5)

    grad_data = Dataset.load_from_disk(str(tmp_path / "grad" / "data.hf"))
    for name in ("fwd", "fwd_tokens"):
        data = Dataset.load_from_disk(str(tmp_path / name / "data.hf"))
        assert data.column_names == grad_data.column_names
        np.testing.assert_allclose(
            data["loss"], grad_data["loss"], rtol=1e-4, atol=1e-5
        )


def test_forward_mode_token_rows_are_single_loss_terms(tmp_path: Path):
    """Row t is the query's dot product with the gradient of the loss on token
    t + 1 alone, times the document's advantage, and rows without a label are
    zero."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    )
    dataset = _dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=1)
    tokens = _score(
        tmp_path,
        "fwd_tokens",
        dataset,
        query_path,
        forward_mode=True,
        attribute_tokens=True,
    )

    data = Dataset.load_from_disk(dataset)
    query = torch.from_numpy(np.fromfile(query_path / "gradients.bin", "<f4"))
    # The query's modules, in the order its columns are laid out.
    modules = GradientCollector(
        model.base_model, data=data, cfg=IndexConfig(run_path=str(tmp_path))
    ).shapes()
    weights = [model.base_model.get_submodule(m).weight for m in modules]

    # Document 1 is the shortest, so it's padded in the batch.
    for doc in (1, 2):
        x = torch.tensor([data[doc]["input_ids"]])
        labels = torch.tensor(data[doc]["labels"][1:])
        losses = F.cross_entropy(model(x).logits[0, :-1], labels, reduction="none")
        rows = tokens[tokens.offsets[doc] : tokens.offsets[doc + 1]][:, 0]
        assert len(rows) == len(labels)

        for t in range(len(labels)):
            if labels[t] == -100:
                assert rows[t] == 0
                continue
            grads = torch.autograd.grad(losses[t], weights, retain_graph=True)
            expected = torch.dot(torch.cat([g.flatten() for g in grads]), query)
            expected = expected * data[doc]["advantage"]
            assert rows[t] == pytest.approx(expected.item(), rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    "setting, match",
    [({"projection_dim": 16}, "projection_dim"), ({"precision": "int8"}, "int8")],
)
def test_forward_mode_rejects_unsupported(tmp_path: Path, setting, match):
    cfg = IndexConfig(run_path=str(tmp_path / "scores"), **setting)
    with pytest.raises(ValueError, match=match):
        score_dataset(
            cfg, ScoreConfig(query_path="unused", forward_mode=True), PreprocessConfig()
        )
