import math
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.data import load_scores
from bergson.score.output_influence import (
    check_on_gpus,
    output_token_influence,
    query_directions,
)
from bergson.score.score import score_dataset

from .test_score import _write_query_index

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM-3"
# GPT-2 keeps its weights [in, out] in HF Conv1D modules.
CONV1D_MODEL = "hf-internal-testing/tiny-random-gpt2"


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


def _score(
    tmp_path,
    name,
    dataset,
    query_path,
    token_influence: Literal["gradient", "output"],
    model=MODEL,
    device_map_auto=False,
    nproc_per_node=1,
    **kwargs,
):
    cfg = IndexConfig(run_path=str(tmp_path / name), model=model, **kwargs)
    cfg.data.dataset = dataset
    cfg.distributed.nproc_per_node = nproc_per_node
    # Keep the advantage column.
    cfg.drop_columns = False
    score_dataset(
        cfg,
        ScoreConfig(
            query_path=str(query_path),
            token_influence=token_influence,
            device_map_auto=device_map_auto,
        ),
        PreprocessConfig(),
    )
    return load_scores(tmp_path / name)


@pytest.mark.parametrize(
    "loss_reduction, filter_modules, model_name",
    [
        ("sum", None, MODEL),
        ("mean", None, MODEL),
        ("sum", "*.mlp.*", MODEL),
        ("sum", None, CONV1D_MODEL),
    ],
)
def test_output_influence_matches_gradient_scores(
    tmp_path: Path, loss_reduction, filter_modules, model_name
):
    """Per-document output influence scores equal the gradient dot products, and
    per-token scores sum to them. The saved dataset carries the same losses."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    dataset = _dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=2)

    common = dict(
        loss_reduction=loss_reduction, filter_modules=filter_modules, model=model_name
    )
    if model_name == CONV1D_MODEL:
        # The tiny GPT-2's context is 512 tokens.
        common["token_batch_size"] = 512
    expected = _score(
        tmp_path, "grad", dataset, query_path, token_influence="gradient", **common
    )[:]
    docs = _score(
        tmp_path, "output", dataset, query_path, token_influence="output", **common
    )
    tokens = _score(
        tmp_path,
        "output_tokens",
        dataset,
        query_path,
        token_influence="output",
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
    for name in ("output", "output_tokens"):
        data = Dataset.load_from_disk(str(tmp_path / name / "data.hf"))
        assert data.column_names == grad_data.column_names
        np.testing.assert_allclose(
            data["loss"], grad_data["loss"], rtol=1e-4, atol=1e-5
        )


def test_output_influence_token_rows_are_single_loss_terms(tmp_path: Path):
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
        "output_tokens",
        dataset,
        query_path,
        token_influence="output",
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_influence_device_map_auto_matches(tmp_path: Path):
    """device_map_auto runs in one process and scores as the default path does.
    The model may fit on one GPU, so this doesn't check that it's split."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    dataset = _dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=2)

    kwargs = dict(attribute_tokens=True, token_influence="output")
    expected = _score(tmp_path, "one", dataset, query_path, **kwargs)
    split = _score(
        tmp_path,
        "auto",
        dataset,
        query_path,
        device_map_auto=True,
        nproc_per_node=2,
        **kwargs,
    )
    np.testing.assert_allclose(split[:], expected[:], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two GPUs")
def test_output_influence_split_model_matches_one_gpu(tmp_path: Path):
    """Moving each query direction to its weight's GPU gives the same rates and
    losses as running the whole model on one GPU."""
    load = partial(
        AutoModelForCausalLM.from_pretrained,
        MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    one_gpu = load(device_map={"": "cuda:0"})
    split = load(
        device_map={
            "model.embed_tokens": "cuda:0",
            "model.rotary_emb": "cuda:0",
            "model.layers.0": "cuda:0",
            "model.layers.1": "cuda:1",
            "model.norm": "cuda:1",
            "lm_head": "cuda:1",
        }
    )
    assert {p.device for p in split.parameters()} == {
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    }

    targets = GradientCollector.discover_targets(one_gpu.base_model)
    rng = torch.Generator().manual_seed(0)
    # Forward mode copies a non-contiguous direction onto its weight's device,
    # so a single query column is the case that needs the direction moved.
    query = {
        name: torch.randn(math.prod(shape), 1, generator=rng).cuda()
        for name, (_, shape, _) in targets.items()
    }
    data = Dataset.load_from_disk(_dataset(tmp_path))
    x = torch.tensor(data["input_ids"][2:3], device="cuda:0")
    y = torch.tensor(data["labels"][2:3], device="cuda:0")
    cfg = IndexConfig(run_path=str(tmp_path))

    results = [
        output_token_influence(
            model, x, y, query_directions(model, query, targets), cfg, [2.0]
        )
        for model in (one_gpu, split)
    ]
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "setting, match",
    [({"projection_dim": 16}, "projection_dim"), ({"precision": "int8"}, "int8")],
)
def test_output_influence_rejects_unsupported(tmp_path: Path, setting, match):
    cfg = IndexConfig(run_path=str(tmp_path / "scores"), **setting)
    with pytest.raises(ValueError, match=match):
        score_dataset(
            cfg,
            ScoreConfig(query_path="unused", token_influence="output"),
            PreprocessConfig(),
        )


def test_device_map_auto_rejects_gradient(tmp_path: Path):
    with pytest.raises(ValueError, match="token_influence"):
        score_dataset(
            IndexConfig(run_path=str(tmp_path / "scores")),
            ScoreConfig(query_path="unused", device_map_auto=True),
            PreprocessConfig(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_check_on_gpus_rejects_offload():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"model": "cuda:0", "lm_head": "cpu"},
    )
    with pytest.raises(ValueError, match="offloaded to cpu"):
        check_on_gpus(model)
