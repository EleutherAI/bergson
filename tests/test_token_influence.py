import math
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
from bergson.gradients import LayerAdapter
from bergson.score.score import score_dataset

from .test_score import _write_query_index

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"
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
    token_influence: Literal["gradient", "output", "input"],
    model=MODEL,
    **kwargs,
):
    cfg = IndexConfig(run_path=str(tmp_path / name), model=model, **kwargs)
    cfg.data.dataset = dataset
    cfg.distributed.nproc_per_node = 1
    # Keep the advantage column.
    cfg.drop_columns = False
    score_dataset(
        cfg,
        ScoreConfig(query_path=str(query_path), token_influence=token_influence),
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


@pytest.mark.parametrize(
    "loss_reduction, model_name",
    [("sum", MODEL), ("mean", CONV1D_MODEL)],
)
def test_input_influence_token_rows_are_embedding_scale_derivatives(
    tmp_path: Path, loss_reduction, model_name
):
    """Row t is the derivative of the query's dot product with the document's
    gradient with respect to scaling token t's embedding, found here by
    differentiating the gradient a second time. GPT-2 uses F.layer_norm and ties
    its output layer to the embeddings that bergson leaves trainable."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    dataset = _dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=2)
    tokens = _score(
        tmp_path,
        "input_tokens",
        dataset,
        query_path,
        token_influence="input",
        attribute_tokens=True,
        loss_reduction=loss_reduction,
        model=model_name,
        # The tiny GPT-2's context is 512 tokens.
        token_batch_size=512,
    )

    assert tokens.offsets is not None
    data = Dataset.load_from_disk(dataset)
    query = torch.from_numpy(
        np.fromfile(query_path / "gradients.bin", "<f4").reshape(2, -1)
    )
    modules = GradientCollector(
        model.base_model, data=data, cfg=IndexConfig(run_path=str(tmp_path))
    ).shapes()
    layers = [model.base_model.get_submodule(m) for m in modules]
    weights = [layer.weight for layer in layers]

    for doc in (1, 2):
        x = torch.tensor([data[doc]["input_ids"]])
        labels = torch.tensor(data[doc]["labels"][1:])
        scale = torch.zeros(x.shape[1], requires_grad=True)
        embeds = model.get_input_embeddings()(x).detach() * (1 + scale)[:, None]
        losses = F.cross_entropy(
            model(inputs_embeds=embeds).logits[0, :-1], labels, reduction="none"
        )
        loss = losses.sum() * data[doc]["advantage"]
        if loss_reduction == "mean":
            loss = loss / (labels != -100).sum()
        grads = torch.autograd.grad(loss, weights, create_graph=True)
        # Query blocks are laid out [out, in] whatever the layer stores.
        flat = torch.cat(
            [
                (g.T if LayerAdapter.weight_transposed(layer) else g).flatten()
                for g, layer in zip(grads, layers)
            ]
        )

        rows = tokens[tokens.offsets[doc] : tokens.offsets[doc + 1]]
        assert len(rows) == x.shape[1] - 1
        for q in range(2):
            (expected,) = torch.autograd.grad(flat @ query[q], scale, retain_graph=True)
            # The last token predicts no label.
            assert expected[-1] == 0
            np.testing.assert_allclose(
                rows[:, q], expected[:-1].numpy(), rtol=1e-4, atol=1e-5
            )


@pytest.mark.parametrize(
    "token_influence, setting, match",
    [
        ("output", {"projection_dim": 16}, "projection_dim"),
        ("output", {"precision": "int8"}, "int8"),
        ("input", {"attribute_tokens": False}, "attribute_tokens"),
    ],
)
def test_token_influence_rejects_unsupported(
    tmp_path: Path, token_influence, setting, match
):
    cfg = IndexConfig(run_path=str(tmp_path / "scores"), **setting)
    with pytest.raises(ValueError, match=match):
        score_dataset(
            cfg,
            ScoreConfig(query_path="unused", token_influence=token_influence),
            PreprocessConfig(),
        )
