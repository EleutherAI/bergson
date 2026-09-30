import math
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from bergson import (
    CollectorComputer,
    GradientProcessor,
    InMemoryCollector,
    collect_gradients,
)
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.config.config import DataConfig
from bergson.data import load_scores, span_rows
from bergson.score.score import score_dataset
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.score.scorer import Scorer
from bergson.spans import span_row_bounds
from bergson.utils.utils import get_gradient_dtype

from .test_output_influence import MODEL, _query

SPANS = [[0, 3, 7], [0, 4]]


@pytest.fixture
def span_dataset():
    """Two documents cut into spans, one of 10 tokens and one of 8."""
    ids = [list(range(1, 11)), list(range(11, 19))]
    return Dataset.from_dict(
        {
            "input_ids": ids,
            "labels": ids,
            "length": [len(x) for x in ids],
            "span_starts": SPANS,
        }
    )


# ---------------------------------------------------------------------------
# Span geometry
# ---------------------------------------------------------------------------


def test_span_row_bounds_follow_the_token_influence():
    """A span owns position rows directly, but loss rows shifted one earlier.
    A span whose tokens carry no loss row keeps an empty range rather than
    dropping out and changing the store's shape."""
    starts = np.array([0, 3, 7])
    np.testing.assert_array_equal(
        span_row_bounds(starts, 9, shift=0), [[0, 3], [3, 7], [7, 9]]
    )
    np.testing.assert_array_equal(
        span_row_bounds(starts, 9, shift=1), [[0, 2], [2, 6], [6, 9]]
    )
    np.testing.assert_array_equal(
        span_row_bounds(np.array([0, 1, 5]), 9, shift=1), [[0, 0], [0, 4], [4, 9]]
    )


def test_span_rows_validates_and_skips_short_documents():
    """Starts that leave leading tokens unattributed are rejected, and a
    document with no gradient row gets no span row, as with tokens."""
    ds = Dataset.from_dict(
        {"input_ids": [[1] * 6, [1]], "length": [6, 1], "span_starts": [[0, 3], [0]]}
    )
    np.testing.assert_array_equal(span_rows(ds, "span_starts")[0], [2, 0])

    gapped = ds.remove_columns("span_starts").add_column(
        "span_starts", [[2, 5], [0]], new_fingerprint="gap"
    )
    with pytest.raises(ValueError, match="belong to no span"):
        span_rows(gapped, "span_starts")


def test_attribute_tokens_and_span_column_conflict():
    """The two granularity switches are mutually exclusive."""
    with pytest.raises(ValueError, match="per-token or per-span"):
        IndexConfig(
            run_path="unused",
            attribute_tokens=True,
            data=DataConfig(span_column="span_starts"),
        )


def test_num_rows_counts_spans(tmp_path: Path, model, span_dataset):
    """An autocorrelation Gram over span gradients is normalized by their count."""
    collector = GradientCollector(
        model.base_model,
        data=span_dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path / "rows"),
            data=DataConfig(span_column="span_starts"),
        ),
        processor=GradientProcessor(),
        skip_index=True,
    )
    assert collector.num_rows(span_dataset) == sum(len(s) for s in SPANS)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _collect(model, ds, targets, run_path, **kwargs):
    cfg = IndexConfig(
        run_path=run_path, token_batch_size=1024, loss_reduction="sum", **kwargs
    )
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    collector = InMemoryCollector(
        model=model.base_model,
        data=ds,
        cfg=cfg,
        processor=GradientProcessor(),
        target_modules=targets,
        attention_cfgs={},
    )
    CollectorComputer(
        model=model, data=ds, collector=collector, cfg=cfg
    ).run_with_collector_hooks()
    return collector


def test_span_grads_sum_their_token_grads(tmp_path: Path, model, span_dataset):
    """Each span gradient is the sum of its positions', so a document's spans
    still sum to its sequence gradient."""
    model = model.float()
    targets = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    tok = _collect(
        model, span_dataset, targets, str(tmp_path / "tok"), attribute_tokens=True
    )
    spans = _collect(
        model,
        span_dataset,
        targets,
        str(tmp_path / "span"),
        data=DataConfig(span_column="span_starts"),
    )
    seq = _collect(model, span_dataset, targets, str(tmp_path / "seq"))

    assert tok.builder is not None and spans.builder is not None
    tok_off, span_off = tok.builder.offsets, spans.builder.offsets
    bounds = [
        span_row_bounds(np.array(s), n - 1)
        for s, n in zip(SPANS, span_dataset["length"])
    ]

    for name, seq_grads in seq.gradients.items():
        tok_grads, span_grads = tok.gradients[name], spans.gradients[name]
        for doc in range(len(span_dataset)):
            rows = tok_grads[tok_off[doc] : tok_off[doc + 1]]
            for k, (lo, hi) in enumerate(bounds[doc]):
                torch.testing.assert_close(
                    span_grads[span_off[doc] + k].float(),
                    rows[lo:hi].sum(dim=0).float(),
                    atol=1e-3,
                    rtol=1e-3,
                    msg=f"{name}: doc {doc} span {k} is not its tokens' sum",
                )
            torch.testing.assert_close(
                span_grads[span_off[doc] : span_off[doc + 1]].sum(dim=0).float(),
                seq_grads[doc].float(),
                atol=1e-3,
                rtol=1e-3,
                msg=f"{name}: doc {doc}'s spans do not sum to its sequence gradient",
            )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _score(model, ds, tmp_path, name, **kwargs):
    """Score ``ds`` against one fixed query and return the store."""
    cfg = IndexConfig(run_path=str(tmp_path / name), token_batch_size=1024, **kwargs)
    processor = GradientProcessor()
    shapes = GradientCollector(
        model.base_model, data=ds, cfg=cfg, processor=processor
    ).shapes()
    modules = list(shapes)

    torch.manual_seed(0)
    query_grads = {m: torch.randn(1, math.prod(shapes[m])) for m in modules}
    dtype = get_gradient_dtype(model)
    span_column = cfg.data.span_column
    path = tmp_path / f"{name}_scores"
    if span_column is None:
        writer = MemmapTokenScoreWriter.from_dataset(path, ds, 1, dtype=dtype)
    else:
        writer = MemmapTokenScoreWriter.from_spans(
            path, ds, 1, span_column, dtype=dtype
        )

    scorer = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=writer,
        device=torch.device("cpu"),
        dtype=dtype,
        attribute_tokens=cfg.attribute_tokens,
    )
    collect_gradients(model=model, data=ds, processor=processor, cfg=cfg, scorer=scorer)
    writer.flush()
    return load_scores(path)


def test_span_scores_sum_their_token_scores(tmp_path: Path, model, span_dataset):
    """Scoring spans gives the same numbers as summing per-token scores over
    each span, which is what the store's row bounds describe."""
    model = model.float()
    tokens = _score(model, span_dataset, tmp_path, "tok", attribute_tokens=True)
    spans = _score(
        model,
        span_dataset,
        tmp_path,
        "span",
        data=DataConfig(span_column="span_starts"),
    )
    assert spans.is_written()
    assert spans.spans is not None
    np.testing.assert_array_equal(
        spans.spans, span_rows(span_dataset, "span_starts")[1]
    )

    for doc in range(len(span_dataset)):
        rows = np.asarray(tokens[tokens.offsets[doc] : tokens.offsets[doc + 1]])
        lo, hi = spans.offsets[doc], spans.offsets[doc + 1]
        for (row_lo, row_hi), got in zip(spans.spans[lo:hi], np.asarray(spans[lo:hi])):
            want = rows[row_lo:row_hi].sum(axis=0)
            np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)


def test_span_store_to_grid_repeats_each_span(tmp_path: Path):
    """``to_grid`` lays a span's score over every row it covers, so a filter
    ranking token rows removes whole spans."""
    ds = Dataset.from_dict(
        {"input_ids": [[1] * 10, [1] * 8], "length": [10, 8], "span_starts": SPANS}
    )
    writer = MemmapTokenScoreWriter.from_spans(tmp_path / "s", ds, 1, "span_starts")
    writer(list(range(len(ds))), torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    writer.flush()

    grid = load_scores(tmp_path / "s").to_grid()
    assert grid.shape == (2, 10)
    torch.testing.assert_close(grid[0], torch.tensor([1.0, 1, 1, 2, 2, 2, 2, 3, 3, 0]))
    torch.testing.assert_close(grid[1], torch.tensor([4.0, 4, 4, 4, 5, 5, 5, 0, 0, 0]))


# ---------------------------------------------------------------------------
# End to end, through both token influences
# ---------------------------------------------------------------------------


def _disk_dataset(tmp_path: Path) -> str:
    """Documents with a masked prompt and span starts, saved for score_dataset."""
    rng = np.random.default_rng(0)
    ids, labels, starts = [], [], []
    for length, cuts in ((7, [0, 2, 5]), (5, [0, 3]), (9, [0, 1, 4]), (6, [0, 4])):
        doc = rng.integers(2, 100, size=length).tolist()
        ids.append(doc)
        labels.append([-100, -100] + doc[2:])
        starts.append(cuts)
    path = str(tmp_path / "train.hf")
    Dataset.from_dict(
        {"input_ids": ids, "labels": labels, "span_starts": starts}
    ).save_to_disk(path)
    return path


@pytest.mark.parametrize("token_influence", ["gradient", "output"])
def test_span_store_matches_token_store_end_to_end(tmp_path: Path, token_influence):
    """Through the whole score pipeline, each span row equals the sum of the
    per-token rows its recorded bounds cover -- including the one-row shift the
    ``output`` token influence puts between a position and the loss it feeds."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    dataset = _disk_dataset(tmp_path)
    query_path = _query(tmp_path, model, dataset, num_queries=2)

    tokens = _score_dataset(
        tmp_path, "tok", dataset, query_path, token_influence, attribute_tokens=True
    )
    spans = _score_dataset(
        tmp_path,
        "span",
        dataset,
        query_path,
        token_influence,
        data=DataConfig(dataset=dataset, span_column="span_starts"),
    )
    assert spans.is_written() and spans.spans is not None

    for doc in range(len(tokens)):
        rows = np.asarray(tokens[tokens.offsets[doc] : tokens.offsets[doc + 1]])
        lo, hi = spans.offsets[doc], spans.offsets[doc + 1]
        for (row_lo, row_hi), got in zip(spans.spans[lo:hi], np.asarray(spans[lo:hi])):
            np.testing.assert_allclose(
                got, rows[row_lo:row_hi].sum(axis=0), rtol=1e-4, atol=1e-5
            )


def _score_dataset(tmp_path, name, dataset, query_path, token_influence, **kwargs):
    """Run the score command over ``dataset`` and load the store it writes."""
    cfg = IndexConfig(run_path=str(tmp_path / name), model=MODEL, **kwargs)
    cfg.data.dataset = dataset
    cfg.distributed.nproc_per_node = 1
    score_dataset(
        cfg,
        ScoreConfig(query_path=str(query_path), token_influence=token_influence),
        PreprocessConfig(),
    )
    return load_scores(tmp_path / name)
