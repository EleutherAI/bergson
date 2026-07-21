"""Scoring the query set in row slices must equal scoring it in one pass.

``query_batch_size`` (ScoreConfig) bounds GPU memory by scoring a slice of the
queries at a time and writing each slice's columns at ``query_offset``. That is
purely an execution strategy, so the resulting score matrix must be identical
to the unchunked one — this file pins that equivalence.
"""

import pytest
import torch

from bergson.score.score_writer import InMemorySequenceScoreWriter
from bergson.score.scorer import Scorer

MODULES = {"mod_a": 6, "mod_b": 4}
NUM_QUERIES = 5
NUM_INDEX = 7


def _query_grads(generator):
    return {
        m: torch.randn(NUM_QUERIES, dim, generator=generator)
        for m, dim in MODULES.items()
    }


def _index_grads(generator):
    return {
        m: torch.randn(NUM_INDEX, dim, generator=generator)
        for m, dim in MODULES.items()
    }


def _score_in_slices(query_grads, index_grads, slices, *, unit_normalize):
    """Score `slices` of the query set, writing each at its column offset."""
    writer = InMemorySequenceScoreWriter(NUM_INDEX, NUM_QUERIES, dtype=torch.float32)
    for start, stop in slices:
        scorer = Scorer(
            query_grads={m: g[start:stop] for m, g in query_grads.items()},
            modules=list(MODULES),
            writer=writer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            unit_normalize=unit_normalize,
            query_offset=start,
        )
        scorer(list(range(NUM_INDEX)), index_grads)
    return writer.scores


@pytest.mark.parametrize("unit_normalize", [False, True])
@pytest.mark.parametrize("slices", [[(0, 2), (2, 5)], [(0, 1), (1, 3), (3, 5)]])
def test_query_batching_matches_single_pass(slices, unit_normalize):
    """Chunked scoring reproduces the unchunked score matrix exactly."""
    g = torch.Generator().manual_seed(0)
    query_grads = _query_grads(g)
    index_grads = _index_grads(g)

    whole = _score_in_slices(
        query_grads, index_grads, [(0, NUM_QUERIES)], unit_normalize=unit_normalize
    )
    chunked = _score_in_slices(
        query_grads, index_grads, slices, unit_normalize=unit_normalize
    )

    torch.testing.assert_close(chunked, whole)


def test_query_offset_writes_the_right_columns():
    """A slice's scores land at its own columns, not column zero."""
    g = torch.Generator().manual_seed(1)
    query_grads = _query_grads(g)
    index_grads = _index_grads(g)

    whole = _score_in_slices(
        query_grads, index_grads, [(0, NUM_QUERIES)], unit_normalize=False
    )
    tail = _score_in_slices(query_grads, index_grads, [(3, 5)], unit_normalize=False)

    # Columns 3:5 carry the slice; the untouched columns stay zero.
    torch.testing.assert_close(tail[:, 3:5], whole[:, 3:5])
    assert torch.all(tail[:, :3] == 0), "slice wrote into columns it does not own"


def test_unit_normalize_is_not_computed_per_slice():
    """Normalization divides by the index gradient's full norm.

    ``unit_normalize`` scales each row by ||g|| over *all* modules. That norm is
    a property of the index gradient, not of the query slice, so it must not
    change when the query set is chunked.
    """
    g = torch.Generator().manual_seed(2)
    query_grads = _query_grads(g)
    index_grads = _index_grads(g)

    whole = _score_in_slices(
        query_grads, index_grads, [(0, NUM_QUERIES)], unit_normalize=True
    )
    chunked = _score_in_slices(
        query_grads, index_grads, [(0, 2), (2, 5)], unit_normalize=True
    )

    torch.testing.assert_close(chunked, whole)

    # Non-triviality: normalization actually changed the scores.
    raw = _score_in_slices(
        query_grads, index_grads, [(0, NUM_QUERIES)], unit_normalize=False
    )
    assert not torch.allclose(whole, raw), "unit_normalize had no effect"
