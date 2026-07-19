"""Per-token leave-k-out re-weighting in ``validate_scores``.

The re-weighting is method-agnostic: it acts on the ``scores`` tensor and the
training ``stream.weights`` via flat indexing, so it works for any attribution
method that emits per-token scores in the ``[docs, seq_len]`` grid layout
(matching the per-token training weight shape). These tests cover the flat
indexing invariants -- that a flat subset re-weights exactly the intended token
positions, that whole-document subsets reduce to the per-document behaviour, and
that the per-document (1-D) path is unchanged.
"""

import torch
from datasets import Dataset

from bergson.magic.data_stream import DataStream
from bergson.validate import load_attribution_scores


def test_flat_reweight_per_doc_matches_direct_indexing():
    """1-D (per-doc) weights: ``view(-1)[subset]`` equals direct indexing, so
    the per-document leave-k-out path is unchanged by the flat generalization."""
    weights = torch.ones(8)
    subset = torch.tensor([1, 4, 6])

    flat = weights.clone()
    flat.view(-1)[subset] = 0.0
    direct = torch.ones(8)
    direct[subset] = 0.0

    torch.testing.assert_close(flat, direct)


def test_flat_reweight_per_token_hits_grid_positions():
    """2-D (per-token) weights: a flat subset re-weights exactly the intended
    ``doc * seq_len + token`` grid positions and nothing else."""
    n_docs, seq_len = 4, 5
    weights = torch.ones(n_docs, seq_len)
    # doc 1 token 2, doc 3 token 0, doc 3 token 4
    positions = [(1, 2), (3, 0), (3, 4)]
    subset = torch.tensor([d * seq_len + t for d, t in positions])

    weights.view(-1)[subset] = 2.0

    expected = torch.ones(n_docs, seq_len)
    for d, t in positions:
        expected[d, t] = 2.0
    torch.testing.assert_close(weights, expected)


def test_whole_doc_token_subset_reweights_full_row():
    """Selecting every token position of a document is equivalent to selecting
    that document in the per-document formulation (its whole row is set)."""
    n_docs, seq_len = 3, 6
    weights = torch.ones(n_docs, seq_len)
    doc = 1
    subset = torch.arange(doc * seq_len, (doc + 1) * seq_len)

    weights.view(-1)[subset] = 0.0

    expected = torch.ones(n_docs, seq_len)
    expected[doc] = 0.0
    torch.testing.assert_close(weights, expected)


def test_reweight_sequence_preserves_padding_rows():
    """The retrain loop's exact sequence -- fill with 1.0, zero the padding
    rows, then flat-reweight the subset -- leaves padding rows at zero, so
    batch-size padding docs never re-enter training."""
    n_docs, seq_len, pad_count = 4, 5, 1
    weights = torch.ones(n_docs, seq_len)
    weights.data[-pad_count:] = 0.0
    subset = torch.tensor([0 * seq_len + 2, 1 * seq_len + 4])

    weights.view(-1)[subset] = 0.0

    assert weights[-pad_count:].eq(0).all()
    expected_real = torch.ones(n_docs - pad_count, seq_len)
    expected_real[0, 2] = 0.0
    expected_real[1, 4] = 0.0
    torch.testing.assert_close(weights[:-pad_count], expected_real)


def test_score_sum_flat_per_token():
    """``scores.reshape(-1)[subset].sum()`` sums the selected per-token scores
    regardless of the score tensor's shape (1-D per doc or 2-D per token)."""
    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    subset = torch.tensor([0 * 4 + 1, 2 * 4 + 3])  # scores 1 and 11

    got = scores.reshape(-1)[subset].sum()

    torch.testing.assert_close(got, torch.tensor(12.0))
    # Same expression on a flat (per-doc) score vector.
    flat = torch.arange(6, dtype=torch.float32)
    torch.testing.assert_close(
        flat.reshape(-1)[torch.tensor([0, 5])].sum(), torch.tensor(5.0)
    )


def test_score_sum_flat_multi_query_per_token():
    """``scores.reshape(-1, num_queries)[subset].sum(dim=0)`` sums each query's
    scores over the selected token positions of a ``[docs, seq_len, queries]``
    grid -- the same expression that serves per-doc multi-query scores."""
    n_docs, seq_len, num_queries = 2, 3, 2
    scores = torch.arange(12, dtype=torch.float32).reshape(n_docs, seq_len, num_queries)
    # doc 0 token 1, doc 1 token 2
    subset = torch.tensor([0 * seq_len + 1, 1 * seq_len + 2])

    got = scores.reshape(-1, num_queries)[subset].sum(dim=0)

    torch.testing.assert_close(got, scores[0, 1] + scores[1, 2])
    # Per-doc multi-query [docs, queries] is served by the same expression.
    per_doc = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.testing.assert_close(
        per_doc.reshape(-1, 2)[torch.tensor([1, 3])].sum(dim=0), per_doc[1] + per_doc[3]
    )


def test_load_attribution_scores_per_token_pt(tmp_path):
    """A 2-D ``.pt`` tensor is treated as per-token MAGIC scores, never
    multi-query -- so ``validate_scores`` keeps it 2-D for token re-weighting."""
    scores = torch.randn(6, 5)
    path = tmp_path / "scores.pt"
    torch.save(scores, path)

    loaded, multi_query = load_attribution_scores(str(path))

    assert not multi_query
    assert loaded.shape == (6, 5)
    torch.testing.assert_close(loaded, scores)


def test_datastream_serves_reweighted_per_token_weights():
    """End of the pipeline: after a flat per-token re-weight, the DataStream
    serves the updated per-token ``example_weight`` for each batch, so the
    retrain actually sees the re-weighted tokens."""
    input_ids = [[1, 2, 3, 4], [5, 6, 7, 8]]
    data = Dataset.from_dict({"input_ids": input_ids})
    n_docs, seq_len = 2, 4
    stream = DataStream(data, batch_size=2, weight_shape=(n_docs, seq_len))
    # Retrains hold weights fixed, exactly as validate_scores does before
    # re-weighting a subset.
    stream.requires_grad = False

    # Down-weight doc 0's token 2 and doc 1's token 0 to zero.
    subset = torch.tensor([0 * seq_len + 2, 1 * seq_len + 0])
    stream.weights.view(-1)[subset] = 0.0

    batch = stream[0]
    ew = batch["example_weight"]
    assert ew.shape == (n_docs, seq_len)
    expected = torch.ones(n_docs, seq_len)
    expected[0, 2] = 0.0
    expected[1, 0] = 0.0
    torch.testing.assert_close(ew.cpu(), expected)


def _write_token_dir(path, ntg, values, num_scores=1):
    import numpy as np

    from bergson.score.score_writer import save_token_scores

    offsets = np.zeros(len(ntg) + 1, dtype=np.int64)
    np.cumsum(ntg, out=offsets[1:])
    total = int(offsets[-1])
    values_arr = np.asarray(values, dtype=np.float32).reshape(total, num_scores)
    save_token_scores(path, values_arr, offsets)


def test_load_token_dir_scatters_packed_to_grid(tmp_path):
    """A packed EK-FAC/TrackStar/SOURCE token dir loads as the same
    ``[docs, seq_len]`` grid as MAGIC, with tokens at positions 0..len-2."""
    _write_token_dir(tmp_path, ntg=[3, 2], values=[1, 2, 3, 4, 5])

    scores, multi_query = load_attribution_scores(str(tmp_path))

    assert not multi_query
    assert scores.shape == (2, 4)  # width = max(len-1)+1
    expected = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]])
    torch.testing.assert_close(scores, expected)


def test_scores_are_per_token_inference(tmp_path):
    """``bergson validate`` infers per-token-ness from the scores themselves:
    token dirs and 2-D ``.pt`` tensors are per-token; 1-D ``.pt``, ``.npy``,
    and plain score dirs are per-document."""
    from bergson.magic.cli import scores_are_per_token

    token_dir = tmp_path / "token_dir"
    token_dir.mkdir()
    _write_token_dir(token_dir, ntg=[3, 2], values=[1, 2, 3, 4, 5])
    assert scores_are_per_token(str(token_dir))

    pt_2d = tmp_path / "per_token.pt"
    torch.save(torch.randn(6, 5), pt_2d)
    assert scores_are_per_token(str(pt_2d))

    pt_1d = tmp_path / "per_doc.pt"
    torch.save(torch.randn(6), pt_1d)
    assert not scores_are_per_token(str(pt_1d))

    pt_col = tmp_path / "per_doc_col.pt"
    torch.save(torch.randn(6, 1), pt_col)
    assert not scores_are_per_token(str(pt_col))

    import numpy as np

    npy = tmp_path / "multi_query.npy"
    np.save(npy, np.random.randn(6, 5))
    assert not scores_are_per_token(str(npy))

    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir()
    assert not scores_are_per_token(str(plain_dir))


def test_load_token_dir_keeps_query_dim_when_multiscore(tmp_path):
    _write_token_dir(
        tmp_path, ntg=[2, 1], values=[[1, 2], [3, 4], [5, 6]], num_scores=2
    )

    scores, multi_query = load_attribution_scores(str(tmp_path))

    assert multi_query
    assert scores.shape == (2, 3, 2)
    torch.testing.assert_close(scores[0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scores[1, 0], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(scores[1, 1], torch.tensor([0.0, 0.0]))
