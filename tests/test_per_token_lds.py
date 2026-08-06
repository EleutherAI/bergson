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
    token dirs and 2-D ``.pt`` tensors are per-token; 1-D ``.pt`` and plain
    score dirs are per-document."""
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


def test_scores_are_per_token_from_run_config(tmp_path):
    """With a run config beside it, the flag decides rather than the shape --
    including ``attribute_tokens``, not just the deprecated ``per_token``."""
    import yaml

    from bergson.magic.cli import scores_are_per_token

    # A shape the fallback above would read as per-doc.
    torch.save(torch.zeros(4, 1), tmp_path / "scores.pt")
    cfg = {"steps": [{"magic": {"query_method": "mean", "attribute_tokens": True}}]}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    assert scores_are_per_token(str(tmp_path / "scores.pt"))
