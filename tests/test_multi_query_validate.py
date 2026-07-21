import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from bergson.magic.data_stream import DataStream
from bergson.score.score_writer import MemmapSequenceScoreWriter
from bergson.validate import load_attribution_scores, per_doc_query_losses


def test_per_doc_query_losses_matches_hf_loss(model):
    """One doc per row: per-doc losses equal HF loss on each row alone."""
    data = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8],
                [9, 10, 11, 12],
                [13, 14],
            ],
        }
    )
    stream = DataStream(data, batch_size=2)

    losses = per_doc_query_losses(model, stream, num_docs=len(data))

    for i, row in enumerate(data["input_ids"]):
        x = torch.tensor([row])
        expected = model(input_ids=x, labels=x).loss
        torch.testing.assert_close(losses[i], expected, rtol=1e-4, atol=1e-5)


def test_per_doc_query_losses_packed_rows(model):
    """Packed rows: token losses are attributed to the label token's doc."""
    input_ids = [[1, 2, 3, 4, 5, 6]]
    doc_ids = [[0, 0, 0, 1, 1, 1]]
    data = Dataset.from_dict({"input_ids": input_ids, "doc_ids": doc_ids})
    stream = DataStream(data, batch_size=1, weight_shape=(2,))

    losses = per_doc_query_losses(model, stream, num_docs=2)

    x = torch.tensor(input_ids)
    logits = model(input_ids=x).logits
    token_loss = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="none")
    label_docs = torch.tensor(doc_ids)[0, 1:]
    for d in range(2):
        expected = token_loss[label_docs == d].mean()
        torch.testing.assert_close(losses[d], expected, rtol=1e-4, atol=1e-5)


def test_load_attribution_scores_score_dir(tmp_path):
    """Score dirs load all query columns and flag multi-query."""
    writer = MemmapSequenceScoreWriter(tmp_path, num_items=6, num_scores=3)
    values = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    writer(list(range(6)), values)
    writer.flush()

    scores, multi_query = load_attribution_scores(str(tmp_path))
    assert multi_query
    assert scores.shape == (6, 3)
    # No score_cfg saved, so no higher_is_better negation is applied.
    torch.testing.assert_close(scores, values)


def test_load_attribution_scores_single_column_dir(tmp_path):
    writer = MemmapSequenceScoreWriter(tmp_path, num_items=4, num_scores=1)
    writer(list(range(4)), torch.ones(4, 1))
    writer.flush()

    scores, multi_query = load_attribution_scores(str(tmp_path))
    assert not multi_query
    assert scores.shape == (4, 1)


def test_load_attribution_scores_npy_and_pt(tmp_path):
    npy_path = tmp_path / "scores.npy"
    np.save(npy_path, np.zeros((5, 2), dtype=np.float32))
    scores, multi_query = load_attribution_scores(str(npy_path))
    assert multi_query
    assert scores.shape == (5, 2)

    # 2D tensors in .pt files are per-token MAGIC scores, never multi-query.
    pt_path = tmp_path / "scores.pt"
    torch.save(torch.zeros(5, 2), pt_path)
    scores, multi_query = load_attribution_scores(str(pt_path))
    assert not multi_query


def test_weighted_ce_sum_of_means_reduction():
    """sum_of_means = per-sample token-mean, summed over batch (no /B) —
    the MAGIC/metagradients convention (arXiv 2503.13751 App. D)."""
    import torch.nn.functional as F2

    from bergson.utils.math import weighted_causal_lm_ce

    torch.manual_seed(0)
    B, T, V = 4, 6, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[2, 4:] = -100  # ragged row
    w = torch.tensor([1.0, 0.0, 1.0, 1.0])

    tok = F2.cross_entropy(
        logits[:, :-1].reshape(-1, V).float(),
        labels[:, 1:].reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(B, T - 1)
    counts = (labels[:, 1:] != -100).sum(1).clamp(min=1).float()

    ss = weighted_causal_lm_ce(
        logits, labels, example_weight=w, reduction="sum_of_means"
    )
    torch.testing.assert_close(ss, ((tok * w[:, None]).sum(1) / counts).sum())

    # Default mean reduction divides by the batch's valid-token count, so it
    # agrees with F.cross_entropy(reduction="mean") when the weights are ones.
    tm = weighted_causal_lm_ce(logits, labels, example_weight=w)
    valid = (labels[:, 1:] != -100).sum()
    torch.testing.assert_close(tm, (tok * w[:, None]).sum() / valid)


def test_metasmoothness_score():
    """Def. 2 of arXiv 2503.13751: weighted sign agreement of consecutive
    finite differences."""
    from bergson.magic.metasmoothness import metasmoothness_score

    torch.manual_seed(0)
    theta0 = torch.randn(1000)
    direction = torch.randn(1000)

    # Perfectly linear response -> score 1.
    score = metasmoothness_score(theta0, theta0 + direction, theta0 + 2 * direction)
    assert abs(score - 1.0) < 1e-5

    # Second step exactly reverses the first -> score -1.
    score = metasmoothness_score(theta0, theta0 + direction, theta0 - direction)
    assert abs(score + 1.0) < 1e-5

    # Independent random increments: Def. 2's movement weighting
    # (d = |theta_2h - theta_0|) up-weights coordinates where the increments
    # constructively interfere, so independent noise scores ~+0.45, not 0.
    r1, r2 = torch.randn(1000), torch.randn(1000)
    score = metasmoothness_score(theta0, theta0 + r1, theta0 + r1 + r2)
    assert 0.3 < score < 0.6

    # Mean-reverting second step (theta_2h drawn around theta0, not theta_h)
    # anti-correlates the increments: the metric flags it as non-smooth.
    score = metasmoothness_score(
        theta0, theta0 + torch.randn(1000), theta0 + torch.randn(1000)
    )
    assert score < -0.1

    # Zero movement -> defined as perfectly smooth.
    assert metasmoothness_score(theta0, theta0, theta0) == 1.0
