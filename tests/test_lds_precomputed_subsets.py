"""LDS against a precomputed subset-retrain ground truth.

``lds_from_precomputed_subsets`` is the scoring side of a published bank
(kronfluence's ``masks.pt``/``losses.pt``, say): the prediction for a subset is
the summed score of its members, and the LDS is the per-query Spearman against
the measured subset losses. These cover the group-prediction arithmetic, the
per-query averaging, and the shape guards that catch a scores/masks mismatch.
"""

import numpy as np
import pytest
import torch

from bergson.validate import lds_from_precomputed_subsets


@pytest.fixture
def bank(tmp_path):
    """A 6-subset bank over 5 training rows and 2 queries, with losses made
    exactly linear in the summed scores so the LDS is 1.0 by construction."""
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(6, 5)).astype(np.float64)
    scores = rng.normal(size=(5, 2))

    torch.save(torch.from_numpy(masks), tmp_path / "masks.pt")
    torch.save(torch.from_numpy(masks @ scores), tmp_path / "losses.pt")
    np.save(tmp_path / "scores.npy", scores.astype(np.float32))
    return tmp_path


def test_perfectly_predictive_scores_score_one(bank):
    lds = lds_from_precomputed_subsets(
        str(bank / "scores.npy"), str(bank / "masks.pt"), str(bank / "losses.pt")
    )
    assert lds == pytest.approx(1.0)


def test_sign_flipped_scores_score_minus_one(bank):
    scores = -np.load(bank / "scores.npy")
    np.save(bank / "flipped.npy", scores)

    lds = lds_from_precomputed_subsets(
        str(bank / "flipped.npy"), str(bank / "masks.pt"), str(bank / "losses.pt")
    )
    assert lds == pytest.approx(-1.0)


def test_summary_csv_has_one_row_per_query_plus_mean(bank):
    out = bank / "lds.csv"
    lds = lds_from_precomputed_subsets(
        str(bank / "scores.npy"),
        str(bank / "masks.pt"),
        str(bank / "losses.pt"),
        summary_path=str(out),
    )

    rows = out.read_text().strip().splitlines()
    assert len(rows) == 1 + 2 + 1  # header, two queries, mean
    assert rows[-1].startswith("mean,")
    assert float(rows[-1].split(",")[1]) == pytest.approx(lds)


def test_mismatched_training_row_count_is_rejected(bank):
    np.save(bank / "short.npy", np.zeros((4, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="masks cover 5 training rows"):
        lds_from_precomputed_subsets(
            str(bank / "short.npy"), str(bank / "masks.pt"), str(bank / "losses.pt")
        )


def test_mismatched_query_count_is_rejected(bank):
    np.save(bank / "wide.npy", np.zeros((5, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="losses shape"):
        lds_from_precomputed_subsets(
            str(bank / "wide.npy"), str(bank / "masks.pt"), str(bank / "losses.pt")
        )
