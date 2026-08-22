import pytest
import torch

from bergson.validate import _select_filter_slice

# Scores follow the load_scores_loss_signed convention: negative reduces query
# loss, so doc 0 is the strongest proponent and doc 4 the strongest detractor.
SCORES = torch.tensor([[-5.0], [-2.0], [0.0], [1.0], [7.0]])
ALL = torch.arange(5)


def test_proponents_are_the_most_negative_scores():
    """Proponents reduce query loss, so they are the smallest signed scores."""
    got = _select_filter_slice(SCORES, ALL, 0, 2, "filter-proponents")
    assert sorted(got.tolist()) == [0, 1]


def test_detractors_are_the_most_positive_scores():
    got = _select_filter_slice(SCORES, ALL, 0, 2, "filter-detractors")
    assert sorted(got.tolist()) == [3, 4]


def test_the_two_ends_are_disjoint():
    """A sanity check that the two methods do not select the same slice."""
    pro = set(_select_filter_slice(SCORES, ALL, 0, 2, "filter-proponents").tolist())
    det = set(_select_filter_slice(SCORES, ALL, 0, 2, "filter-detractors").tolist())
    assert pro.isdisjoint(det)


def test_selection_respects_valid_indices():
    """Excluded rows are never selected, and returned ids index the full pool."""
    valid = torch.tensor([1, 2, 3, 4])  # doc 0, the top proponent, is excluded
    got = _select_filter_slice(SCORES, valid, 0, 2, "filter-proponents")
    assert sorted(got.tolist()) == [1, 2]


def test_per_query_columns_are_ranked_independently():
    scores = torch.tensor([[-1.0, 4.0], [4.0, -1.0]])
    idx = torch.arange(2)
    assert _select_filter_slice(scores, idx, 0, 1, "filter-proponents").tolist() == [0]
    assert _select_filter_slice(scores, idx, 1, 1, "filter-proponents").tolist() == [1]


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="not a tail-filter method"):
        _select_filter_slice(SCORES, ALL, 0, 1, "lds")


def test_slice_size_clamped_to_pool():
    """k larger than the pool must not error or over-select."""
    got = _select_filter_slice(SCORES, ALL, 0, 99, "filter-proponents")
    assert len(got) == 5
