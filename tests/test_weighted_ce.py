"""Reduction semantics of :func:`weighted_causal_lm_ce`.

The ``"mean"`` reduction is documented as the mean over every valid token in
the batch. The weighted and unweighted branches compute it separately, so the
tests below pin them to each other and to an explicit reference.
"""

import pytest
import torch

from bergson.utils.math import weighted_causal_lm_ce

IGNORE = -100


def _batch(B: int, T: int, V: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    return logits, labels


def _reference_mean(logits, labels, ignore_index=IGNORE):
    """Mean CE over all valid shifted tokens, computed the obvious way."""
    shift_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction="mean"
    )


@pytest.mark.parametrize("B", [1, 2])
def test_mean_matches_reference(B):
    """Both branches equal an explicit token-mean over the batch."""
    logits, labels = _batch(B, 8, 50, seed=1)
    expected = _reference_mean(logits, labels)

    torch.testing.assert_close(weighted_causal_lm_ce(logits, labels), expected)
    torch.testing.assert_close(
        weighted_causal_lm_ce(logits, labels, example_weight=torch.ones(B)), expected
    )


def test_ignored_tokens_excluded_from_denominator():
    """Padding must not dilute the mean: the denominator counts valid tokens."""
    logits, labels = _batch(4, 8, 50, seed=2)
    labels[1, 3:] = IGNORE
    labels[2, :] = IGNORE  # a fully-padded row, as empty documents produce

    expected = _reference_mean(logits, labels)

    torch.testing.assert_close(weighted_causal_lm_ce(logits, labels), expected)
    torch.testing.assert_close(
        weighted_causal_lm_ce(logits, labels, example_weight=torch.ones(4)), expected
    )


def test_all_tokens_ignored_does_not_divide_by_zero():
    """An entirely-padded batch yields a finite (zero) loss, not NaN."""
    logits, labels = _batch(2, 8, 50, seed=3)
    labels[:] = IGNORE

    loss = weighted_causal_lm_ce(logits, labels, example_weight=torch.ones(2))

    assert torch.isfinite(loss), f"expected a finite loss, got {loss}"
    torch.testing.assert_close(loss, torch.zeros(()))


def test_weights_scale_their_own_rows():
    """Doubling one row's weight adds that row's contribution once more."""
    logits, labels = _batch(2, 8, 50, seed=4)

    base = weighted_causal_lm_ce(logits, labels, example_weight=torch.ones(2))
    bumped = weighted_causal_lm_ce(
        logits, labels, example_weight=torch.tensor([2.0, 1.0])
    )
    row0 = weighted_causal_lm_ce(
        logits, labels, example_weight=torch.tensor([1.0, 0.0])
    )

    # Denominator is shared (it counts valid tokens, not weight mass), so the
    # difference is exactly row 0's weighted contribution.
    torch.testing.assert_close(bumped - base, row0)


def test_unknown_reduction_raises():
    """An unrecognized ``reduction`` must fail loudly.

    It used to fall through to the mean branch, so ``reduction="sum"`` (the
    default of ``IndexConfig.loss_reduction``, which
    ``setup_model_and_peft`` forwards here) and even ``reduction="garbage"``
    silently returned the *mean*.
    """
    logits, labels = _batch(2, 8, 50, seed=5)

    with pytest.raises(AssertionError, match="Unknown reduction"):
        weighted_causal_lm_ce(logits, labels, reduction="garbage")
    with pytest.raises(AssertionError, match="Unknown reduction"):
        weighted_causal_lm_ce(
            logits, labels, example_weight=torch.ones(2), reduction="garbage"
        )


def test_sum_reduction_is_a_real_sum():
    """``reduction="sum"`` sums over every valid token, matching the index
    path (``fwd_bwd_factory`` with ``denoms=1.0`` then ``losses.sum()``).

    Before the fix this returned the mean -- off by a factor of n_tokens.
    """
    logits, labels = _batch(2, 8, 50, seed=6)
    labels[1, 5:] = IGNORE

    got = weighted_causal_lm_ce(logits, labels, reduction="sum")

    shift_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    expected = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=IGNORE, reduction="sum"
    )
    torch.testing.assert_close(got, expected)

    n_valid = int((labels[:, 1:] != IGNORE).sum())
    mean = weighted_causal_lm_ce(logits, labels, reduction="mean")
    assert n_valid > 1
    torch.testing.assert_close(got, mean * n_valid)
