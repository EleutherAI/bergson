import csv
import json

import pytest
import torch

from bergson.validate import (
    _report_filter_baseline,
    _select_filter_slice,
    load_and_validate_subsets_match,
)

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


def _cfg(tmp_path, **kwargs):
    from bergson.magic.config import MagicConfig

    return MagicConfig(run_path=str(tmp_path / "run"), model="gpt2", **kwargs)


def _bank(tmp_path, subsets, name="bank", base=True, **cfg_overrides):
    """A bank directory holding only what the baseline checks read."""
    from bergson.config.config_io import save_run_config

    root = tmp_path / name
    if base:
        (root / "retrained" / "base").mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "subsets.json", "w") as f:
        json.dump(subsets, f)
    save_run_config(_cfg(tmp_path, **cfg_overrides), root)
    return root


def test_bank_of_the_same_removal_size_is_accepted(tmp_path):
    """The LDS default partitions the pool, so chunk sizes differ by one."""
    root = _bank(tmp_path, [[0, 1, 2], [3, 4], [5, 6]])
    subsets = load_and_validate_subsets_match(_cfg(tmp_path), [root], num_filtered=2)
    assert [s.tolist() for s in subsets] == [[0, 1, 2], [3, 4], [5, 6]]


def test_bank_of_a_different_removal_size_is_rejected(tmp_path):
    root = _bank(tmp_path, [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="set subset_fraction to match"):
        load_and_validate_subsets_match(_cfg(tmp_path), [root], num_filtered=10)


def test_bank_without_a_base_model_is_rejected(tmp_path):
    """Loss changes are measured against the bank's own no-leave-out model."""
    root = _bank(tmp_path, [[0, 1]], base=False)
    with pytest.raises(AssertionError, match="not valid"):
        load_and_validate_subsets_match(_cfg(tmp_path), [root], num_filtered=2)


@pytest.mark.parametrize(
    "field,value", [("subset_weight", 0.5), ("exclude_zero_scores", True)]
)
def test_bank_trained_with_different_settings_is_rejected(tmp_path, field, value):
    root = _bank(tmp_path, [[0, 1]], **{field: value})
    with pytest.raises(ValueError, match=field):
        load_and_validate_subsets_match(_cfg(tmp_path), [root], num_filtered=2)


def test_averaged_banks_must_share_their_removal_sets(tmp_path):
    a = _bank(tmp_path, [[0, 1]], name="a")
    b = _bank(tmp_path, [[2, 3]], name="b")
    with pytest.raises(AssertionError, match="doesn't match others"):
        load_and_validate_subsets_match(_cfg(tmp_path), [a, b], num_filtered=2)


def test_filter_is_ranked_against_the_random_draws(tmp_path):
    """Rank 1 is the largest loss increase; the filter beats 2 of 3 draws."""
    _report_filter_baseline(
        str(tmp_path),
        "filter-proponents",
        torch.tensor([0.5]),
        torch.tensor([[0.1], [0.2], [0.9]]),
        k=2,
        source="retrained here",
    )
    with open(tmp_path / "filter_summary.csv") as f:
        row = list(csv.DictReader(f))[0]
    assert int(row["rank"]) == 2
    assert int(row["random_n"]) == 3
    assert float(row["filter_change"]) == pytest.approx(0.5)
    assert float(row["random_mean"]) == pytest.approx(0.4)


def _e2e_shared(tmp_path) -> dict:
    from datasets import Dataset

    train = Dataset.from_dict({"text": [f"the cat sat on mat {i}" for i in range(8)]})
    train.save_to_disk(tmp_path / "train")
    query = Dataset.from_dict({"text": ["a dog ran", "birds fly high"]})
    query.save_to_disk(tmp_path / "query")
    return {
        "model": "trl-internal-testing/tiny-Phi3ForCausalLM",
        "batch_size": 2,
        "num_epochs": 1,
        "num_subsets": 2,
        "subset_fraction": 0.25,
        "data": {"dataset": str(tmp_path / "train")},
        "query": {"dataset": str(tmp_path / "query")},
        "distributed": {"nproc_per_node": 1},
    }


def test_a_filter_run_compares_against_random_filters(tmp_path):
    from bergson.cli.commands import Magic, Validate

    shared = _e2e_shared(tmp_path)
    Magic.from_dict(shared | {"run_path": str(tmp_path / "magic")}).execute()
    run = tmp_path / "filter"
    Validate.from_dict(
        shared
        | {
            "run_path": str(run),
            "scores": str(tmp_path / "magic" / "scores"),
            "method": "filter-proponents",
        }
    ).execute()

    with open(run / "random_filter.csv") as f:
        random_rows = list(csv.DictReader(f))
    assert {r["subset"] for r in random_rows} == {"0", "1"}
    with open(run / "filter_summary.csv") as f:
        summary = list(csv.DictReader(f))
    assert [r["query"] for r in summary] == ["0", "1"]
    assert all(1 <= int(r["rank"]) <= 3 for r in summary)


def test_a_bank_gives_the_same_random_filters_as_retraining_them(tmp_path):
    """The bank's models are those retrains, so the two must agree."""
    from bergson.cli.commands import Magic, Validate

    shared = _e2e_shared(tmp_path)
    Magic.from_dict(shared | {"run_path": str(tmp_path / "magic")}).execute()
    scores = str(tmp_path / "magic" / "scores")

    bank = tmp_path / "bank"
    Validate.from_dict(
        shared | {"run_path": str(bank), "scores": scores, "save_models": True}
    ).execute()

    def random_changes(name, **overrides) -> list[str]:
        Validate.from_dict(
            shared
            | {
                "run_path": str(tmp_path / name),
                "scores": scores,
                "method": "filter-proponents",
                "subsets": str(bank / "subsets.json"),
            }
            | overrides
        ).execute()
        with open(tmp_path / name / "random_filter.csv") as f:
            return [r["loss_change"] for r in csv.DictReader(f)]

    here = random_changes("here")
    from_bank = random_changes("from_bank", retrained_dir=str(bank))
    assert [round(float(x), 5) for x in from_bank] == [round(float(x), 5) for x in here]
