"""Tests for datasets that are already tokenized."""

import pytest
from datasets import Dataset

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline

GPT2 = "openai-community/gpt2"


def _save_pretokenized(tmp_path, columns):
    path = tmp_path / "pretok"
    Dataset.from_dict(columns).save_to_disk(str(path))
    return str(path)


def _run(tmp_path, columns, **data_kwargs):
    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        model=GPT2,
        token_batch_size=1024,
        data=DataConfig(dataset=_save_pretokenized(tmp_path, columns), **data_kwargs),
    )
    ds, _ = setup_data_pipeline(cfg)
    return ds


@pytest.mark.parametrize("truncation", [True, False])
def test_length_derived_from_input_ids(tmp_path, truncation):
    """A pre-tokenized dataset without a `length` column gets one derived."""
    input_ids = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    ds = _run(tmp_path, {"input_ids": input_ids}, truncation=truncation)

    assert ds["length"] == [3, 2, 4]
    assert ds["input_ids"] == input_ids


def test_existing_length_preserved(tmp_path):
    """An explicit `length` column is left alone."""
    ds = _run(tmp_path, {"input_ids": [[1, 2, 3], [4, 5]], "length": [3, 2]})

    assert ds["length"] == [3, 2]


def test_labels_preserved(tmp_path):
    """Label masks on a pre-tokenized dataset survive preprocessing."""
    labels = [[-100, 2, 3], [-100, 5]]
    ds = _run(tmp_path, {"input_ids": [[1, 2, 3], [4, 5]], "labels": labels})

    assert ds["labels"] == labels
    assert ds["length"] == [3, 2]
