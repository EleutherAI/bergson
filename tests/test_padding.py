"""The rows appended to fill the last batch, and the handles that silence them."""

import torch
from datasets import Dataset

from bergson.magic.data_stream import Padding, pad_dataset_to_batch_size


def _chunks(doc_ids: list[list[int]]) -> Dataset:
    return Dataset.from_dict(
        {"input_ids": [[1] * len(r) for r in doc_ids], "doc_ids": doc_ids}
    )


def test_pad_rows_get_a_document_of_their_own():
    """Repeating the last row copies its doc_ids, so pad rows are re-routed."""
    ds = _chunks([[0, 0], [1, 1], [1, 2]])  # 3 chunks over 3 docs
    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 4, 3, "Test", 0)

    assert (padding.rows, padding.docs) == (1, 1)
    assert len(padded) == 4 and num_docs == 4
    # The pad row claims the synthetic doc, not doc 2 whose row it copied.
    assert padded["doc_ids"][-1] == [3, 3]
    assert padded["doc_ids"][:3] == ds["doc_ids"]

    weights = torch.ones(num_docs)
    padding.silence(weights)
    assert weights.tolist() == [1.0, 1.0, 1.0, 0.0]  # only the synthetic doc


def test_rows_are_documents_without_doc_ids():
    ds = Dataset.from_dict({"input_ids": [[1], [2], [3]]})
    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 2, 3, "Test", 0)

    assert (padding.rows, padding.docs) == (1, 1) and num_docs == 4
    assert padded["input_ids"][-1] == [3]  # the copied row, now its own doc


def test_nothing_to_pad():
    ds = Dataset.from_dict({"input_ids": [[1], [2]]})
    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 2, 2, "Test", 0)

    assert padded is ds and num_docs == 2 and not padding
    weights = torch.ones(2)
    padding.silence(weights)
    assert weights.tolist() == [1.0, 1.0]


def test_silence_and_trim_follow_the_layout():
    """Per-document weights own one entry; per-token weights own one row each."""
    padding = Padding(rows=3, docs=1)

    per_doc = torch.ones(5)
    padding.silence(per_doc)
    assert per_doc.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]

    per_token = torch.ones(5, 2)
    padding.silence(per_token)
    assert per_token[:2].all() and not per_token[2:].any()

    assert padding.trim(torch.arange(5)).tolist() == [0, 1, 2, 3]
    assert padding.trim(torch.ones(5, 2)).shape == (2, 2)
