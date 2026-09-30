"""The rows appended to fill the last batch, and the handles that silence them."""

import torch
from datasets import Dataset

from bergson.magic.data_stream import Padding, pad_dataset_to_batch_size


def test_pad_rows_get_a_document_of_their_own():
    """Repeating the last row copies its doc_ids, so the pad rows are re-routed."""
    ds = Dataset.from_dict(
        {"input_ids": [[1, 1]] * 3, "doc_ids": [[0, 0], [1, 1], [1, 2]]}
    )
    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 4, 3, "Test", 0)

    assert (padding.rows, padding.docs, num_docs) == (1, 1, 4)
    assert list(padded["doc_ids"]) == [[0, 0], [1, 1], [1, 2], [3, 3]]

    weights = torch.ones(num_docs)
    padding.silence(weights)
    assert weights.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_rows_are_documents_without_doc_ids():
    """Every row is its own document, and an exact multiple pads nothing."""
    ds = Dataset.from_dict({"input_ids": [[1], [2], [3]]})
    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 2, 3, "Test", 0)
    assert (padding.rows, padding.docs, num_docs) == (1, 1, 4)
    assert padded["input_ids"][-1] == [3]

    padded, num_docs, padding = pad_dataset_to_batch_size(ds, 3, 3, "Test", 0)
    assert padded is ds and num_docs == 3 and not padding


def test_silence_and_trim_follow_the_layout():
    """Per-document weights own one entry; per-token weights own one row each."""
    padding = Padding(rows=3, docs=1)
    per_doc, per_token = torch.ones(5), torch.ones(5, 2)
    padding.silence(per_doc)
    padding.silence(per_token)

    assert per_doc.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert per_token[:2].all() and not per_token[2:].any()
    assert padding.trim(torch.arange(5)).tolist() == [0, 1, 2, 3]
    assert padding.trim(torch.ones(5, 2)).shape == (2, 2)
