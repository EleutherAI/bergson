"""The unit one gradient or score row attributes to."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from datasets import Dataset

from .config.config import IndexConfig


def compute_num_token_grads(data: Dataset) -> np.ndarray:
    """Number of per-token gradient rows stored per example.

    Position ``t``'s row is ``g_t (x) a_t``. ``g_t`` is generally nonzero even
    at prompt / masked positions, because the completion-token losses backprop
    through them via causal attention -- masking is applied to the *loss*, not
    to the gradient. Storing every position therefore makes the per-token rows
    sum to the per-document gradient. Only the final position (which predicts
    nothing after ``logits[:, :-1]``) and right-padding are excluded, so
    ``num_token_grads = length - 1`` regardless of the label mask.

    Documents shorter than 2 tokens contribute **zero** rows: they have no
    valid next-token label, and :func:`_allocate_batches_world` drops them from
    the forward pass. ``length - 1`` is clamped at 0 accordingly.

    Returns
    -------
    np.ndarray of shape ``(len(data),)`` with dtype int64.
    """
    if "length" in data.column_names:
        lengths = np.asarray(data["length"], dtype=np.int64)
    elif "input_ids" in data.column_names:
        lengths = np.asarray([len(x) for x in data["input_ids"]], dtype=np.int64)
    else:
        lengths = np.asarray([len(x) for x in data["labels"]], dtype=np.int64)
    return np.maximum(lengths - 1, 0)


@dataclass(frozen=True)
class AttributionUnit:
    """The unit one gradient or score row attributes to: a whole document or
    one of its token positions."""

    kind: Literal["document", "token"]

    @classmethod
    def from_cfg(cls, cfg: IndexConfig) -> "AttributionUnit":
        """The unit ``cfg`` attributes to."""
        return cls("token" if cfg.attribute_tokens else "document")

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> "AttributionUnit":
        """The unit of a gradient or score store, read from its ``info.json``."""
        return cls("token" if info.get("attribute_tokens") else "document")

    def to_info(self) -> dict[str, Any]:
        """The ``info.json`` entries :meth:`from_info` reads back."""
        return {"attribute_tokens": self.kind == "token"}

    @property
    def ragged(self) -> bool:
        """Whether an example owns a variable number of rows, so its store
        records each example's rows in ``offsets.npy``."""
        return self.kind != "document"

    def row_counts(self, data: Dataset) -> np.ndarray:
        """Number of rows each example of ``data`` owns."""
        if self.kind == "token":
            return compute_num_token_grads(data)
        return np.ones(len(data), dtype=np.int64)
