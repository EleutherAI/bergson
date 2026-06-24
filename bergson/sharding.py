"""Embarrassingly-parallel sharded runs that share one run_path.

Multiple independent ``bergson build``/``bergson score`` invocations
(typically one per SLURM array task) each process one contiguous slice of
the dataset and publish into the same ``run_path``:

    run_path/
    ├── config.yaml              # canonical config, identical across shards
    └── shards/
        ├── 00000-of-00064/      # published shard
        │   ├── gradients.bin / scores.bin
        │   ├── info.json
        │   ├── shard.json       # provenance: dataset slice, host, timestamp
        │   └── ...
        └── 00017-of-00064.part/ # in progress, or left over from a crash

A shard is written under a ``.part`` suffix and atomically renamed into
place on success, so a published shard is always complete and a crashed
shard is rebuilt by simply re-running the same command (a published shard
is skipped, making restarts idempotent). Readers present the published
shards as one logically concatenated index, so no manual stitching is
needed at any point.
"""

import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SHARDS_DIRNAME = "shards"
PART_SUFFIX = ".part"
SHARD_RECORD_FILENAME = "shard.json"

_SHARD_DIR_RE = re.compile(r"^(\d{5})-of-(\d{5})$")


def shard_dir_name(shard_id: int, num_shards: int) -> str:
    """Directory name for one shard, e.g. ``00017-of-00064``."""
    return f"{shard_id:05d}-of-{num_shards:05d}"


def shard_row_range(total_rows: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """[start, end) row range of a contiguous shard.

    Matches ``datasets.Dataset.shard(num_shards, shard_id, contiguous=True)``.
    """
    div, mod = divmod(total_rows, num_shards)
    start = shard_id * div + min(shard_id, mod)
    end = start + div + (1 if shard_id < mod else 0)
    return start, end


def is_sharded_run(run_path: Path | str) -> bool:
    """Whether ``run_path`` holds per-shard subdirectories."""
    return (Path(run_path) / SHARDS_DIRNAME).is_dir()


def make_shard_record(
    shard_id: int,
    num_shards: int,
    split: str,
    row_range: tuple[int, int] | None,
    num_items: int,
) -> dict[str, Any]:
    """Provenance record written to ``shard.json`` inside a shard directory.

    ``row_range`` is the [start, end) slice of the resolved parent split
    *before* tokenization; ``num_items`` is the number of index items the
    shard produced, which can differ when chunking is enabled. The shard's
    own ``info.json`` holds the authoritative gradient counts.
    """
    record: dict[str, Any] = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "split": split,
        "num_items": num_items,
        "hostname": socket.gethostname(),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if row_range is not None:
        record["row_start"], record["row_end"] = row_range
        record["num_rows"] = row_range[1] - row_range[0]
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        if var in os.environ:
            record[var.lower()] = os.environ[var]
    return record


def write_shard_record(shard_dir: Path, record: dict[str, Any]) -> None:
    with (Path(shard_dir) / SHARD_RECORD_FILENAME).open("w") as f:
        json.dump(record, f, indent=2)


def shard_status(
    run_path: Path | str,
) -> tuple[dict[int, Path], dict[int, Path], int | None]:
    """Inventory of a sharded run.

    Returns ``(published, partial, num_shards)`` where the dicts map
    shard_id to its directory. ``num_shards`` is ``None`` if no shard
    directories exist yet.
    """
    shards_dir = Path(run_path) / SHARDS_DIRNAME
    published: dict[int, Path] = {}
    partial: dict[int, Path] = {}
    num_shards: int | None = None

    if not shards_dir.is_dir():
        return published, partial, num_shards

    for child in sorted(shards_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.removesuffix(PART_SUFFIX)
        match = _SHARD_DIR_RE.match(name)
        if match is None:
            continue
        shard_id, total = int(match.group(1)), int(match.group(2))
        if num_shards is None:
            num_shards = total
        elif total != num_shards:
            raise ValueError(
                f"Inconsistent shard counts in {shards_dir}: found shards "
                f"of {num_shards} and of {total}. The run_path mixes "
                f"runs with different --num_shards."
            )
        if child.name.endswith(PART_SUFFIX):
            partial[shard_id] = child
        else:
            published[shard_id] = child

    return published, partial, num_shards


def published_shard_dirs(
    run_path: Path | str, allow_partial: bool = False
) -> list[Path]:
    """Published shard directories in shard order.

    Raises if any shard is missing or unpublished, unless ``allow_partial``.
    """
    published, partial, num_shards = shard_status(run_path)
    if num_shards is None:
        raise FileNotFoundError(f"No shard directories found in {run_path}")

    missing = sorted(set(range(num_shards)) - published.keys())
    if missing and not allow_partial:
        in_progress = sorted(partial.keys())
        raise RuntimeError(
            f"Sharded run {run_path} is incomplete: missing shards {missing}"
            + (f" (in progress or crashed: {in_progress})" if in_progress else "")
            + ". Re-run the missing shards, or pass allow_partial=True to "
            "read the published subset."
        )

    return [published[i] for i in sorted(published)]


class ShardedMemmap:
    """Read-only view over per-shard arrays, concatenated along axis 0.

    Quacks enough like an ``np.memmap`` for the index readers: ``len``,
    ``shape``, ``dtype``, field access on structured arrays, and
    int/slice/fancy indexing. Indexing materializes the requested rows as
    a regular ndarray while the underlying per-shard memmaps stay lazy, so
    avoid full-index reads (``mmap[:]``) on very large runs — iterate
    ``shards`` instead.
    """

    def __init__(self, arrays: "Sequence[np.ndarray | ShardedMemmap]"):
        if not arrays:
            raise ValueError("ShardedMemmap needs at least one array")
        head = arrays[0]
        for arr in arrays[1:]:
            if arr.dtype != head.dtype or arr.shape[1:] != head.shape[1:]:
                raise ValueError(
                    f"Shards disagree on dtype/shape: {head.dtype}{head.shape[1:]} "
                    f"vs {arr.dtype}{arr.shape[1:]}"
                )
        self.shards = list(arrays)
        # offsets[i] is the global index of shard i's first row
        self._offsets = np.cumsum([0] + [len(a) for a in self.shards])

    @property
    def dtype(self) -> np.dtype:
        return self.shards[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self._offsets[-1]), *self.shards[0].shape[1:])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def copy(self) -> np.ndarray:
        """Materialize the full concatenated array."""
        return self[:]

    def reshape(self, *shape: Any) -> np.ndarray:
        """Materialize the full concatenated array and reshape it."""
        return self[:].reshape(*shape)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, str):
            return np.concatenate([np.asarray(a[key]) for a in self.shards], axis=0)

        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"index {key} out of range for length {len(self)}")
            shard = int(np.searchsorted(self._offsets, idx, side="right")) - 1
            return self.shards[shard][idx - self._offsets[shard]]

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return self[np.arange(start, stop, step)]
            pieces = [
                np.asarray(arr[max(start - off, 0) : max(stop - off, 0)])
                for arr, off in zip(self.shards, self._offsets)
            ]
            return np.concatenate(pieces, axis=0)

        indices = np.asarray(key)
        if indices.dtype == bool:
            indices = np.nonzero(indices)[0]
        indices = np.where(indices < 0, indices + len(self), indices)
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError(f"index out of range for length {len(self)}")

        out = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        shard_ids = np.searchsorted(self._offsets, indices, side="right") - 1
        for shard in np.unique(shard_ids):
            mask = shard_ids == shard
            out[mask] = self.shards[shard][indices[mask] - self._offsets[shard]]
        return out
