"""Draw a control bank's removal sets as random spans, and an all-zero span store
for the bank's validate step to read the grid shape from.

    python examples/pretraining_spans/random_span_bank.py \
        <train.hf with span_starts> <token influence> <fraction> <count> <seed> \
        <bank run_path> <zero store out>

Each set holds random whole spans over the training set totalling ``fraction`` of
the ``to_grid`` token cells, the last one cut to the filter's exact count, and is
written to ``<bank run_path>/subsets.json``, which the LDS step reads in place of
its own random token cells.
"""

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset

from bergson.data import span_rows
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.spans import TOKEN_INFLUENCE_SHIFT

train, influence, fraction = sys.argv[1], sys.argv[2], float(sys.argv[3])
count, seed, bank, store = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]

counts, spans = span_rows(
    Dataset.load_from_disk(train).select_columns(["length", "span_starts"]),
    "span_starts",
    TOKEN_INFLUENCE_SHIFT[influence],
)
writer = MemmapTokenScoreWriter(
    Path(store), counts, 1, overwrite=True, distributed=False, spans=spans
)
writer.scores["written_0"] = True
writer.flush()

# The grid Scores.to_grid lays a span store out on, flattened row-major.
seq_len = int(spans[:, 1].max()) + 1
num_filtered = max(1, round(fraction * len(counts) * seq_len))
doc = np.repeat(np.arange(len(counts)), counts)
sizes = spans[:, 1] - spans[:, 0]
nonempty = np.flatnonzero(sizes)

rng = np.random.default_rng(seed)
subsets, take = [], nonempty[:0]
for _ in range(count):
    order = rng.permutation(nonempty)
    take = order[: np.searchsorted(np.cumsum(sizes[order]), num_filtered) + 1]
    cells = np.concatenate(
        [doc[s] * seq_len + np.arange(spans[s, 0], spans[s, 1]) for s in take]
    )[:num_filtered]
    assert len(cells) == num_filtered
    subsets.append(np.sort(cells).tolist())

Path(bank).mkdir(parents=True, exist_ok=True)
with open(Path(bank) / "subsets.json", "w") as f:
    json.dump(subsets, f)
print(
    f"{count} sets of {num_filtered} cells ({len(take)} spans in the last) of "
    f"{len(counts)} x {seq_len} -> {bank}/subsets.json; zero store {store}"
)
