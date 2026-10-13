"""Spread a candidate span store over every training document for the span filter.

    python examples/pretraining_spans/expand_candidates.py \
        <candidate store> <train.hf with the span column> <out>

Spans of non-candidate documents get a score after every proponent in the loss-signed
order, so the proponent filter takes its fraction of all training tokens from the
candidates.
"""

import shutil
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
from datasets import Dataset

from bergson.config import IndexConfig, ScoreConfig
from bergson.config.config_io import load_subconfig
from bergson.data import load_scores, span_rows
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.spans import TOKEN_INFLUENCE_SHIFT
from bergson.utils.utils import convert_dtype_to_torch

src, train, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
store = load_scores(src)
assert store.spans is not None and store.offsets is not None, f"{src} is not spans"
assert store.candidates is not None, f"{src} has no candidates"
assert store.is_written(), f"{src} is incomplete"

score_cfg = load_subconfig(src, "score_cfg", ScoreConfig)
index_cfg = load_subconfig(src, "index_cfg", IndexConfig)
assert score_cfg is not None and index_cfg is not None, f"{src} has no saved config"
column = index_cfg.data.span_column
assert column is not None, f"{src} was not scored with a span column"

counts, spans = span_rows(
    Dataset.load_from_disk(train).select_columns(["length", column]),
    column,
    TOKEN_INFLUENCE_SHIFT[score_cfg.token_influence],
)
dtype = store.mmap.dtype[0]
sentinel = ml_dtypes.finfo(dtype).max * (-1 if score_cfg.higher_is_better else 1)

writer = MemmapTokenScoreWriter(
    out,
    counts,
    store.num_scores,
    dtype=convert_dtype_to_torch(dtype),
    overwrite=True,
    distributed=False,
    spans=spans,
)
for i in range(store.num_scores):
    writer.scores[f"score_{i}"] = sentinel
    writer.scores[f"written_{i}"] = True

cand = store.candidates
rows = np.concatenate(
    [np.arange(int(writer.offsets[c]), int(writer.offsets[c + 1])) for c in cand]
)
assert len(rows) == len(store.mmap), "candidate span counts differ"
assert (spans[rows] == store.spans).all(), "candidate span bounds differ"
writer.scores[rows] = store.mmap[:]
writer.flush()
for f in ("config.yaml", "processor_config.yaml"):
    if (src / f).exists():
        shutil.copy(src / f, out / f)
print(f"{len(cand)} of {len(counts)} documents scored; the rest at {sentinel:g}")
