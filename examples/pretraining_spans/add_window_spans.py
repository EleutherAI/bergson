"""Copy a pre-tokenized dataset with a ``span_starts`` column of fixed-width windows.

python examples/pretraining_spans/add_window_spans.py <train.hf> <width> <out.hf>
"""

import sys

import numpy as np
import pyarrow as pa
from datasets import Dataset, List, Value

src, width, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ds = Dataset.load_from_disk(src)
lengths = np.asarray(ds["length"], dtype=np.int64)
counts = -(-lengths // width)
offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
np.cumsum(counts, out=offsets[1:])
# Each row's starts are 0, width, 2 * width, ... below its length.
values = (np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)) * width
starts = pa.ListArray.from_arrays(offsets.astype(np.int32), values.astype(np.int32))
ds = ds.add_column(
    "span_starts",
    starts,
    new_fingerprint=f"{ds._fingerprint}_spans{width}",
    feature=List(Value("int32")),
)
ds.save_to_disk(out)
print(f"{len(ds)} rows, {offsets[-1]} spans of {width} tokens -> {out}")
