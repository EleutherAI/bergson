"""Copy a query gradient store, replacing some of its rows with another store's.

    python examples/gpt2_wikitext_leaderboard/replace_query_rows.py \
        <store> <other store> <out> <row> [<row> ...]
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from bergson.data import load_gradients

store, other_store, out = map(Path, sys.argv[1:4])
rows = [int(r) for r in sys.argv[4:]]
info = json.loads((store / "info.json").read_text())
assert info == json.loads((other_store / "info.json").read_text()), "layouts differ"

shutil.copytree(store, out)
other = load_gradients(other_store)
merged = np.memmap(
    out / "gradients.bin", dtype=other.dtype, mode="r+", shape=other.shape
)
merged[rows] = other[rows]
merged.flush()
print(f"{out}: rows {rows} from {other_store}")
