"""Keep every training token in the per-token filter's pool.

    python examples/compare_wikitext/tokens/full_pool.py <per-token store> <out>

``exclude_zero_scores`` drops rows that are zero for every query and takes the 1%
from the rest, which is meant to skip padding. Lexical and embedding baselines give
many real tokens an exact zero (no overlap with any query), which would shrink their
removal count; this sets those rows to +1e-30, after every proponent in the
loss-signed order, so the filter removes 1% of all training tokens as it does for the
gradient methods.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from bergson.data import load_scores
from bergson.score.score_writer import save_token_scores

src, out = Path(sys.argv[1]), Path(sys.argv[2])
store = load_scores(src)
rows = np.asarray(store[:], dtype=np.float32).copy()
zero = ~rows.any(axis=1)
rows[zero] = 1e-30
print(f"{zero.sum()} of {len(rows)} rows were zero for every query")
save_token_scores(out, rows, store.offsets)
for f in ("config.yaml", "processor_config.yaml"):
    if (src / f).exists():
        shutil.copy(src / f, out / f)
