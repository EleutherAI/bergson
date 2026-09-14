"""Average an ``ekfac`` run's preconditioned query rows per memory opener.

The run scored one row per sampled continuation; this writes a query index
with one row per opener, the mean of its on-topic continuations, so the
scoring pass costs one column per opener instead of one per continuation.
``openers.json`` beside it records which continuation rows each column
averages.

    python -m examples.contrastive_queries.aggregate_query \\
        --run <ekfac run> --queries queries/<set>.jsonl --set <set>
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bergson.data import create_index, load_gradients

from .memory_prompts import SETS, on_topic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="ekfac run directory")
    ap.add_argument(
        "--queries", required=True, help="the jsonl the run scored, in order"
    )
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--out", default="", help="default <run>/kfac_query_openers")
    args = ap.parse_args()

    run = Path(args.run)
    src_dir = run / "kfac_query"
    out_dir = Path(args.out) if args.out else run / "kfac_query_openers"
    rows = [json.loads(line) for line in open(args.queries)]
    src = load_gradients(src_dir)
    info = json.load(open(src_dir / "info.json"))
    assert src.shape[0] == len(rows), (src.shape, len(rows))

    columns: dict[str, list[int]] = {}
    for j, r in enumerate(rows):
        if on_topic(args.set, r["continuation"]):
            columns.setdefault(r["prompt"], []).append(j)
    openers = [p for p in SETS[args.set] if p in columns]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out = create_index(
        root=out_dir,
        num_grads=len(openers),
        grad_sizes=info["grad_sizes"],
        dtype=np.float32,
    )
    for i, prompt in enumerate(openers):
        acc = np.zeros(src.shape[1], dtype=np.float64)
        for j in columns[prompt]:
            acc += np.asarray(src[j], dtype=np.float64)
        out[i] = (acc / len(columns[prompt])).astype(np.float32)
        print(f"{i}: {len(columns[prompt])} continuations <- {prompt}")
    out.flush()
    for name in ("config.yaml", "processor_config.yaml"):
        if (src_dir / name).exists():
            shutil.copy(src_dir / name, out_dir / name)
    json.dump(
        {"openers": openers, "columns": {p: columns[p] for p in openers}},
        open(out_dir / "openers.json", "w"),
        indent=1,
    )
    print(f"wrote {out_dir} with {len(openers)} rows")


if __name__ == "__main__":
    main()
