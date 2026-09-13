"""Linear combination of score stores, e.g. debiased scores.

    python examples/compare_wikitext/combine_scores.py \\
        --term 1 runs/compare_wikitext/ekfac/scores \\
        --term -1 runs/compare_wikitext/ekfac_base/scores \\
        --out runs/compare_wikitext/ekfac_debiased/scores

Copies the first term's score directory (its config and metadata) and writes
``scores.bin`` as the coefficient-weighted sum of the terms' score columns; a
row's ``written`` flags are set where every term wrote it. Subtracting the
scores taken at the model the fine-tune started from is the debias step of
Wu et al. (2024, DDA); a contrast query is the difference of two query sets.
Stores taken at different models or Hessians differ in scale, so pass
``--standardize`` to put every store on unit standard deviation first.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load(path: Path) -> np.memmap:
    info = json.load(open(path / "info.json"))
    # The stored dtype carries offsets and itemsize; use it verbatim.
    dtype = np.dtype(info["dtype"])
    return np.memmap(
        path / "scores.bin", dtype=dtype, mode="r", shape=(info["num_rows"],)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--term",
        nargs=2,
        action="append",
        metavar=("COEF", "SCORES"),
        required=True,
        help="coefficient and score directory; repeat for every term",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="divide each store's score columns by their overall standard "
        "deviation before combining, so stores from different models or "
        "Hessians enter on the same scale",
    )
    args = ap.parse_args()

    coefs = [float(c) for c, _ in args.term]
    stores = [load(Path(d)) for _, d in args.term]
    first = stores[0]
    assert all(s.dtype == first.dtype and s.shape == first.shape for s in stores)
    score_names = [n for n in first.dtype.names if n.startswith("score_")]
    scales = [1.0] * len(stores)
    if args.standardize:
        scales = [
            1.0 / np.std([s[n].astype(np.float64) for n in score_names]) for s in stores
        ]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        Path(args.term[0][1]), out, ignore=shutil.ignore_patterns("scores.bin")
    )
    combined = np.memmap(
        out / "scores.bin", dtype=first.dtype, mode="w+", shape=first.shape
    )
    for name in first.dtype.names:
        if name.startswith("score_"):
            combined[name] = sum(
                c * k * s[name].astype(np.float64)
                for c, k, s in zip(coefs, scales, stores)
            )
        elif name.startswith("written"):
            combined[name] = np.all([s[name] for s in stores], axis=0)
        else:
            combined[name] = first[name]
    combined.flush()
    terms = " + ".join(f"{c:g}*{d}" for c, (_, d) in zip(coefs, args.term))
    print(f"wrote {out} = {terms} ({first.shape[0]} rows)")


if __name__ == "__main__":
    main()
