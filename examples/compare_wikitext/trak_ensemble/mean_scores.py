"""Average the TRAK score stores of the ensemble members into one score dir.

    python examples/compare_wikitext/trak_ensemble/mean_scores.py \\
        runs/compare_wikitext/trak_ens/score_s*_*/scores \\
        --out runs/compare_wikitext/trak_ens/scores

Copies the first member's score directory (its config and metadata) and
writes ``scores.bin`` as the mean of every member's score columns; a row's
``written`` flag is set where every member wrote it. With ``--separate_q``
the whitened inner products and the ``1 - p`` weights (``trak_weights.npy``)
are averaged separately and then multiplied, the form of Engstrom et al.
(2024, DsDm); otherwise the members' weighted scores are averaged.
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
    ap.add_argument("members", nargs="+", help="score directories to average")
    ap.add_argument("--out", required=True)
    ap.add_argument("--separate_q", action="store_true")
    args = ap.parse_args()

    members = [load(Path(m)) for m in args.members]
    first = members[0]
    assert all(m.dtype == first.dtype and m.shape == first.shape for m in members)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        Path(args.members[0]), out, ignore=shutil.ignore_patterns("scores.bin")
    )
    mean = np.memmap(
        out / "scores.bin", dtype=first.dtype, mode="w+", shape=first.shape
    )
    weights = [np.load(Path(d) / "trak_weights.npy") for d in args.members]
    q_mean = np.mean(weights, axis=0)
    for name in first.dtype.names:
        if name.startswith("score_"):
            if args.separate_q:
                unweighted = [
                    m[name].astype(np.float64) / w for m, w in zip(members, weights)
                ]
                mean[name] = np.mean(unweighted, axis=0) * q_mean
            else:
                mean[name] = np.mean(
                    [m[name].astype(np.float64) for m in members], axis=0
                )
        else:
            mean[name] = np.all([m[name] for m in members], axis=0)
    mean.flush()
    print(f"averaged {len(members)} members into {out} ({first.shape[0]} rows)")


if __name__ == "__main__":
    main()
