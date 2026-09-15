"""Compare MAGIC scores from a run over the expert set alone with the full
run's scores on the same documents.

Usage:
    python examples/pac_labeling/compare_subset.py \
        --full <full run>/scores --subset <subset run>/magic/scores \
        --ids <subset run>/orig_ids.npy --query-ids <subset run>/query_ids.npy \
        [--cheap <run>/ekfac_scores/scores --bank <bank> --pac <pac out>]

Per query: Pearson and Spearman between the two score vectors over the
subset, the overlap of their strongest 1% (of the full corpus size, so the
same count in both), and when ``--cheap``, ``--bank`` and ``--pac`` are given,
the LDS of the hybrid vector that takes subset-run scores on the expert set
and calibrated cheap scores elsewhere, against the hybrid built from the full
run's scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bergson.data import load_scores_loss_signed
from examples.pac_labeling.pac_attribution import lds, load_bank, overlap, top_set

TOP_FRACTION = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--query-ids", type=Path, required=True)
    ap.add_argument("--cheap", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--pac", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    full, _ = load_scores_loss_signed(str(args.full))
    sub, _ = load_scores_loss_signed(str(args.subset))
    full, sub = full.numpy().astype(np.float64), sub.numpy().astype(np.float64)
    ids = np.load(args.ids)
    qids = np.load(args.query_ids)
    assert sub.shape == (len(ids), len(qids)), (sub.shape, len(ids), len(qids))
    n = full.shape[0]
    k = int(TOP_FRACTION * n)

    cheap = subsets = diffs = None
    if args.cheap is not None:
        cheap, _ = load_scores_loss_signed(str(args.cheap))
        cheap = cheap.numpy().astype(np.float64)
        subsets, diffs = load_bank(args.bank)

    rows = []
    for j, q in enumerate(qids):
        y_full, y_sub = full[ids, q], sub[:, j]
        row = {
            "query": int(q),
            "n_subset": len(ids),
            "pearson": pearsonr(y_full, y_sub)[0],
            "spearman": spearmanr(y_full, y_sub).correlation,
            "scale": np.polyfit(y_full, y_sub, 1)[0],
            "top1_overlap": overlap(top_set(y_sub, k), top_set(y_full, k)),
            "top1_full_in_subset": np.isin(top_set(full[:, q], k), ids).mean(),
        }
        if cheap is not None:
            meta = json.loads((args.pac / f"expert_set_q{q}.json").read_text())
            cal = meta["calibration"]
            yhat = cal["slope"] * cheap[:, q] + cal["intercept"]
            expert = np.zeros(n, dtype=bool)
            expert[np.load(args.pac / f"expert_set_q{q}.npy")] = True
            hybrid_full = np.where(expert, full[:, q], yhat)
            hybrid_sub = hybrid_full.copy()
            hybrid_sub[ids] = y_sub
            row.update(
                lds_full=lds(full[:, q], subsets, diffs[:, q]),
                lds_cheap=lds(yhat, subsets, diffs[:, q]),
                lds_hybrid_full=lds(hybrid_full, subsets, diffs[:, q]),
                lds_hybrid_subset=lds(hybrid_sub, subsets, diffs[:, q]),
                top1_hybrid_full=overlap(
                    top_set(hybrid_full, k), top_set(full[:, q], k)
                ),
                top1_hybrid_subset=overlap(
                    top_set(hybrid_sub, k), top_set(full[:, q], k)
                ),
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out is not None:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
