"""Per-query LDS of a bergson score store against a leave-k-out bank.

Usage:
    python examples/compare_wikitext/lds_from_bank.py --scores <run>/scores --bank <bank> \
        [--sign grad|loss]
    python examples/compare_wikitext/lds_from_bank.py --validation <bank>/validation.csv
    python examples/compare_wikitext/lds_from_bank.py --npy <scores.npy> --bank <bank> --sign loss

The bank is a magic/validate run dir holding ``subsets.json`` (one doc-id
list per subset) and ``validation.csv`` (columns subset, query, diff,
score_sum), where ``diff`` is the measured query-loss change for that subset.
For each query, Spearman between the summed scores of each subset's removed
docs and ``diff``; the LDS is the mean over queries, with a 95% CI from a 10k
bootstrap over subsets. ``--sign grad`` (default) negates influence-style
scores, where higher means more helpful (EK-FAC, TrackStar), so that the sum is
loss-signed like MAGIC's ``score_sum``; SOURCE (approxunrolling) scores are
already loss-signed, use ``--sign loss``. The ``--validation`` form scores the
bank's own MAGIC ``score_sum`` column.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def load_scores(scores_dir: Path) -> np.ndarray:
    info = json.loads((scores_dir / "info.json").read_text())
    dt = np.dtype(
        {k: info["dtype"][k] for k in ("names", "formats", "offsets", "itemsize")}
    )
    raw = np.memmap(scores_dir / "scores.bin", dtype=dt, mode="r")
    n_q = info["num_scores"]
    for q in range(n_q):
        assert raw[f"written_{q}"].all(), f"query {q} has unwritten scores"
    return np.stack([raw[f"score_{q}"] for q in range(n_q)], axis=1)


def lds(sums: np.ndarray, diffs: np.ndarray, n_boot: int, seed: int) -> dict:
    n_sub, n_q = diffs.shape
    res = [spearmanr(sums[:, q], diffs[:, q]) for q in range(n_q)]
    per_q = np.array([r.statistic for r in res])
    pvals = np.array([r.pvalue for r in res])

    r_sums = np.stack([rankdata(sums[:, q]) for q in range(n_q)], axis=1)
    r_diffs = np.stack([rankdata(diffs[:, q]) for q in range(n_q)], axis=1)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_sub, n_sub)
        a = r_sums[idx] - r_sums[idx].mean(0)
        d = r_diffs[idx] - r_diffs[idx].mean(0)
        boots[b] = ((a * d).sum(0) / np.sqrt((a**2).sum(0) * (d**2).sum(0))).mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "lds": float(per_q.mean()),
        "ci95": [float(lo), float(hi)],
        "median": float(np.median(per_q)),
        "min": float(per_q.min()),
        "max": float(per_q.max()),
        "n_sig": int((pvals < 0.05).sum()),
        "n_queries": int(n_q),
        "n_subsets": int(n_sub),
        "per_query": per_q.round(4).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path)
    ap.add_argument(
        "--npy", type=Path, help="a [docs, queries] score matrix instead of --scores"
    )
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--validation", type=Path)
    ap.add_argument("--sign", choices=["grad", "loss"], default="grad")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.validation is not None:
        val = pd.read_csv(args.validation)
        sums = val.pivot(index="subset", columns="query", values="score_sum").to_numpy()
        source = str(args.validation)
    else:
        assert (
            args.scores is not None or args.npy is not None
        ) and args.bank is not None
        val = pd.read_csv(args.bank / "validation.csv")
        subsets = json.loads((args.bank / "subsets.json").read_text())
        if args.npy is not None:
            scores = np.load(args.npy)
        else:
            scores = load_scores(args.scores)
        n_sub = val["subset"].nunique()
        sums = np.stack([scores[np.asarray(subsets[s])].sum(0) for s in range(n_sub)])
        if args.sign == "grad":
            sums = -sums
        source = str(args.npy if args.npy is not None else args.scores)
    diffs = val.pivot(index="subset", columns="query", values="diff").to_numpy()
    assert diffs.shape == sums.shape, (diffs.shape, sums.shape)

    result = {
        "scores": source,
        "sign": args.sign,
        **lds(sums, diffs, args.n_boot, args.seed),
    }
    print(
        f"LDS {result['lds']:.4f} [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}]  "
        f"median {result['median']:.4f} min {result['min']:.4f} "
        f"max {result['max']:.4f}  "
        f"p<.05 {result['n_sig']}/{result['n_queries']}  "
        f"n_subsets={result['n_subsets']}"
    )
    print("per-query:", result["per_query"])
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
