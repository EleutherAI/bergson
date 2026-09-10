"""Proponent-filter QLD per method from bergson filter-proponents runs.

    python examples/compare_wikitext/qld_from_filters.py runs/<set> [<bank dir>]

Reads ``<set>/filter_<method>/`` for each method.

For each method: the mean over queries of ``filter_change`` (the query's loss
change after retraining without its top proponents), a 95% CI from a 10k
bootstrap over queries (seed 0), and the matched random control (mean of
``random_mean``, the bank's random same-size removals). Writes
``<set>/qld_<method>.json``.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def summarise(
    filter_dir: Path, bank: Path | None, n_boot: int = 10000, seed: int = 0
) -> dict:
    summary = filter_dir / "filter_summary.csv"
    if summary.exists():
        rows = list(csv.DictReader(open(summary)))
        change = np.array([float(r["filter_change"]) for r in rows])
        random_mean = np.array([float(r["random_mean"]) for r in rows])
    else:
        # The per-query results are written before the run's own evaluation of
        # the bank; take the random control from the bank's validation.csv
        # instead (its ``diff`` is baseline minus retrained, so negate).
        assert bank is not None, f"{summary} missing and no --bank given"
        rows = list(csv.DictReader(open(filter_dir / "filter_proponents.csv")))
        change = np.array([float(r["loss_change"]) for r in rows])
        val = list(csv.DictReader(open(bank / "validation.csv")))
        by_q: dict[int, list[float]] = {}
        for r in val:
            by_q.setdefault(int(r["query"]), []).append(-float(r["diff"]))
        random_mean = np.array([np.mean(by_q[int(r["query"])]) for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(change), size=(n_boot, len(change)))
    boots = change[idx].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "qld": float(change.mean()),
        "ci95": [float(lo), float(hi)],
        "random": float(random_mean.mean()),
        "n_queries": int(len(change)),
        "n_removed": int(rows[0]["n_removed"]),
        "per_query": change.round(5).tolist(),
    }


run_dir = Path(sys.argv[1])
bank = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "bank"
print("| method | proponent QLD | 95% CI | random 1% | queries |")
print("|---|---|---|---|---|")
for d in sorted(run_dir.glob("filter_*")):
    if (
        not (d / "filter_summary.csv").exists()
        and not (d / "filter_proponents.csv").exists()
    ):
        continue
    method = d.name[len("filter_") :]
    res = summarise(d, bank if bank.exists() else None)
    (run_dir / f"qld_{method}.json").write_text(json.dumps(res, indent=2))
    lo, hi = res["ci95"]
    print(
        f"| {method} | {res['qld']:.4f} | [{lo:.4f}, {hi:.4f}] | "
        f"{res['random']:.4f} | {res['n_queries']} |"
    )
