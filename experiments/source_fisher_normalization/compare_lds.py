"""Compare LDS across SOURCE Fisher-normalization settings.

Reads each validate run's summary.csv (one row per query) and reports the mean
correlation with a 95% CI over queries, which is the convention behind the
published `0.387 +/- 0.039`.

    python experiments/source_fisher_normalization/compare_lds.py \
        none=runs/lotus_source_q50_damp0_validate \
        document=runs/lotus_source_q50_docnorm_validate \
        token=runs/lotus_source_q50_toknorm_validate
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CI95 = 1.959963985


def summarize(path: Path) -> dict:
    df = pd.read_csv(path / "summary.csv")
    out: dict = {"n_queries": len(df)}
    for col, key in (("spearman_corr", "rho"), ("pearson_corr", "r")):
        v = df[col].to_numpy()
        out[key] = v.mean()
        out[f"{key}_ci"] = CI95 * v.std(ddof=1) / np.sqrt(len(v))
        out[f"{key}_raw"] = v
    return out


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    runs = {}
    for arg in args:
        label, _, path = arg.partition("=")
        p = Path(path) / "summary.csv"
        if not p.exists():
            print(f"skip {label}: no {p}")
            continue
        runs[label] = summarize(Path(path))

    print(f"\n{'normalization':<14}{'rho':>18}{'r':>18}{'queries':>9}")
    print("-" * 59)
    for label, s in runs.items():
        print(
            f"{label:<14}"
            f"{s['rho']:>10.3f} +/- {s['rho_ci']:.3f}"
            f"{s['r']:>10.3f} +/- {s['r_ci']:.3f}"
            f"{s['n_queries']:>9}"
        )

    # Paired over the same 50 queries, so a paired test is the informative one.
    if "none" in runs:
        base = runs["none"]
        for label, s in runs.items():
            if label == "none" or len(s["rho_raw"]) != len(base["rho_raw"]):
                continue
            delta = s["rho_raw"] - base["rho_raw"]
            stat = wilcoxon(delta)
            print(
                f"\n{label} vs none: mean drho={delta.mean():+.3f} "
                f"(won {int((delta > 0).sum())}/{len(delta)} queries, "
                f"wilcoxon p={stat.pvalue:.2g})"
            )

    print("\nLaTeX rows:")
    for label, s in runs.items():
        print(
            f"SOURCE ({label}) & ${s['rho']:.3f} \\pm {s['rho_ci']:.3f}$ "
            f"& ${s['r']:.3f} \\pm {s['r_ci']:.3f}$ \\\\"
        )


if __name__ == "__main__":
    main()
