"""LDS of EKFAC-SOURCE and ASTRA-SOURCE, and their difference, with 95%
bootstrap CIs over queries.

python examples/replications/bae_approx_unrolling_source_replication/astra_source_lds.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default="runs/wikitext_repro")
args = parser.parse_args()
runs = Path(args.runs)


def spearman_per_query(validate_dir):
    df = pd.read_csv(runs / validate_dir / "summary.csv")
    return df.set_index("query")["spearman_corr"].sort_index().to_numpy()


def report(name, values):
    samples = np.random.default_rng(0).integers(0, len(values), (10_000, len(values)))
    low, high = np.quantile(values[samples].mean(1), [0.025, 0.975])
    print(
        f"{name}: {values.mean():.3f} [{low:.3f}, {high:.3f}] on {len(values)} queries"
    )


ekfac = spearman_per_query("validate_ekfac_source_mlp")
astra = spearman_per_query("validate_astra_source_mlp")
report("EKFAC-SOURCE", ekfac)
report("ASTRA-SOURCE", astra)
report("ASTRA-SOURCE - EKFAC-SOURCE", astra - ekfac)
print(f"ASTRA-SOURCE scores higher on {(astra > ekfac).sum()} of {len(astra)} queries")
