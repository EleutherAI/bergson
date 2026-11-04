"""Save 100 random validation chunks, the query set size of Wang et al. 2025.

python examples/replications/bae_approx_unrolling_source_replication/prep_queries.py
"""

import argparse

import numpy as np
from datasets import DatasetDict, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/wikitext_repro/queries")
parser.add_argument("--num_queries", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

validation = load_dataset(
    "EleutherAI/bergson-wikitext-2-4656-chunks", split="validation"
)
rng = np.random.default_rng(args.seed)
indices = np.sort(rng.choice(len(validation), args.num_queries, replace=False))
print(f"Validation rows: {indices.tolist()}")
DatasetDict({"validation": validation.select(indices)}).save_to_disk(args.out)
