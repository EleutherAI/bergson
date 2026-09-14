"""Write the Anthropic power-seeking evaluation as a contrastive query set.

Each row of ``power_seeking.jsonl`` holds the question, the power-seeking
answer and the other answer. The LM-written file is used; in the
human-written one ``answer_matching_behavior`` is the answer declining
power. ``formats/power_seeking.yaml`` targets the
first and ``formats/power_seeking_control.yaml`` the second, so the query
loss minus the control loss measures the preference for power-seeking on the
same questions.

    python -m examples.contrastive_queries.build_power_seeking_queries --out queries/
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA = (
    "hf://datasets/Anthropic/model-written-evals/advanced-ai-risk/"
    "lm_generated_evals/power-seeking-inclination.jsonl"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="queries")
    ap.add_argument("--n", type=int, default=100, help="rows to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [
        {
            "question": r["question"],
            "answer_matching_behavior": r["answer_matching_behavior"],
            "answer_not_matching_behavior": r["answer_not_matching_behavior"],
        }
        for r in load_dataset("json", data_files=DATA, split="train")
    ]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "power_seeking.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"power_seeking: {len(rows)} rows")


if __name__ == "__main__":
    main()
