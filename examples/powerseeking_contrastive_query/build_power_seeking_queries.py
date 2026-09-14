"""Write the Anthropic power-seeking evaluation as a query and its control.

``power_seeking.jsonl`` pairs each question with the power-seeking answer and
``power_seeking_control.jsonl`` with the other one; both render through
``formats/question_answer.yaml``. The LM-written file is used: in the
human-written one ``answer_matching_behavior`` is the answer declining power.

    python -m examples.powerseeking_contrastive_query.build_power_seeking_queries \\
        --out queries/
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

    rows = list(load_dataset("json", data_files=DATA, split="train"))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, key in [
        ("power_seeking", "answer_matching_behavior"),
        ("power_seeking_control", "answer_not_matching_behavior"),
    ]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(
                    json.dumps({"question": r["question"], "answer": r[key]}) + "\n"
                )
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
