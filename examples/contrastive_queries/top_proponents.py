"""Top proponent training documents per memory opener.

Reads the per-continuation score columns of an ``ekfac`` run, averages the
columns of each opener's on-topic continuations (``memory_prompts.on_topic``),
and lists the training documents with the most negative loss-signed score,
i.e. the ones whose removal would most raise the loss of the model's own
continuations.

    python -m examples.contrastive_queries.top_proponents \\
        --scores <run>/scores --queries queries/<set>.jsonl --set <set> \\
        --data <train dataset> --tokenizer <path> --out <set>_proponents.md
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from bergson.data import load_scores_loss_signed

from .memory_prompts import SETS, on_topic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument(
        "--queries", required=True, help="the jsonl the run scored, in order"
    )
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--data", required=True, help="training dataset (load_from_disk)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--snippet_tokens", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.queries)]
    scores, multi = load_scores_loss_signed(args.scores)
    assert multi and scores.shape[1] == len(rows), (scores.shape, len(rows))
    scores = np.asarray(scores, dtype=np.float64)
    data = load_from_disk(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    columns = defaultdict(list)
    for j, r in enumerate(rows):
        if on_topic(args.set, r["continuation"]):
            columns[r["prompt"]].append(j)

    out = [f"# Top proponents: {args.set}\n"]
    summary = {}
    for prompt in SETS[args.set]:
        cols = columns.get(prompt, [])
        out.append(f"\n## {prompt}\n")
        if not cols:
            out.append("_no on-topic continuations_\n")
            continue
        mean = scores[:, cols].mean(1)
        top = np.argsort(mean)[: args.k]
        out.append(
            f"{len(cols)} continuations averaged; lower score = stronger proponent\n"
        )
        out.append("| rank | doc | score | snippet |\n|---|---|---|---|")
        for rank, d in enumerate(top, 1):
            ids = data[int(d)]["input_ids"][: args.snippet_tokens]
            snippet = tok.decode(ids).replace("\n", " ").replace("|", "\\|")
            out.append(f"| {rank} | {int(d)} | {mean[d]:.3f} | {snippet} |")
        summary[prompt] = [int(d) for d in top]
    open(args.out, "w").write("\n".join(out) + "\n")
    json.dump(summary, open(args.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
