"""Top proponent training documents per memory opener.

Reads the score columns of a run scored against ``aggregate_query.py``'s
per-opener query index and lists, for each opener, the training documents
with the most negative loss-signed score: the ones whose removal would most
raise the loss of the model's own continuations.

    python -m examples.contrastive_queries.top_proponents \\
        --scores <run>/scores --openers <run>/kfac_query_openers/openers.json \\
        --data <train dataset> --tokenizer <path> --out <set>_proponents.md
"""

import argparse
import json

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from bergson.data import load_scores_loss_signed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--openers", required=True, help="openers.json of the query index")
    ap.add_argument("--data", required=True, help="training dataset (load_from_disk)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--snippet_tokens", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.openers))
    openers, columns = meta["openers"], meta["columns"]
    scores, multi = load_scores_loss_signed(args.scores)
    assert multi and scores.shape[1] == len(openers), (scores.shape, len(openers))
    scores = np.asarray(scores, dtype=np.float64)
    data = load_from_disk(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out, summary = ["# Top proponents\n"], {}
    for i, prompt in enumerate(openers):
        top = np.argsort(scores[:, i])[: args.k]
        out.append(f"\n## {prompt}\n")
        out.append(
            f"{len(columns[prompt])} continuations averaged; "
            "lower score = stronger proponent\n"
        )
        out.append("| rank | doc | score | snippet |\n|---|---|---|---|")
        for rank, d in enumerate(top, 1):
            ids = data[int(d)]["input_ids"][: args.snippet_tokens]
            snippet = tok.decode(ids).replace("\n", " ").replace("|", "\\|")
            out.append(f"| {rank} | {int(d)} | {scores[d, i]:.3f} | {snippet} |")
        summary[prompt] = [int(d) for d in top]
    open(args.out, "w").write("\n".join(out) + "\n")
    json.dump(summary, open(args.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
