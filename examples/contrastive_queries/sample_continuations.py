"""Sample a model's continuations of the memory prompts and write them as
query sets (``prompt``, ``continuation`` rows; format ``formats/continuation.yaml``).

Continuations are sampled at temperature 1 so they are the model's own
recollection. Degenerate ones (looping n-grams, mostly non-ASCII, too short)
are dropped and counted.

    python -m examples.contrastive_queries.sample_continuations \
        --model <path> --out queries/
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory_prompts import SETS


def degenerate(text: str) -> str | None:
    words = text.split()
    if len(words) < 8:
        return "short"
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if len(set(grams)) < 0.6 * len(grams):
        return "loop"
    if sum(ord(c) > 127 for c in text) > 0.1 * len(text):
        return "non-ascii"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--out", default="queries")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, prompts in SETS.items():
        kept, dropped = [], {}
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                gen = model.generate(
                    ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=1.0,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.k,
                    pad_token_id=tok.eos_token_id,
                )
            for g in gen:
                text = tok.decode(g[ids.shape[1] :], skip_special_tokens=True)
                why = degenerate(text)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                    continue
                kept.append({"prompt": prompt, "continuation": text})
        with open(out / f"{name}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: kept {len(kept)} of {len(prompts) * args.k}, dropped {dropped}")


if __name__ == "__main__":
    main()
