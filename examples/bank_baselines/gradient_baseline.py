"""Baseline: gradient cosine similarity (TracIn-style, full parameters).

Scores a train doc against a query by the cosine similarity of their full
per-example loss gradients on the bank's model -- the raw, unpreconditioned
influence signal that TrackStar/EK-FAC refine. A train doc whose gradient
aligns with the query's is predicted influential, so the loss-diff-convention
score is ``-cosine``.

Full gradients are never stored: the 50 query gradients stay resident on the
GPU (GPT-2 is 124M params, ~25 GB in fp32) and each train-doc gradient is
computed, dotted against all queries, and discarded.

Run with (builds the default bank if --bank is omitted):
    python -m examples.bank_baselines.gradient_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import common


def per_example_grad(
    text: str, model, tokenizer, device: str, max_len: int
) -> torch.Tensor:
    """Flattened full-parameter gradient of the mean-CE loss on one document."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(
        device
    )
    model.zero_grad(set_to_none=True)
    model(input_ids=enc["input_ids"], labels=enc["input_ids"]).loss.backward()
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters()
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument("--out", default=str(common.REPO / "runs" / "bank_baselines"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    out_dir = Path(args.out)

    tokenizer = AutoTokenizer.from_pretrained(spec.model)
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model, dtype=torch.float32, attn_implementation="eager"
    ).to(args.device)
    model.eval()

    train_texts, query_texts = common.load_texts(spec, args.query_split)

    # Keep the query gradients resident; stream train gradients against them.
    # Pre-allocate and fill row by row so we never hold a second copy (each full
    # gradient is ~0.5 GB, the [Q, P] block ~25 GB in fp32).
    print(f"Computing {len(query_texts)} query gradients ...")
    n_params = sum(p.numel() for p in model.parameters())
    query_grads = torch.empty(len(query_texts), n_params, device=args.device)
    for i, t in enumerate(query_texts):
        query_grads[i] = per_example_grad(
            t, model, tokenizer, args.device, args.max_len
        )
    query_norms = query_grads.norm(dim=1).clamp_min(1e-12)

    print(f"Scoring {len(train_texts)} train gradients against queries ...")
    scores = np.empty((len(train_texts), len(query_texts)), dtype=np.float32)
    for i, text in enumerate(train_texts):
        g = per_example_grad(text, model, tokenizer, args.device, args.max_len)
        cos = (query_grads @ g) / (query_norms * g.norm().clamp_min(1e-12))
        scores[i] = cos.detach().cpu().numpy()

    scores = -scores  # loss-diff convention (aligned gradient => influential)
    score_path = common.save_scores(scores, out_dir, "gradient_scores")

    rhos = common.evaluate_lds(
        bank, score_path, out_dir / "gradient_validate", spec, args.query_split
    )
    common.report("gradient cosine similarity", rhos)


if __name__ == "__main__":
    main()
