"""Baseline: semantic search with Qwen3-Embedding-8B.

A SOTA decoder-based embedding model (causal-attention Qwen3 backbone,
last-token pooling), top of the open MTEB leaderboard -- a much stronger,
attention-based embedder than the jina-v3 encoder. Training docs are embedded
as passages and queries with the retrieval prompt; a train doc scores against a
query by cosine similarity, so the loss-diff-convention score is ``-cosine``.

(NVIDIA's NV-Embed-v2 is a comparable model but its custom modeling code is
incompatible with transformers 5.x; Qwen3-Embedding uses the native Qwen3
architecture and loads cleanly. Pass --model Qwen/Qwen3-Embedding-4B for the
smaller, faster variant.)

Run with (builds the default bank if --bank is omitted):
    python -m examples.bank_baselines.qwen3_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from . import common

MODEL = "Qwen/Qwen3-Embedding-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument("--out", default=str(common.REPO / "runs" / "bank_baselines"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    out_dir = Path(args.out)

    model = SentenceTransformer(
        args.model, model_kwargs={"torch_dtype": "bfloat16"}, device=args.device
    )
    model.max_seq_length = args.max_length

    train_texts, query_texts = common.load_texts(spec, args.query_split)
    print(f"Embedding {len(train_texts)} train docs (passages) ...")
    train_emb = model.encode(
        train_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedding {len(query_texts)} query docs (query prompt) ...")
    query_emb = model.encode(
        query_texts,
        prompt_name="query",
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cosine = train_emb @ query_emb.T  # rows unit-norm => dot == cosine
    scores = -cosine  # loss-diff convention (similar => influential)
    score_path = common.save_scores(scores, out_dir, "qwen3_scores")

    rhos = common.evaluate_lds(
        bank, score_path, out_dir / "qwen3_validate", spec, args.query_split
    )
    common.report(f"{args.model} semantic similarity", rhos)


if __name__ == "__main__":
    main()
