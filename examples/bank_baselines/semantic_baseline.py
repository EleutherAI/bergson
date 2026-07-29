"""Baseline: semantic-search similarity with a Jina AI embedding model.

Embeds each training document as a retrieval passage and each query document
as a retrieval query with ``jinaai/jina-embeddings-v3`` (a strong 570M-param,
1024-dim, 8192-context text embedder), then scores a train doc against a query
by their cosine similarity -- ordinary dense semantic search. This ignores the
attributed model entirely; it is a pure content-similarity baseline.

A training doc semantically similar to a query is predicted to be influential
(removing it should raise query loss), so the loss-diff-convention score is
``-cosine``.

jina-embeddings-v3 ships custom modeling code that predates transformers 5.x,
so ``load_model`` patches two load-time incompatibilities (see there) to run it
for inference.

Run with (builds the default bank if --bank is omitted):
    python -m examples.bank_baselines.semantic_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

from . import common

MODEL = "jinaai/jina-embeddings-v3"


def load_model(device: str):
    # jina-embeddings-v3's custom code predates transformers 5.x; two fixes:
    # (1) from_pretrained reads all_tied_weights_keys, which the custom class
    #     never defines -- give a benign empty default so load doesn't crash.
    if not isinstance(
        getattr(PreTrainedModel, "all_tied_weights_keys", None), property
    ):
        PreTrainedModel.all_tied_weights_keys = {}
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32
    )

    # (2) its LoRA task adapters leave the per-forward lora_dropout_mask buffers
    #     uninitialized (NaN), so every embedding comes out NaN. In eval there
    #     is no dropout, so reset them to ones.
    for name, buf in model.named_buffers():
        if "lora_dropout_mask" in name:
            buf.data = torch.ones_like(buf)
    return model.to(device).eval()


@torch.no_grad()
def encode(model, texts: list[str], task: str, batch_size: int) -> np.ndarray:
    """L2-normalized embeddings via jina's task-specific encode API."""
    out = []
    for start in range(0, len(texts), batch_size):
        emb = model.encode(
            texts[start : start + batch_size],
            task=task,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        out.append(np.asarray(emb, dtype=np.float32))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument("--out", default=str(common.REPO / "runs" / "bank_baselines"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    out_dir = Path(args.out)
    model = load_model(args.device)

    train_texts, query_texts = common.load_texts(spec, args.query_split)
    print(f"Embedding {len(train_texts)} train docs (retrieval.passage) ...")
    train_emb = encode(model, train_texts, "retrieval.passage", args.batch_size)
    print(f"Embedding {len(query_texts)} query docs (retrieval.query) ...")
    query_emb = encode(model, query_texts, "retrieval.query", args.batch_size)

    cosine = train_emb @ query_emb.T  # rows unit-norm => dot == cosine
    scores = -cosine  # loss-diff convention (similar => influential)
    score_path = common.save_scores(scores, out_dir, "semantic_scores")

    rhos = common.evaluate_lds(
        bank, score_path, out_dir / "semantic_validate", spec, args.query_split
    )
    common.report("Jina v3 semantic similarity", rhos)


if __name__ == "__main__":
    main()
