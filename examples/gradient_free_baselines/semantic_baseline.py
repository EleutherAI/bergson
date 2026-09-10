"""Baseline: semantic-search similarity with a Jina AI embedding model.

Embeds each training document as a retrieval document and each query document
as a retrieval query with ``jinaai/jina-embeddings-v5-text-small`` (Jina's
current text embedder; ``--model`` swaps in another, e.g. the older
``jinaai/jina-embeddings-v3``), then scores a train doc against a query by
their cosine similarity -- ordinary dense semantic search. This ignores the
attributed model entirely; it is a pure content-similarity baseline.

A training doc semantically similar to a query is predicted to be influential
(removing it should raise query loss), so the loss-diff-convention score is
``-cosine``.

jina-embeddings-v3 ships custom modeling code that predates transformers 5.x,
so ``load_model`` patches two load-time incompatibilities (see there) when that
model is selected.

Run with (builds the default bank if --bank is omitted):
    python -m examples.gradient_free_baselines.semantic_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

from . import common

MODEL = "jinaai/jina-embeddings-v5-text-small"

# jina-v5 takes one ``task`` plus a ``prompt_name``; jina-v3 folds both into
# the task name.
TASKS = {
    "v5": {
        "query": dict(task="retrieval", prompt_name="query"),
        "document": dict(task="retrieval", prompt_name="document"),
    },
    "v3": {
        "query": dict(task="retrieval.query"),
        "document": dict(task="retrieval.passage"),
    },
}


def api_version(model_name: str) -> str:
    return "v3" if "jina-embeddings-v3" in model_name else "v5"


def load_model(model_name: str, device: str):
    if api_version(model_name) == "v3":
        # jina-embeddings-v3's custom code predates transformers 5.x; two fixes:
        # (1) from_pretrained reads all_tied_weights_keys, which the custom class
        #     never defines -- give a benign empty default so load doesn't crash.
        if not isinstance(
            getattr(PreTrainedModel, "all_tied_weights_keys", None), property
        ):
            PreTrainedModel.all_tied_weights_keys = {}
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.float32
    )

    if api_version(model_name) == "v3":
        # (2) its LoRA task adapters leave the per-forward lora_dropout_mask
        #     buffers uninitialized (NaN), so every embedding comes out NaN. In
        #     eval there is no dropout, so reset them to ones.
        for name, buf in model.named_buffers():
            if "lora_dropout_mask" in name:
                buf.data = torch.ones_like(buf)
    return model.to(device).eval()


@torch.no_grad()
def encode(
    model, texts: list[str], role: str, batch_size: int, version: str
) -> np.ndarray:
    """L2-normalized embeddings via jina's task-specific encode API."""
    kwargs = TASKS[version][role]
    out = []
    for start in range(0, len(texts), batch_size):
        emb = model.encode(texts[start : start + batch_size], **kwargs)
        if torch.is_tensor(emb):
            emb = emb.float().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32)
        emb /= np.clip(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12, None)
        out.append(emb)
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument(
        "--query_dataset",
        default=None,
        help="query dataset; default = bank train dataset",
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--out", default=str(common.REPO / "runs" / "gradient_free_baselines")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)
    version = api_version(args.model)
    model = load_model(args.model, args.device)

    train_texts, query_texts = common.load_texts(spec, query_dataset, args.query_split)
    print(f"Embedding {len(train_texts)} train docs (document) ...")
    train_emb = encode(model, train_texts, "document", args.batch_size, version)
    print(f"Embedding {len(query_texts)} query docs (query) ...")
    query_emb = encode(model, query_texts, "query", args.batch_size, version)

    cosine = train_emb @ query_emb.T  # rows unit-norm => dot == cosine
    scores = -cosine  # loss-diff convention (similar => influential)
    score_path = common.save_scores(scores, out_dir, "semantic_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / "semantic_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report(f"{args.model} semantic similarity", rhos)


if __name__ == "__main__":
    main()
