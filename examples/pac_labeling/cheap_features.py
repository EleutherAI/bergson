"""Forward-pass features for PAC labeling: per-document loss under the
trained model, activation cosine to each query, and dense-embedding cosine to
each query.

Usage:
    python examples/pac_labeling/cheap_features.py --model <run>/base/model \
        --train <train.hf> --query <query.hf> --out <dir> \
        [--embedder BAAI/bge-base-en-v1.5]

Writes ``doc_loss.npy`` (mean token loss per document), and two score
directories in the loss-signed convention (more similar = more negative):
``activation_scores`` (cosine of per-module mean-pooled input activations of
every Linear/Conv1D module except ``lm_head``, each module L2-normalized and
concatenated) and ``semantic_scores`` (cosine of embeddings of the decoded
text; queries carry the embedder's retrieval instruction).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from bergson.score.score_writer import save_sequence_scores

BGE_QUERY = "Represent this sentence for searching relevant passages: "


def target_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (Conv1D, torch.nn.Linear)) and not n.endswith("lm_head")
    ]


@torch.no_grad()
def activations_and_loss(
    model, ids: torch.Tensor, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document mean token loss and the concatenated per-module
    L2-normalized mean input activation."""
    pooled: dict[str, torch.Tensor] = {}
    handles = []
    for name, mod in target_modules(model):

        def hook(_, inputs, name=name):
            pooled[name] = inputs[0].float().mean(1)

        handles.append(mod.register_forward_pre_hook(hook))
    feats, losses = [], []
    for start in range(0, len(ids), batch_size):
        x = ids[start : start + batch_size].to(device)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
        losses.append(nll.mean(1).cpu())
        parts = [
            torch.nn.functional.normalize(pooled[n], dim=-1)
            for n, _ in target_modules(model)
        ]
        feats.append(torch.cat(parts, -1).cpu())
    for h in handles:
        h.remove()
    feats = torch.cat(feats).numpy()
    feats /= np.sqrt(len(target_modules(model)))
    return torch.cat(losses).numpy(), feats


@torch.no_grad()
def embed(texts: list[str], name: str, batch_size: int, device: str) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tok(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        emb = model(**enc).last_hidden_state[:, 0].float()
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_from_disk(args.train)
    query = load_from_disk(args.query)
    train_ids = torch.tensor(train["input_ids"], dtype=torch.long)
    query_ids = torch.tensor(query["input_ids"], dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    doc_loss, train_act = activations_and_loss(
        model, train_ids, args.batch_size, args.device
    )
    _, query_act = activations_and_loss(model, query_ids, args.batch_size, args.device)
    np.save(args.out / "doc_loss.npy", doc_loss)
    save_sequence_scores(args.out / "activation_scores", -(train_act @ query_act.T))
    del model
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.model)
    train_text = tok.batch_decode(train_ids, skip_special_tokens=True)
    query_text = tok.batch_decode(query_ids, skip_special_tokens=True)
    prefix = BGE_QUERY if "bge" in args.embedder else ""
    train_emb = embed(train_text, args.embedder, args.batch_size, args.device)
    query_emb = embed(
        [prefix + t for t in query_text], args.embedder, args.batch_size, args.device
    )
    save_sequence_scores(args.out / "semantic_scores", -(train_emb @ query_emb.T))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
