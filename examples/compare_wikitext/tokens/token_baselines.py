"""Per-token scores for the gradient-free baselines.

    python examples/compare_wikitext/tokens/token_baselines.py \
        {activation,bm25,qwen3} <out> [--shard i --num_shards n] [--merge]

Row ``t`` of a chunk scores the loss on token ``t + 1``, as in the forward-mode
rows of the gradient methods, and a chunk's rows add up to its document score
(``-similarity``, from ``examples/gradient_free_baselines``):

- activation: a document's pooled input activation to each attributed module is a
  mean over positions, so position ``t``'s share of the cosine goes to row ``t``,
  the loss position ``t`` predicts.
- bm25: a query term's weight in a chunk is split evenly over the term's
  occurrences and each occurrence's share over the GPT-2 tokens it spans; token
  ``j``'s share goes to row ``j - 1``, the loss on token ``j``.
- qwen3: Qwen3-Embedding pools the last token of a causal model, so position
  ``k``'s hidden state embeds the prefix up to token ``k``; token ``k``'s share is
  the change in the prefix embedding's cosine when it is appended, mapped to the
  GPT-2 tokens covering its characters.

The first token of a chunk has no loss row, so its share is dropped. Shards write
``<out>/shard_<i>.npy``; ``--merge`` writes ``<out>/scores`` in the layout of a
per-token score store and prints how well the chunk sums match the doc scores.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RUNS = Path("runs/compare_wikitext")
LAYOUT = RUNS / "tokens" / "ekfac_tokens"
DATASET = "EleutherAI/bergson-wikitext-512-chunks"


def gpt2_spans(texts):
    tok = AutoTokenizer.from_pretrained("gpt2")
    return [tok(t, return_offsets_mapping=True)["offset_mapping"] for t in texts]


def spread(rows, spans, lo, hi, value):
    """Add ``value`` to the rows of the GPT-2 tokens overlapping characters
    ``[lo, hi)``, split evenly; token ``j`` owns row ``j - 1``."""
    hit = [
        j for j, (s, e) in enumerate(spans) if s < hi and e > lo and 0 < j <= len(rows)
    ]
    for j in hit:
        rows[j - 1] += value / len(hit)


def activation_rows(queries, shard):
    from examples.gradient_free_baselines import activation_baseline as ab
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        RUNS / "random" / "retrained" / "base",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.cuda().eval()
    names = ab.target_module_names(model)
    q = ab.embed(queries, model, tok, names, "cuda", 1024, 32)
    q = torch.tensor(q / np.linalg.norm(q, axis=1, keepdims=True), device="cuda")
    # Conv1D weights are [in, out]; each module's block of q is its input width.
    qs = torch.split(q, [model.get_submodule(n).weight.shape[0] for n in names], 1)
    mods = dict(model.named_modules())
    out = []
    for text in shard:
        cap = {}
        hooks = [
            mods[n].register_forward_hook(
                lambda _m, i, _o, n=n: cap.__setitem__(n, i[0][0].float())
            )
            for n in names
        ]
        enc = tok([text], return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            model(**enc.to("cuda"))
        for h in hooks:
            h.remove()
        T = len(cap[names[0]])
        share = torch.zeros(T, len(queries), device="cuda", dtype=torch.float64)
        for n, qm in zip(names, qs):
            a = cap[n]
            share += (a @ qm.T).double() / (T * a.mean(0).norm())
        out.append((-share / np.sqrt(len(names))).cpu().numpy())
    return out


def bm25_rows(train, queries, idx, k1=1.5, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer()
    tf = vec.fit_transform(train).astype(np.float64).tocsr()
    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = doc_len.mean()
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    idf = np.log((tf.shape[0] - df + 0.5) / (df + 0.5) + 1.0)
    qbin = (vec.transform(queries) > 0).astype(np.float64).toarray()
    pattern = re.compile(vec.token_pattern)
    out = []
    for d, spans in zip(idx, gpt2_spans([train[d] for d in idx])):
        rows = np.zeros((len(spans), len(queries)))
        row = tf.getrow(d)
        counts = dict(zip(row.indices, row.data))
        for m in pattern.finditer(train[d]):
            w = vec.vocabulary_.get(m.group().lower())
            if w is None or not qbin[:, w].any():
                continue
            t = counts[w]
            weight = idf[w] * t * (k1 + 1) / (t + k1 * (1 - b + b * doc_len[d] / avgdl))
            spread(rows, spans, m.start(), m.end(), -weight / t * qbin[:, w])
        out.append(rows)
    return out


def qwen3_rows(queries, shard):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={"torch_dtype": "bfloat16"},
        device="cuda",
    )
    st.max_seq_length = 512
    e_q = st.encode(
        queries, prompt_name="query", batch_size=8, normalize_embeddings=True
    )
    e_q = torch.tensor(e_q, device="cuda").float()
    lm = st[0].auto_model
    out = []
    for text, spans in zip(shard, gpt2_spans(shard)):
        enc = st.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            h = lm(**enc.to("cuda")).last_hidden_state[0]
        c = torch.nn.functional.normalize(h.float(), dim=-1) @ e_q.T  # [T, Q]
        delta = torch.diff(c, dim=0, prepend=torch.zeros_like(c[:1])).cpu().numpy()
        rows = np.zeros((len(spans), len(queries)))
        last = None
        for k, (s, e) in enumerate(offsets):
            if e > s:
                last = (s, e)
            if last is not None:  # special tokens join the token before them
                spread(rows, spans, *last, -delta[k])
        out.append(rows)
    return out


def merge(args, train, queries):
    from bergson.data import load_scores
    from bergson.score.score_writer import save_token_scores

    offsets = load_scores(LAYOUT).offsets
    n_rows = np.diff(offsets)
    n = args.num_shards
    shards = [np.load(args.out / f"shard_{i}.npy", allow_pickle=True) for i in range(n)]
    rows = np.zeros((offsets[-1], len(queries)), dtype=np.float32)
    sums = np.zeros((len(train), len(queries)))
    for d in range(len(train)):
        r = shards[d % n][d // n]
        sums[d] = r.sum(0)
        rows[offsets[d] : offsets[d + 1]] = r[: n_rows[d]]
    doc = np.asarray(
        load_scores(RUNS / "baselines" / f"{args.method}_scores" / "scores")[:]
    )
    corr = np.mean(
        [np.corrcoef(sums[:, q], doc[:, q])[0, 1] for q in range(len(queries))]
    )
    print(f"{args.method}: chunk sums vs doc scores, corr {corr:.6f}")
    save_token_scores(args.out / "scores", rows, offsets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["activation", "bm25", "qwen3"])
    p.add_argument("out", type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATASET, split="train")["text"]
    queries = load_dataset(DATASET, split="test[0:50]")["text"]
    if args.merge:
        return merge(args, train, queries)

    idx = list(range(args.shard, len(train), args.num_shards))
    shard = [train[d] for d in idx]
    if args.method == "activation":
        rows = activation_rows(queries, shard)
    elif args.method == "bm25":
        rows = bm25_rows(train, queries, idx)
    else:
        rows = qwen3_rows(queries, shard)
    out = np.empty(len(rows), dtype=object)
    out[:] = rows
    np.save(args.out / f"shard_{args.shard}.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
