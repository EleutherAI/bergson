"""EK-FAC scores split into module x eigenvalue-band contributions.

    python examples/compare_wikitext/ridge/band_contributions.py \
        --hessian runs/compare_wikitext/ekfac/hessian/kfac \
        --out runs/compare_wikitext/ridge/contributions.npy \
        [--check runs/compare_wikitext/ekfac/scores]

Each training and query gradient is whitened by the EK-FAC inverse square root
(relative damping 0.1, eigenvalue correction) and rotated into the EK-FAC
eigenbasis. Per module, the eigenbasis coordinates are sorted by corrected
eigenvalue (largest first) and cut into bands at ``--edges`` (fractions of the
module's coordinates). ``contributions[d, m, b, q]`` is the inner product of
training doc ``d`` and query ``q`` restricted to module ``m``, band ``b``; summed
over modules and bands it is the EK-FAC score. ``--check`` compares that sum to
an EK-FAC score store.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPT2LMHeadModel

from bergson.config import InversionConfig
from bergson.data import load_scores
from bergson.hessians.preconditioner import FactoredPreconditioner

DATASET = "EleutherAI/bergson-wikitext-512-chunks"
EDGES = "0,0.001,0.01,0.03,0.1,0.25,0.5,1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="runs/compare_wikitext/interval/exported/checkpoint-72"
    )
    ap.add_argument("--hessian", required=True, type=Path)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--query_split", default="test[0:50]")
    ap.add_argument("--edges", default=EDGES)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", type=Path, help="EK-FAC score store to compare against")
    args = ap.parse_args()

    edges = [float(e) for e in args.edges.split(",")]
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(dev).eval().requires_grad_(False)
    # Backpropagate through the network without accumulating weight gradients.
    model.transformer.wte.weight.requires_grad_(True)
    mods = {
        n.removeprefix("transformer."): m
        for n, m in model.named_modules()
        if type(m).__name__ == "Conv1D"
    }
    names = list(mods)
    pre = FactoredPreconditioner.from_path(
        args.hessian,
        inversion_cfg=InversionConfig(inversion="damped_inverse", damping_factor=0.1),
        power=-0.5,
        ev_correction=True,
        device=dev,
    )
    perm = {n: pre.lambdas[n].flatten().argsort(descending=True) for n in names}
    cuts = {n: [round(e * perm[n].numel()) for e in edges] for n in names}

    acts: dict[str, list] = {}

    def hook(name):
        def fwd(mod, inp, out):
            acts[name] = [inp[0].detach(), None]
            out.register_hook(lambda g: acts[name].__setitem__(1, g.detach()))

        return fwd

    for n, m in mods.items():
        m.register_forward_hook(hook(n))

    def coords(texts: list[str]) -> dict[str, torch.Tensor]:
        """Whitened per-doc gradients in the eigenbasis, by descending eigenvalue."""
        enc = tok(texts, truncation=True, max_length=1024)["input_ids"]
        ids = torch.full((len(enc), max(map(len, enc))), tok.eos_token_id)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(enc):
            ids[i, : len(row)] = labels[i, : len(row)] = torch.tensor(row)
        ids, labels = ids.to(dev), labels.to(dev)
        with torch.enable_grad():
            logits = model(input_ids=ids).logits[:, :-1]
            F.cross_entropy(
                logits.flatten(0, 1), labels[:, 1:].flatten(), reduction="sum"
            ).backward()
        model.transformer.wte.weight.grad = None
        out = {}
        with torch.no_grad():
            for n in names:
                a, d = acts[n]
                g = torch.einsum("bto,bti->boi", d, a)
                w = pre.apply({n: g.flatten(1)})[n].view_as(g)
                rot = pre.eigen_g[n].T @ w @ pre.eigen_a[n]
                out[n] = rot.flatten(1)[:, perm[n]].half()
        return out

    queries = load_dataset(args.dataset, split=args.query_split)["text"]
    train = load_dataset(args.dataset, split=args.train_split)["text"]
    fq: dict[str, list] = {n: [] for n in names}
    for i in range(0, len(queries), args.batch_size):
        for n, c in coords(queries[i : i + args.batch_size]).items():
            fq[n].append(c)
    q = {n: torch.cat(v).float() for n, v in fq.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nb = len(edges) - 1
    shape = (len(train), len(names), nb, len(queries))
    contrib = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float32, shape=shape
    )
    for i in range(0, len(train), args.batch_size):
        f = coords(train[i : i + args.batch_size])
        for mi, n in enumerate(names):
            c = cuts[n]
            for b in range(nb):
                band = f[n][:, c[b] : c[b + 1]].float() @ q[n][:, c[b] : c[b + 1]].T
                contrib[i : i + args.batch_size, mi, b] = band.cpu().numpy()
        if i % (args.batch_size * 100) == 0:
            print(f"{i + args.batch_size}/{len(train)} docs", flush=True)
    contrib.flush()
    meta = {"modules": names, "edges": edges, "model": args.model}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    if args.check is not None:
        total = contrib.sum((1, 2))
        ref = np.asarray(load_scores(args.check)[:], dtype=np.float64)
        rho = [spearmanr(total[:, j], ref[:, j]).statistic for j in range(ref.shape[1])]
        print(f"Spearman of the band sum against {args.check}: mean {np.mean(rho):.5f}")


if __name__ == "__main__":
    main()
