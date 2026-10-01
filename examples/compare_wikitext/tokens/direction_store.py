"""Full-parameter query directions for per-token output influence.

    python examples/compare_wikitext/tokens/direction_store.py \
        <doc-level scores> <out> [--layout <unprojected query store>] [--check N]

Rebuilds the direction a doc-level ``score`` run dotted each training gradient
with: the saved query, preconditioned and normalized as the scorer does it, then
preconditioned once more under split (unit-normalized) scoring. The direction is
mapped back through the query store's random projection and written as an
unprojected ``[O, I]`` query store, which ``score`` with ``attribute_tokens: true``
and ``token_influence: output`` reads. A chunk's rows then add up to its doc-level
score up to one factor per chunk (the inverse norm of its training gradient under
unit normalization; TRAK's row weights). ``--check`` compares the first N training
chunks' gradients along the direction with the doc-level scores.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.collector.collector import (
    HookCollectorBase,
    create_module_projection_matrix,
    global_projection_blocks,
    token_losses,
)
from bergson.config.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.config.config_io import load_subconfig
from bergson.data import load_scores
from bergson.gradients import GradientProcessor, LayerAdapter
from bergson.hessians.preconditioner import load_preconditioner
from bergson.process_grads import normalize_and_aggregate_grads
from bergson.score.score import get_query_grads


def scored_query(scores: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """The query side of the doc-level score, in the query store's gradient space."""
    score_cfg = load_subconfig(scores, "score_cfg", ScoreConfig)
    pre = (
        load_subconfig(scores, "preprocess_cfg", PreprocessConfig) or PreprocessConfig()
    )
    q, qpre = get_query_grads(score_cfg)
    precond = load_preconditioner(
        pre.hessian_path,
        inversion_cfg=pre.inversion_cfg,
        power=-0.5 if pre.unit_normalize else -1.0,
        ev_correction=pre.ev_correction,
        device=device,
    )
    q = {k: v.to(device, torch.float32) for k, v in q.items()}
    if precond is not None and not qpre.hessian_path:
        q = precond.apply(q)
    q = normalize_and_aggregate_grads(
        q,
        list(q),
        unit_normalize=pre.unit_normalize and not qpre.unit_normalize,
        device=device,
        dtype=torch.float32,
    )
    # Split scoring gives the training side H^-1/2 too: <H^-1/2 g, q> = <g, H^-1/2 q>.
    if precond is not None and pre.unit_normalize:
        q = precond.apply(q)
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--layout", default="runs/compare_wikitext/kfac/kfac_query")
    p.add_argument("--check", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda")

    index_cfg = load_subconfig(args.scores, "index_cfg", IndexConfig)
    score_cfg = load_subconfig(args.scores, "score_cfg", ScoreConfig)
    assert not index_cfg.include_bias, "bias columns are not handled"
    q = scored_query(args.scores, device)
    num_q = len(next(iter(q.values())))
    proc = GradientProcessor.load_config(score_cfg.query_path)
    assert not proc.reshape_to_square
    layout = json.load(open(Path(args.layout) / "info.json"))["grad_sizes"]

    model = AutoModelForCausalLM.from_pretrained(index_cfg.model, dtype=torch.float32)
    model = model.to(device).eval()
    base = model.base_model

    def dims(name):
        m = base.get_submodule(name)
        return getattr(m, LayerAdapter.out_attr(m)), getattr(m, LayerAdapter.in_attr(m))

    if not proc.projection_dim:
        out = {n: q[n] for n in layout}
    elif proc.projection_target == "global":
        # Each module's block of the global projection, transposed.
        assert proc.projection_scale == "jl"
        (y,) = q.values()
        m = proc.projection_dim
        out = {}
        for name in layout:
            o, i = dims(name)
            ident = HookCollectorBase.projection_identifier(
                name, "single", proc.projection_seed
            )
            v = torch.zeros(num_q, o * i, device=device)
            for start, stop, block in global_projection_blocks(
                ident, m, o * i, torch.float32, device, proc.projection_type
            ):
                v[:, start:stop] = y @ block
            out[name] = v / math.sqrt(m)
    else:
        # A per-module projection maps G to L G R^T, so its transpose maps Y to L^T Y R.
        k = proc.projection_dim
        out = {}
        for name in layout:
            o, i = dims(name)
            L, R = (
                create_module_projection_matrix(
                    name,
                    role,
                    k,
                    n,
                    torch.float32,
                    device,
                    proc.projection_type,
                    proc.projection_scale,
                    proc.projection_seed,
                )
                for role, n in (("left", o), ("right", i))
            )
            Y = q[name].view(num_q, k, k)
            out[name] = torch.einsum("po,npq,qi->noi", L, Y, R).reshape(num_q, o * i)

    args.out.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(
        args.out / "gradients.bin",
        mode="w+",
        dtype=np.float32,
        shape=(num_q, sum(layout.values())),
    )
    col = 0
    for name, size in layout.items():
        mm[:, col : col + size] = out[name].cpu().numpy()
        col += size
    mm.flush()
    info = {
        "attribute_tokens": False,
        "num_items": num_q,
        "num_grads": num_q,
        "grad_sizes": layout,
        "base_dtype": "float32",
    }
    json.dump(info, open(args.out / "info.json", "w"), indent=1)
    print(f"wrote {num_q} directions over {len(layout)} modules to {args.out}")

    if not args.check:
        return
    docs = load_dataset(index_cfg.data.dataset, split="train")["text"][: args.check]
    tokenizer = AutoTokenizer.from_pretrained(index_cfg.model)
    for w in model.parameters():
        w.requires_grad_(False)
    params = {n: base.get_submodule(n).weight for n in layout}
    for w in params.values():
        w.requires_grad_(True)
    doc_scores = np.asarray(load_scores(args.scores)[:], dtype=np.float64)
    for d, text in enumerate(docs):
        ids = torch.tensor([tokenizer(text)["input_ids"][:1024]], device=device)
        losses = token_losses(index_cfg.loss_fn, model(ids).logits[:, :-1], ids[:, 1:])
        loss = losses.mean() if index_cfg.loss_reduction == "mean" else losses.sum()
        grads = torch.autograd.grad(loss, list(params.values()))
        s = torch.zeros(num_q, device=device, dtype=torch.float64)
        for (name, w), g in zip(params.items(), grads):
            layer = base.get_submodule(name)
            g = (g.T if LayerAdapter.weight_transposed(layer) else g).reshape(-1)
            s += out[name].double() @ g.double()
        ratio = doc_scores[d] / s.cpu().numpy()
        print(
            f"chunk {d}: doc score / direction {np.median(ratio):.6g}, "
            f"spread {np.std(ratio) / abs(np.median(ratio)):.1e} over queries"
        )


if __name__ == "__main__":
    main()
