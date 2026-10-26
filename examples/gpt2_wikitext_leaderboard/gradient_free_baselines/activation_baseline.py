"""Baseline: activation similarity over the attribution target modules.

Represents each document by the mean-pooled input activation to every target
linear module of the bank's fully-trained model (the same modules the gradient
methods attribute through: all Conv1D/Linear except ``lm_head``). Each
per-module pooled vector is L2-normalized, then concatenated, so cosine
similarity between two docs equals the average per-module cosine similarity.

``--hessian`` whitens each pooled vector by ``(A + damping * mean(eig A) I)^-1/2``,
with ``A`` its module's EK-FAC activation factor; ``--projection_dim`` then
applies bergson's per-module right-side projection.

A training doc that is activation-similar to a query is predicted to be
influential (removing it should raise query loss), so the loss-diff-convention
score is ``-cosine`` -- larger => larger predicted loss reduction from keeping
the doc.

Run with (builds the default bank if --bank is omitted):
    B=examples.gpt2_wikitext_leaderboard.gradient_free_baselines
    python -m $B.activation_baseline --bank runs/retrain_bank_path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from bergson.collector.collector import create_module_projection_matrix
from bergson.gradients import LayerAdapter

from . import common


def target_module_names(model: torch.nn.Module) -> list[str]:
    """Linear/Conv1D modules the gradient methods attribute through (not lm_head)."""
    names = []
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            continue
        if isinstance(mod, (Conv1D, torch.nn.Linear)):
            names.append(name)
    return names


def load_factor(hessian: Path, name: str) -> list[dict[str, torch.Tensor]]:
    """Shards of one EK-FAC factor, in rank order."""
    paths = sorted(
        (hessian / name).glob("shard_*.safetensors"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return [load_file(p) for p in paths]


def module_transforms(
    model: torch.nn.Module,
    module_names: list[str],
    hessian: Path | None,
    damping: float,
    projection_dim: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Per-module matrix applied to the pooled activations."""
    modules = dict(model.named_modules())
    prefix = f"{model.base_model_prefix}."
    if hessian is not None:
        vec_shards = load_factor(hessian, "eigen_activation_sharded")
        eig_vals = load_factor(hessian, "factor_eig_a")[0]

    out = {}
    for mn in module_names:
        # Factor keys and projection matrices are named relative to the base model.
        key = mn.removeprefix(prefix)
        transform = None
        if hessian is not None:
            q = torch.cat([s[key] for s in vec_shards]).to(device).float()
            lam = eig_vals[key].to(device).float().clamp_min(0)
            transform = (q * (lam + damping * lam.mean()).rsqrt()) @ q.T
        if projection_dim:
            module = modules[mn]
            i = getattr(module, LayerAdapter.in_attr(module))
            proj = create_module_projection_matrix(
                key,
                "right",
                projection_dim,
                i,
                torch.float32,
                torch.device(device),
                "rademacher",
                "jl",
                None,
            )
            transform = proj if transform is None else proj @ transform
        if transform is not None:
            out[mn] = transform
    return out


@torch.no_grad()
def embed(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    module_names: list[str],
    device: str,
    max_len: int,
    batch_size: int,
    transforms: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    """Per-module-normalized, concatenated mean-pooled activation embeddings."""
    modules = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for mn in module_names:

        def hook(_m, inp, _out, _mn=mn):
            captured[_mn] = inp[0].detach()

        handles.append(modules[mn].register_forward_hook(hook))

    out_rows = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            captured.clear()
            model(**enc)

            per_module = []
            denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
            for mn in module_names:
                act = captured[mn].float()  # [B, T, d]
                pooled = (act * mask).sum(dim=1) / denom  # [B, d]
                if transforms and mn in transforms:
                    pooled = pooled @ transforms[mn].T
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                per_module.append(pooled)
            emb = torch.cat(per_module, dim=-1)  # [B, sum_d]
            out_rows.append(emb.cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(out_rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None, help="re-train bank dir; built if omitted")
    ap.add_argument("--query_split", default=common.DEFAULT_QUERY_SPLIT)
    ap.add_argument(
        "--query_dataset",
        default=None,
        help="query dataset; default = bank train dataset",
    )
    ap.add_argument(
        "--out", default=str(common.REPO / "runs" / "gradient_free_baselines")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--hessian",
        default=None,
        help="EK-FAC hessian dir whose activation factors whiten each module",
    )
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--projection_dim", type=int, default=0)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)

    tokenizer = AutoTokenizer.from_pretrained(spec.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model, dtype=torch.float32, attn_implementation="eager"
    ).to(args.device)
    model.eval()

    module_names = target_module_names(model)
    print(f"Embedding over {len(module_names)} modules, e.g. {module_names[:3]} ...")
    hessian = Path(args.hessian) if args.hessian else None
    transforms = module_transforms(
        model, module_names, hessian, args.damping, args.projection_dim, args.device
    )
    name = "activation"
    if hessian is not None:
        name += "_ekfac"
    if args.projection_dim:
        name += f"_p{args.projection_dim}"

    train_texts, query_texts = common.load_texts(spec, query_dataset, args.query_split)
    print(f"Embedding {len(train_texts)} train docs ...")
    train_emb = embed(
        train_texts,
        model,
        tokenizer,
        module_names,
        args.device,
        args.max_len,
        args.batch_size,
        transforms,
    )
    print(f"Embedding {len(query_texts)} query docs ...")
    query_emb = embed(
        query_texts,
        model,
        tokenizer,
        module_names,
        args.device,
        args.max_len,
        args.batch_size,
        transforms,
    )

    # Cosine similarity (rows already unit-norm per module; renormalize the
    # concatenation so cosine == mean per-module cosine).
    tn = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    qn = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    cosine = tn @ qn.T  # [num_train, num_query]

    scores = -cosine  # loss-diff convention (similar => influential)
    score_path = common.save_scores(scores, out_dir, f"{name}_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / f"{name}_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report(name, rhos)


if __name__ == "__main__":
    main()
