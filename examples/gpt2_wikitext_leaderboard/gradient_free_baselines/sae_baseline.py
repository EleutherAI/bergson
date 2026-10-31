"""Baseline: sparse autoencoder feature similarity with Gemma Scope 2.

Each document is the mean over its tokens (BOS excluded) of the Gemma Scope 2
residual-stream SAE features of ``google/gemma-3-27b-pt`` after layer 31 (the
release's 50%-depth layer, width 262k, medium L0); a train doc scores against a
query by cosine similarity, so the loss-diff-convention score is ``-cosine``.
Documents are tokenized with Gemma's own tokenizer.

Only the embeddings and the layers up to the SAE's layer run on the GPU: the rest
of the model is placed on the CPU and the forward pass stops at the SAE's layer.
Documents are encoded one at a time, as in the per-token scores of
``examples/gpt2_wikitext_leaderboard/tokens/token_baselines.py``, so a chunk's
per-token rows add up to its document score.

Run with (builds the default bank if --bank is omitted):
    B=examples.gpt2_wikitext_leaderboard.gradient_free_baselines
    python -m $B.sae_baseline --bank runs/retrain_bank_path

``--validate N`` instead prints the SAE's fraction of variance explained and mean
L0 on the first N training docs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from . import common

MODEL = "google/gemma-3-27b-pt"
SAE_REPO = "google/gemma-scope-2-27b-pt"
SAE_ID = "resid_post/layer_31_width_262k_l0_medium"


class _Stop(Exception):
    pass


class GemmaSAE:
    """A Gemma 3 model truncated at a Gemma Scope 2 SAE's layer, and the SAE."""

    def __init__(self, model=MODEL, sae_repo=SAE_REPO, sae_id=SAE_ID, device="cuda:0"):
        cfg = json.load(open(hf_hub_download(sae_repo, f"{sae_id}/config.json")))
        hook = cfg["hf_hook_point_in"]  # e.g. model.layers.31.output
        self.layer = int(hook.split(".")[2])
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        n_layers = AutoConfig.from_pretrained(model).text_config.num_hidden_layers
        device_map = {"model.vision_tower": "cpu", "model.multi_modal_projector": "cpu"}
        device_map["model.language_model.embed_tokens"] = device
        device_map["model.language_model.rotary_emb"] = device
        device_map["model.language_model.rotary_emb_local"] = device
        for i in range(n_layers):
            on_gpu = i <= self.layer
            device_map[f"model.language_model.layers.{i}"] = device if on_gpu else "cpu"
        device_map["model.language_model.norm"] = "cpu"
        device_map["lm_head"] = "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self.text_model = self.model.model.language_model
        self.text_model.layers[self.layer].register_forward_hook(self._capture)

        p = load_file(hf_hub_download(sae_repo, f"{sae_id}/params.safetensors"))
        self.w_enc = p["w_enc"].to(device)
        self.b_enc = p["b_enc"].to(device)
        self.threshold = p["threshold"].to(device)
        self.w_dec, self.b_dec = p["w_dec"], p["b_dec"]  # used only by validate

    def _capture(self, _module, _inp, out):
        self._resid = out[0] if isinstance(out, tuple) else out
        raise _Stop

    @torch.no_grad()
    def residuals(self, texts, max_length=2048):
        """Per-text residual stream at the SAE's layer, BOS dropped, with each
        kept token's character offsets."""
        enc = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        try:
            self.text_model(**enc)
        except _Stop:
            pass
        out = []
        for i in range(len(texts)):
            keep = enc["attention_mask"][i].bool()
            keep &= enc["input_ids"][i] != self.tokenizer.bos_token_id
            out.append((self._resid[i][keep].float(), offsets[i][keep.cpu()].tolist()))
        return out

    def encode(self, x):
        """JumpReLU features of residual vectors ``x`` [T, d]."""
        pre = x @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def pooled(self, texts, batch_size):
        """Mean SAE features per text, [N, width]."""
        out = []
        for i in tqdm(range(0, len(texts), batch_size)):
            for x, _ in self.residuals(texts[i : i + batch_size]):
                out.append(self.encode(x).mean(0))
        return torch.stack(out)

    @torch.no_grad()
    def validate(self, texts, batch_size):
        """Fraction of variance explained and mean L0 over the texts' tokens."""
        w_dec, b_dec = self.w_dec.to(self.device), self.b_dec.to(self.device)
        err = var = l0 = n = 0.0
        xs = []
        for i in range(0, len(texts), batch_size):
            for x, _ in self.residuals(texts[i : i + batch_size]):
                f = self.encode(x)
                err += ((f @ w_dec + b_dec - x) ** 2).sum().item()
                l0 += (f > 0).sum().item()
                n += len(x)
                xs.append(x)
        x = torch.cat(xs)
        var = ((x - x.mean(0)) ** 2).sum().item()
        return 1 - err / var, l0 / n, int(n)


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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sae_repo", default=SAE_REPO)
    ap.add_argument("--sae_id", default=SAE_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--validate", type=int, default=0)
    args = ap.parse_args()

    bank = common.ensure_bank(args.bank)
    spec = common.read_bank_spec(bank)
    query_dataset = args.query_dataset or spec.dataset
    out_dir = Path(args.out)

    sae = GemmaSAE(args.model, args.sae_repo, args.sae_id, args.device)
    train_texts, query_texts = common.load_texts(spec, query_dataset, args.query_split)
    if args.validate:
        fve, l0, n = sae.validate(train_texts[: args.validate], args.batch_size)
        print(f"{args.sae_id}: FVE {fve:.4f}, mean L0 {l0:.1f} over {n} tokens")
        return

    print(f"Encoding {len(query_texts)} query docs ...")
    q = torch.nn.functional.normalize(sae.pooled(query_texts, args.batch_size), dim=-1)
    print(f"Encoding {len(train_texts)} train docs ...")
    cosine = []
    for i in range(0, len(train_texts), 256):
        d = sae.pooled(train_texts[i : i + 256], args.batch_size)
        cosine.append((torch.nn.functional.normalize(d, dim=-1) @ q.T).cpu())
    scores = -torch.cat(cosine).numpy()  # loss-diff convention
    score_path = common.save_scores(scores.astype(np.float32), out_dir, "sae_scores")

    rhos = common.evaluate_lds(
        bank,
        score_path,
        out_dir / "sae_validate",
        spec,
        query_dataset,
        args.query_split,
    )
    common.report(f"{args.sae_id} SAE feature similarity", rhos)


if __name__ == "__main__":
    main()
