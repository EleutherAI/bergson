"""Compare gradient precision across dtype/mixed-precision configurations.

For each model, collects per-sample projected gradients under:
  - fp32 (reference)
  - pure bf16  / mixed-precision bf16  (fp32 weights + bf16 autocast)
  - pure fp16  / mixed-precision fp16  (fp32 weights + fp16 autocast)

Caches are saved per-model under runs/mp_caches/<model_slug>/.
Results are printed as a markdown table at the end.

Usage:
    python scripts/gen_bf16_cache.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import CollectorComputer, GradientProcessor, InMemoryCollector
from bergson.config import DataConfig, IndexConfig
from bergson.data import load_data_string, tokenize

MODELS = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-160m",
    "allenai/OLMo-2-0425-1B",
]
DATASET = "NeelNanda/pile-10k"
SPLIT = "train"
N_SAMPLES = 100
MAX_LENGTH = 1024
PROJECTION_DIM = 16
TOKEN_BATCH_SIZE = 1024

# (label, precision_flag, mixed_precision_flag, model_dtype)
CONFIGS = [
    ("pure bf16", "bf16", False, torch.bfloat16),
    ("mp bf16", "bf16", True, torch.float32),
    ("pure fp16", "fp16", False, torch.float16),
    ("mp fp16", "fp16", True, torch.float32),
]


def collect_grads(model, dataset, precision, mixed_precision):
    """Collect projected per-sample gradients for a single module."""
    cfg = IndexConfig(
        run_path=f"/tmp/gen_cache_{precision}_mp{mixed_precision}",
        skip_preconditioners=True,
        token_batch_size=TOKEN_BATCH_SIZE,
        precision=precision,
        mixed_precision=mixed_precision,
        projection_dim=PROJECTION_DIM,
    )
    Path(cfg.partial_run_path).mkdir(parents=True, exist_ok=True)
    collector = InMemoryCollector(
        model.base_model,
        processor=GradientProcessor(projection_dim=cfg.projection_dim),
        data=dataset,
        cfg=cfg,
    )
    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        batches=[[idx] for idx in range(len(dataset))],
        cfg=cfg,
    )
    label = f"{precision} (mp={mixed_precision})"
    computer.run_with_collector_hooks(desc=f"Collecting {label} gradients")
    first_module = list(collector.gradients.keys())[0]
    return collector.gradients[first_module], first_module


def load_model(model_name, dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).cuda()


def metrics_vs_ref(grads, fp32_ref):
    """Cosine similarity and relative L2 error vs fp32 reference."""
    flat = grads.flatten()
    ref = fp32_ref.flatten()
    cos = F.cosine_similarity(flat, ref, dim=0).item()
    l2 = (flat - ref).norm().item() / ref.norm().item()
    return cos, l2


# ── Summary table rows ──────────────────────────────────────────────────
rows: list[tuple[str, str, float, float]] = []

for model_name in MODELS:
    slug = model_name.replace("/", "_")
    out_dir = Path(f"runs/mp_caches/{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    # ── Tokenize dataset (once per model) ────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = min(
        getattr(tokenizer, "model_max_length", MAX_LENGTH),
        MAX_LENGTH,
    )
    ds = load_data_string(DATASET, SPLIT)
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(
            args=DataConfig(truncation=True),
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )
    dataset = ds.select(range(N_SAMPLES))

    # ── FP32 reference ───────────────────────────────────────────────
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = load_model(model_name, torch.float32)
    fp32_grads, module_name = collect_grads(model, dataset, "fp32", False)
    fp32_ref = torch.stack(list(fp32_grads)).float()
    np.save(out_dir / "fp32.npy", fp32_ref.numpy())
    print(f"  fp32 reference: module={module_name}, shape={fp32_grads[0].shape}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Each half-precision config ───────────────────────────────────
    for label, precision, mixed_precision, model_dtype in CONFIGS:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = load_model(model_name, model_dtype)
        grads, _ = collect_grads(model, dataset, precision, mixed_precision)
        grads_f32 = torch.stack([g.float() for g in grads])

        fname = label.replace(" ", "_")
        np.save(out_dir / f"{fname}.npy", grads_f32.numpy())

        cos, l2 = metrics_vs_ref(grads_f32, fp32_ref)
        rows.append((model_name, label, cos, l2))
        print(f"  {label:10s} vs fp32 — cosine: {cos:.8f}, rel L2: {l2:.8f}")

        del model, grads, grads_f32
        gc.collect()
        torch.cuda.empty_cache()

    del fp32_grads, fp32_ref
    gc.collect()
    torch.cuda.empty_cache()

# ── Print markdown table ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  Summary")
print(f"{'=' * 60}\n")
print("| Model | Config | Cosine sim | Rel L2 error |")
print("|---|---|---|---|")
for model_name, label, cos, l2 in rows:
    print(f"| {model_name} | {label} | {cos:.6f} | {l2:.6f} |")
