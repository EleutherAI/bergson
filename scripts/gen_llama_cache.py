"""Generate gradient precision caches for Llama-2-7b.

Run on a node with >= 30GB free VRAM on a single GPU.

Usage:
    python scripts/gen_llama_cache.py

Results are saved under runs/mp_caches/meta-llama_Llama-2-7b-hf/.
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

MODEL = "meta-llama/Llama-2-7b-hf"
DATASET = "NeelNanda/pile-10k"
SPLIT = "train"
N_SAMPLES = 100
MAX_LENGTH = 1024
PROJECTION_DIM = 16
TOKEN_BATCH_SIZE = 1024

CONFIGS = [
    ("pure bf16", "bf16", False, torch.bfloat16),
    ("mp bf16", "bf16", True, torch.float32),
    ("pure fp16", "fp16", False, torch.float16),
    ("mp fp16", "fp16", True, torch.float32),
]

out_dir = Path(f"runs/mp_caches/{MODEL.replace('/', '_')}")
out_dir.mkdir(parents=True, exist_ok=True)


def collect_grads(model, dataset, precision, mixed_precision):
    cfg = IndexConfig(
        run_path=f"/tmp/gen_cache_llama_{precision}_mp{mixed_precision}",
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


def metrics_vs_ref(grads, fp32_ref):
    flat = grads.flatten()
    ref = fp32_ref.flatten()
    cos = F.cosine_similarity(flat, ref, dim=0).item()
    l2 = (flat - ref).norm().item() / ref.norm().item()
    return cos, l2


# ── Tokenize dataset ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL)
max_length = min(getattr(tokenizer, "model_max_length", MAX_LENGTH), MAX_LENGTH)
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

# ── FP32 reference ───────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  {MODEL}")
print(f"{'=' * 60}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32
).cuda()
fp32_grads, module_name = collect_grads(model, dataset, "fp32", False)
fp32_ref = torch.stack(list(fp32_grads)).float()
np.save(out_dir / "fp32.npy", fp32_ref.numpy())
print(f"  fp32 reference: module={module_name}, shape={fp32_grads[0].shape}")
del model
gc.collect()
torch.cuda.empty_cache()

# ── Half-precision configs ───────────────────────────────────────────
rows: list[tuple[str, float, float]] = []

for label, precision, mixed_precision, model_dtype in CONFIGS:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=model_dtype
    ).cuda()
    grads, _ = collect_grads(model, dataset, precision, mixed_precision)
    grads_f32 = torch.stack([g.float() for g in grads])

    fname = label.replace(" ", "_")
    np.save(out_dir / f"{fname}.npy", grads_f32.numpy())

    cos, l2 = metrics_vs_ref(grads_f32, fp32_ref)
    rows.append((label, cos, l2))
    print(f"  {label:10s} vs fp32 — cosine: {cos:.8f}, rel L2: {l2:.8f}")

    del model, grads, grads_f32
    gc.collect()
    torch.cuda.empty_cache()

# ── Summary ──────────────────────────────────────────────────────────
del fp32_grads, fp32_ref
print(f"\n{'=' * 60}")
print("  Summary")
print(f"{'=' * 60}\n")
print(f"| Config | Cosine sim | Rel L2 error |")
print(f"|---|---|---|")
for label, cos, l2 in rows:
    print(f"| {label} | {cos:.6f} | {l2:.6f} |")
