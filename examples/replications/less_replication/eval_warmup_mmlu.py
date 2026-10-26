"""Evaluate warmup model on MMLU. Launch with torchrun.

Usage:
    torchrun --nproc_per_node 8 scripts/eval_warmup_mmlu.py [checkpoint_path]

If no path given, evaluates the final warmup model.
"""

import os
import sys

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

local_rank = int(os.environ.get("LOCAL_RANK", 0))

model_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "runs/less/Llama-2-7b-hfp16_s42_lr2e-05/warmup"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": f"cuda:{local_rank}"},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(
    model=lm, tasks=["mmlu"], num_fewshot=5, batch_size=32
)

if local_rank == 0:
    overall = results["results"]["mmlu"]["acc,none"]
    print(f"\n{model_path}")
    print(f"MMLU 5-shot accuracy: {overall:.4f}")
