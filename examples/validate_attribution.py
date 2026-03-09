#!/usr/bin/env python3
"""Validate MAGIC per-token attribution by finetuning on attributed tokens.

Finetunes the base model on only the positively-attributed tokens (masking
everything else with -100) and measures how much WMDP eval accuracy changes
via lm-eval. If attribution is correct, accuracy should decrease proportionally
to the sum of positive scores.

Experiments:
  positive_only (default): train only on positive attribution tokens
  negative_only: train only on negative tokens (control)
  all_tokens: train on all tokens (control)
  random_mask: random subset matching the positive fraction (control)

Usage:
  python -u runs/validate_attribution.py --experiment positive_only
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
from datasets import load_dataset
from simple_parsing import ArgumentParser
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.distributed import launch_distributed_run, simple_fsdp
from bergson.utils.math import weighted_causal_lm_ce


@dataclass
class ValidateConfig:
    """Validate MAGIC per-token attribution via finetuning."""

    model: str = "EleutherAI/deep-ignorance-unfiltered"
    """Base model to finetune."""

    lr: float = 2e-4
    """Learning rate."""

    batch_size: int = 32
    """Training batch size (must be divisible by world size)."""

    num_steps: int = 250
    """Number of training steps."""

    max_seq_len: int = 256
    """Maximum sequence length (must match scores tensor dim 1)."""

    optimizer: Literal["sgd", "adamw"] = "adamw"
    """Optimizer to use."""

    lr_scheduler: Literal["constant", "cosine"] = "cosine"
    """Learning rate schedule."""

    warmup_ratio: float = 0.1
    """Fraction of steps for linear warmup."""

    scores_path: str = "runs/magic_wmdp_msl1024_output/per_token_scores.pt"
    """Path to WMDP per-token attribution scores [n_examples, max_seq_len]."""

    mmlu_scores_path: str = "runs/magic_mmlu_msl1024_output/per_token_scores.pt"
    """Path to MMLU per-token attribution scores (for selective experiment)."""

    experiment: Literal[
        "positive_only", "negative_only", "all_tokens", "random_mask", "selective",
        "compatible"
    ] = "selective"
    """Which tokens to train on. 'compatible' trains on tokens that hurt WMDP AND help MMLU."""

    model_save_dir: str = "/projects/a6a/public/lucia/validate_attr_model"
    """Where to save the finetuned model."""

    output_dir: str = "/home/a6a/lucia.a6a/bergson3/runs/validate_attribution_output"
    """Directory for results JSON."""

    eval_task: str = "wmdp_bio"
    """lm-eval task to evaluate on."""

    seed: int = 42
    """Random seed for reproducibility."""


def load_wikitext(n_examples: int):
    """Load WikiText-103 with same selection as magic_wmdp.py."""
    print("Loading WikiText-103...")
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    wikitext = wikitext.filter(lambda x: len(x["text"].strip()) > 100)
    wikitext = wikitext.map(lambda x: {"length": len(x["text"])})
    wikitext = wikitext.sort("length")
    print(f"WikiText after filtering: {len(wikitext)} rows")

    start = len(wikitext) // 4
    wikitext = wikitext.select(range(start, start + n_examples))
    print(f"Selected {n_examples} examples (indices {start}..{start + n_examples})")
    return wikitext


def prepare_batches(
    wikitext, scores, tokenizer, run_cfg: ValidateConfig,
    mmlu_scores: torch.Tensor | None = None,
):
    """Tokenize all examples and build masked labels based on experiment type."""
    n_examples = len(wikitext)
    texts = [wikitext[i]["text"] for i in range(n_examples)]

    print(f"Tokenizing {n_examples} examples (max_seq_len={run_cfg.max_seq_len})...")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=run_cfg.max_seq_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]  # [N, T]
    attention_mask = encoded["attention_mask"]  # [N, T]

    # Standard causal LM labels: input_ids where attention_mask=1, else -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Build token mask based on experiment type
    if run_cfg.experiment == "all_tokens":
        mask = attention_mask.bool()
    elif run_cfg.experiment == "positive_only":
        mask = scores > 0
    elif run_cfg.experiment == "negative_only":
        mask = scores < 0
    elif run_cfg.experiment == "selective":
        assert mmlu_scores is not None, "selective experiment requires --mmlu_scores_path"
        # Tokens that hurt WMDP (positive) but don't help MMLU (non-positive)
        mask = (scores > 0) & (mmlu_scores <= 0)
    elif run_cfg.experiment == "compatible":
        assert mmlu_scores is not None, "compatible experiment requires --mmlu_scores_path"
        # Tokens that hurt WMDP (positive score) AND help MMLU (negative score)
        mask = (scores > 0) & (mmlu_scores < 0)
    elif run_cfg.experiment == "random_mask":
        positive_frac = (scores > 0).float().mean().item()
        torch.manual_seed(run_cfg.seed)
        mask = torch.rand_like(scores, dtype=torch.float32) < positive_frac
        mask = mask & attention_mask.bool()
    else:
        raise ValueError(f"Unknown experiment: {run_cfg.experiment}")

    # Apply mask to labels
    labels[~mask] = -100

    n_train_tokens = (labels != -100).sum().item()
    n_total_tokens = attention_mask.sum().item()
    print(f"Experiment: {run_cfg.experiment}")
    frac = n_train_tokens / n_total_tokens
    print(f"Training tokens: {n_train_tokens} / {n_total_tokens} ({frac:.1%})")

    # Pre-split into batches
    batches = []
    for i in range(0, n_examples, run_cfg.batch_size):
        batch = {
            "input_ids": input_ids[i : i + run_cfg.batch_size],
            "labels": labels[i : i + run_cfg.batch_size],
            "attention_mask": attention_mask[i : i + run_cfg.batch_size],
        }
        batches.append(batch)

    return batches


def run_lm_eval(model_path: str, task: str, output_dir: str) -> dict:
    """Run lm-eval as a subprocess and parse results."""
    # Clean stale results from prior runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "/home/a6a/lucia.a6a/miniforge3/bin/torchrun",
        "--nproc_per_node=4",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},dtype=bfloat16",
        "--tasks",
        task,
        "--batch_size",
        "auto",
        "--verbosity",
        "WARNING",
        "--output_path",
        output_dir,
    ]
    print(f"\nlm_eval command:\n  {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"lm_eval stdout (last 3000):\n{result.stdout[-3000:]}")
        print(f"lm_eval stderr (last 3000):\n{result.stderr[-3000:]}")
        raise RuntimeError(f"lm_eval failed with return code {result.returncode}")

    # Parse results JSON — lm_eval saves as results_<timestamp>.json
    results = None
    for root, _dirs, files in os.walk(output_dir):
        for f in sorted(files, reverse=True):
            if f.startswith("results") and f.endswith(".json"):
                with open(os.path.join(root, f)) as fp:
                    results = json.load(fp)
                break
        if results is not None:
            break

    if results is None:
        raise FileNotFoundError(f"Could not find results*.json in {output_dir}")

    return results


def _clean_fsdp_key(key: str) -> str:
    """Strip simple_fsdp parametrization prefixes from state dict keys.

    e.g. 'layers.0.attention.dense.parametrizations.weight.original'
      -> 'layers.0.attention.dense.weight'
    """
    return key.replace(".parametrizations.", ".").replace(".original", "")


def save_model(model, model_name: str, save_dir: str, world_size: int):
    """Save model, handling DTensor params from simple_fsdp if needed."""
    if world_size > 1:
        # state_dict() triggers parametrizations (all-gather), returns DTensors
        state_dict = model.state_dict()
        # Convert DTensors to regular tensors and fix key names
        clean_state_dict = {
            _clean_fsdp_key(k): v.full_tensor() if isinstance(v, DTensor) else v
            for k, v in state_dict.items()
        }
        dist.barrier()

        if dist.get_rank() == 0:
            model.save_pretrained(save_dir, state_dict=clean_state_dict)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(save_dir)
            print(f"Model + tokenizer saved to {save_dir}")
    else:
        model.save_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        print(f"Model + tokenizer saved to {save_dir}")


def worker(global_rank, rank, world_size, batches, run_cfg: ValidateConfig):
    """Distributed worker: finetune and save model. No lm-eval here."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(run_cfg.seed)
    torch.cuda.manual_seed(run_cfg.seed)

    if global_rank == 0:
        print(f"\nWorld size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(rank)}")

    # ── Init process group ──────────────────────────────────────────────
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(device),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    # ── Load model ──────────────────────────────────────────────────────
    if global_rank == 0:
        print(f"\nLoading model: {run_cfg.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(device)

    if world_size > 1:
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            simple_fsdp(model)
        if global_rank == 0:
            print("Applied simple_fsdp")

    if global_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded in {time.time() - t0:.1f}s ({n_params / 1e9:.2f}B params)")

    # ── Finetune ────────────────────────────────────────────────────────
    n_batches = min(run_cfg.num_steps, len(batches))
    if global_rank == 0:
        print(f"\n{'=' * 60}")
        print(
            f"Finetuning ({run_cfg.experiment}, {n_batches} steps, "
            f"{run_cfg.optimizer} lr={run_cfg.lr}, {run_cfg.lr_scheduler})"
        )
        print(f"{'=' * 60}")

    if run_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=run_cfg.lr)

    # LR scheduler
    warmup_steps = int(n_batches * run_cfg.warmup_ratio)
    if run_cfg.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(n_batches - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    model.train()

    t0 = time.time()
    total_loss = 0.0

    for step_i in range(n_batches):
        batch = batches[step_i]

        # Shard across ranks
        if world_size > 1:
            batch = {k: v[rank::world_size].to(device) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if global_rank == 0 and (step_i + 1) % 10 == 0:
            avg = total_loss / (step_i + 1)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step_i + 1}/{n_batches}  "
                f"loss={loss.item():.4f}  avg={avg:.4f}  lr={cur_lr:.2e}"
            )

    train_time = time.time() - t0
    avg_loss = total_loss / n_batches
    if global_rank == 0:
        print(f"Training done in {train_time:.1f}s  avg_loss={avg_loss:.4f}")

    # ── Save finetuned model ────────────────────────────────────────────
    if global_rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Saving model to {run_cfg.model_save_dir}")
        print(f"{'=' * 60}")

    save_model(model, run_cfg.model, run_cfg.model_save_dir, world_size)

    # ── Save training stats for main process ────────────────────────────
    if global_rank == 0:
        stats = {
            "training_loss_avg": avg_loss,
            "training_time_s": train_time,
            "n_batches": n_batches,
        }
        stats_path = os.path.join(run_cfg.output_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Training stats saved to {stats_path}")

    if world_size > 1:
        dist.barrier()


def main():
    parser = ArgumentParser()
    parser.add_arguments(ValidateConfig, dest="run_cfg")
    run_cfg: ValidateConfig = parser.parse_args().run_cfg

    os.makedirs(run_cfg.output_dir, exist_ok=True)

    # ── Load data and scores ────────────────────────────────────────────
    scores = torch.load(run_cfg.scores_path, weights_only=True)
    n_examples = scores.shape[0]
    print(f"Loaded WMDP scores: {scores.shape} from {run_cfg.scores_path}")

    mmlu_scores = None
    if run_cfg.experiment in ("selective", "compatible"):
        mmlu_scores = torch.load(run_cfg.mmlu_scores_path, weights_only=True)
        print(f"Loaded MMLU scores: {mmlu_scores.shape} from {run_cfg.mmlu_scores_path}")
        # Truncate both to the same seq len
        seq_len = min(scores.shape[1], mmlu_scores.shape[1], run_cfg.max_seq_len)
        scores = scores[:, :seq_len]
        mmlu_scores = mmlu_scores[:, :seq_len]
        run_cfg.max_seq_len = seq_len
        print(f"Using seq_len={seq_len} (min of both score tensors and max_seq_len)")

    wikitext = load_wikitext(n_examples)

    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    batches = prepare_batches(wikitext, scores, tokenizer, run_cfg, mmlu_scores=mmlu_scores)
    print(f"Prepared {len(batches)} batches of size {run_cfg.batch_size}")

    # ── Step 1: Baseline lm-eval (main process, no model in GPU) ───────
    print(f"\n{'=' * 60}")
    print("Step 1: Baseline lm-eval")
    print(f"{'=' * 60}")

    baseline_dir = os.path.join(run_cfg.output_dir, "baseline_eval")
    baseline_results = run_lm_eval(run_cfg.model, run_cfg.eval_task, baseline_dir)
    baseline_acc = baseline_results["results"][run_cfg.eval_task]["acc,none"]
    print(f"Baseline {run_cfg.eval_task} accuracy: {baseline_acc:.4f}")

    # ── Step 2: Distributed finetune + save ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Launching distributed training")
    print(f"{'=' * 60}")

    launch_distributed_run("validate-attr", worker, [batches, run_cfg])

    # ── Step 3: Post-finetune lm-eval (main process, GPUs free again) ──
    print(f"\n{'=' * 60}")
    print("Step 3: Post-finetune lm-eval")
    print(f"{'=' * 60}")

    post_dir = os.path.join(run_cfg.output_dir, "post_eval")
    post_results = run_lm_eval(run_cfg.model_save_dir, run_cfg.eval_task, post_dir)
    post_acc = post_results["results"][run_cfg.eval_task]["acc,none"]
    print(f"Post-finetune {run_cfg.eval_task} accuracy: {post_acc:.4f}")
    print(f"Accuracy change: {post_acc - baseline_acc:+.4f}")

    # ── Step 4: Save results ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 4: Saving results")
    print(f"{'=' * 60}")

    # Load training stats saved by worker
    stats_path = os.path.join(run_cfg.output_dir, "training_stats.json")
    with open(stats_path) as f:
        train_stats = json.load(f)

    positive_mask = scores > 0
    negative_mask = scores < 0

    results = {
        "experiment": run_cfg.experiment,
        "baseline_wmdp_bio_acc": baseline_acc,
        "post_wmdp_bio_acc": post_acc,
        "accuracy_change": post_acc - baseline_acc,
        "predicted_delta_loss": float(scores[positive_mask].sum()),
        "n_positive_tokens": int(positive_mask.sum()),
        "n_negative_tokens": int(negative_mask.sum()),
        "n_total_tokens": int((scores != 0).sum()),
        "fraction_positive": float(positive_mask.float().mean()),
        "training_config": {
            "model": run_cfg.model,
            "lr": run_cfg.lr,
            "optimizer": run_cfg.optimizer,
            "lr_scheduler": run_cfg.lr_scheduler,
            "warmup_ratio": run_cfg.warmup_ratio,
            "batch_size": run_cfg.batch_size,
            "num_steps": train_stats["n_batches"],
            "max_seq_len": run_cfg.max_seq_len,
            "experiment": run_cfg.experiment,
            "seed": run_cfg.seed,
        },
        "training_loss_avg": train_stats["training_loss_avg"],
        "training_time_s": train_stats["training_time_s"],
    }

    results_path = os.path.join(run_cfg.output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
