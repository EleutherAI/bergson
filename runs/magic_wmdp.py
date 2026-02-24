#!/usr/bin/env python3
"""MAGIC attribution: fine-tune on WikiText with checkpoints, then backprop
through training to attribute WMDP-bio-robust evaluation loss to individual
training sequences.

Uses simple_fsdp across 4 GPUs for memory efficiency. No CPU offloading.

Eval query: loss only on the correct answer letter token (everything else masked
with -100). Averaged over the full wmdp_bio_robust_mcqa dataset (868 questions).

Resume: skips completed phases automatically based on checkpoint/output existence.

Setup (run once before sbatch):
  source .venv/bin/activate
  bash runs/magic_wmdp_setup.sh   # patches torch for twice-differentiable DTensor

Also requires two edits in bergson/trainer.py:
  1. sorted_checkpoints: remove `if not os.path.isfile(path): continue`
     (FSDP checkpoints are directories, not files)
  2. trainer.backward: change `weight_grads = result[-1] + w_grads`
     to `weight_grads = result[-1] + w_grads if result[-1] is not None else w_grads`
"""

import gc
import json
import os
import shutil
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torchopt
from datasets import concatenate_datasets, load_dataset
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.distributed import launch_distributed_run, simple_fsdp
from bergson.trainer import (
    BackwardState,
    DataStream,
    Trainer,
    TrainerState,
    sorted_checkpoints,
)
from bergson.utils.math import weighted_causal_lm_ce

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"

# Training hyperparams
LR = 1e-4
BATCH_SIZE = 4
NUM_BATCHES = 250  # 1000 training examples
MAX_SEQ_LEN = 256

# Eval — process in small chunks to avoid OOM
EVAL_CHUNK_SIZE = 8

# Paths — use a separate dir for FSDP checkpoints
CKPT_DIR = "/projects/a6a/public/lucia/magic_fsdp_checkpoints"
OUTPUT_DIR = "/home/a6a/lucia.a6a/bergson3/runs/magic_wmdp_per_token_output"
# Eval grads are d(eval_loss)/d(params) — independent of weight shape,
# so reuse the cached grads from the per-example run.
EVAL_GRADS_PATH = "/home/a6a/lucia.a6a/bergson3/runs/magic_wmdp_output/eval_grads.pt"
SCORES_PATH = os.path.join(OUTPUT_DIR, "attribution_scores.pt")


def build_eval_batch(examples: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    """Build a batch where labels are -100 everywhere except the correct answer
    letter token, so loss only measures answer prediction."""
    letters = ["A", "B", "C", "D"]

    texts = []
    answer_letters = []
    for ex in examples:
        q = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer"]

        text = f"Question: {q}\n"
        for i, c in enumerate(choices):
            text += f"{letters[i]}) {c}\n"
        text += "Answer:"
        texts.append(text)
        answer_letters.append(f" {letters[answer_idx]}")

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Labels: -100 everywhere, then place the correct answer token after "Answer:"
    labels = torch.full_like(input_ids, -100)

    for i, ans_str in enumerate(answer_letters):
        ans_tok = tokenizer.encode(ans_str, add_special_tokens=False)
        seq_len = attention_mask[i].sum().item()
        pos = int(seq_len)
        if pos < input_ids.shape[1]:
            input_ids[i, pos] = ans_tok[0]
            labels[i, pos] = ans_tok[0]
            attention_mask[i, pos] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def training_complete() -> bool:
    """Check if training checkpoints exist for all NUM_BATCHES steps."""
    if not os.path.isdir(CKPT_DIR):
        return False
    ckpts = sorted_checkpoints(CKPT_DIR)
    return len(ckpts) >= NUM_BATCHES


def worker(global_rank, rank, world_size, wikitext, wmdp):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if global_rank == 0:
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(rank)}")
        mem_gb = torch.cuda.get_device_properties(rank).total_memory / 1e9
        print(f"GPU memory per device: {mem_gb:.1f} GB")

    # ── Init process group ──────────────────────────────────────────────
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(device),
            rank=rank,
            timeout=timedelta(hours=2),
            world_size=world_size,
        )

    # ── Load model with FSDP ──────────────────────────────────────────
    if global_rank == 0:
        print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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

    n_params = sum(p.numel() for p in model.parameters())
    if global_rank == 0:
        print(f"Model loaded in {time.time() - t0:.1f}s  ({n_params/1e9:.2f}B params per shard)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    # ── Create optimizer + trainer ──────────────────────────────────────
    opt = torchopt.sgd(LR)
    trainer, state0 = Trainer.initialize(model, opt)
    if global_rank == 0:
        print(f"Trainer initialized. GPU: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB")

    stream = DataStream(
        wikitext,
        tokenizer,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        device=device,
        input_key="text",
        per_token=True,
        max_seq_len=MAX_SEQ_LEN,
    )
    if global_rank == 0:
        n_examples = BATCH_SIZE * NUM_BATCHES
        print(f"DataStream: {NUM_BATCHES} batches x {BATCH_SIZE} = {n_examples} examples")

    # ── Step 1: Forward training with checkpoints ───────────────────────
    if training_complete():
        if global_rank == 0:
            ckpts = sorted_checkpoints(CKPT_DIR)
            print(f"\n{'='*60}")
            print(f"Step 1: SKIPPED (found {len(ckpts)} checkpoints in {CKPT_DIR})")
            print(f"{'='*60}")

        # Load final checkpoint
        ckpts = sorted_checkpoints(CKPT_DIR)
        _, last_path = ckpts[-1]
        if global_rank == 0:
            print(f"Loading final checkpoint: {last_path}")
        state = TrainerState.load(last_path)
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        del state0
    else:
        if global_rank == 0:
            print(f"\n{'='*60}")
            print("Step 1: Fine-tuning with checkpoints...")
            print(f"{'='*60}")

        if rank == 0 and os.path.exists(CKPT_DIR):
            shutil.rmtree(CKPT_DIR)
        if world_size > 1:
            dist.barrier()
        os.makedirs(CKPT_DIR, exist_ok=True)

        t0 = time.time()
        state = trainer.train(state0, stream, save_dir=CKPT_DIR)
        del state0
        train_time = time.time() - t0

        # Detach params to free the autograd graph from training
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        gc.collect()

        if global_rank == 0:
            print(f"Training done in {train_time:.1f}s")

    if global_rank == 0:
        print(f"GPU after step 1: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB")

    # ── Step 2: Evaluate on WMDP-bio-robust and accumulate gradients ────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(EVAL_GRADS_PATH):
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Step 2: SKIPPED (loading cached eval grads from {EVAL_GRADS_PATH})")
            print(f"{'='*60}")
        saved = torch.load(EVAL_GRADS_PATH, weights_only=True)
        param_grads = {k: v.to(device) for k, v in saved["param_grads"].items()}
        avg_loss_value = saved["avg_loss"]
        n_chunks = saved["n_chunks"]
        if global_rank == 0:
            print(f"WMDP-bio-robust avg loss: {avg_loss_value:.4f}")
        del state
    else:
        if global_rank == 0:
            print(f"\n{'='*60}")
            print("Step 2: Evaluating on WMDP-bio-robust (answer-only loss)...")
            print(f"{'='*60}")

        wmdp_examples = [wmdp[i] for i in range(len(wmdp))]
        param_keys = list(state.params.keys())
        grad_accum = None
        total_loss_value = 0.0
        n_chunks = 0

        for chunk_start in range(0, len(wmdp_examples), EVAL_CHUNK_SIZE):
            chunk = wmdp_examples[chunk_start : chunk_start + EVAL_CHUNK_SIZE]
            eval_batch = build_eval_batch(chunk, tokenizer)
            eval_inputs = {k: v.to(device) for k, v in eval_batch.items()}

            chunk_loss = trainer.evaluate(state, eval_inputs)
            total_loss_value += chunk_loss.item()

            # Immediately compute gradients and free the graph
            grads = torch.autograd.grad(chunk_loss, list(state.params.values()))
            if grad_accum is None:
                grad_accum = [g.detach().clone() for g in grads]
            else:
                for i, g in enumerate(grads):
                    grad_accum[i] += g.detach()
            n_chunks += 1

            del chunk_loss, grads, eval_inputs

            if n_chunks % 20 == 0 and global_rank == 0:
                print(
                    f"  Eval chunk {n_chunks}, "
                    f"GPU: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB",
                    flush=True,
                )

        avg_loss_value = total_loss_value / n_chunks
        for g in grad_accum:
            g.div_(n_chunks)
        param_grads = dict(zip(param_keys, grad_accum))

        # Save eval grads for resume (rank 0 only)
        if global_rank == 0:
            torch.save(
                {
                    "param_grads": {k: v.cpu() for k, v in param_grads.items()},
                    "avg_loss": avg_loss_value,
                    "n_chunks": n_chunks,
                },
                EVAL_GRADS_PATH,
            )
            print(f"Saved eval grads to {EVAL_GRADS_PATH}")
            print(
                f"WMDP-bio-robust avg loss ({len(wmdp)} questions, {n_chunks} chunks): "
                f"{avg_loss_value:.4f}"
            )
        del state

    if global_rank == 0:
        print(f"GPU after step 2: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB")

    # ── Step 3: Backward through training ───────────────────────────────
    if os.path.exists(SCORES_PATH):
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Step 3: SKIPPED (scores already at {SCORES_PATH})")
            print(f"{'='*60}")
        scores = torch.load(SCORES_PATH, weights_only=True)
    else:
        if global_rank == 0:
            print(f"\n{'='*60}")
            print("Step 3: Backprop through training (MAGIC attribution)...")
            print(f"{'='*60}")

        t0 = time.time()
        # Load last checkpoint to get opt_state shape for zeros
        last_ckpt_state = TrainerState.load(sorted_checkpoints(CKPT_DIR)[-1][1])
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(last_ckpt_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        del last_ckpt_state

        bwd_state = BackwardState(
            param_grads, opt_grads, torch.zeros_like(stream.weights)
        )
        del param_grads
        gc.collect()

        stream.requires_grad = True
        if global_rank == 0:
            print(f"GPU before backward: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB")
            torch.cuda.reset_peak_memory_stats(rank)

        bwd_state = trainer.backward(CKPT_DIR, stream, bwd_state)

        # All-reduce weight grads across ranks
        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads)

        bwd_time = time.time() - t0
        if global_rank == 0:
            peak_gb = torch.cuda.max_memory_allocated(rank) / 1e9
            print(f"Backward done in {bwd_time:.1f}s")
            print(f"Peak GPU memory during backward: {peak_gb:.1f} GB")

        scores = bwd_state.weight_grads.detach().cpu()
        if global_rank == 0:
            torch.save(scores, SCORES_PATH)
            print(f"Saved scores to {SCORES_PATH}")
        del bwd_state

    # ── Step 4: Collect and analyze scores (rank 0 only) ─────────────────
    if global_rank == 0:
        print(f"\n{'='*60}")
        print("Step 4: Collecting attribution scores...")
        print(f"{'='*60}")

        print(f"Per-token score tensor shape: {scores.shape}")
        print(f"Score range: [{scores.min().item():.6e}, {scores.max().item():.6e}]")
        print(f"Score mean:  {scores.mean().item():.6e}")
        print(f"Score std:   {scores.std().item():.6e}")

        # Per-example aggregate: sum over token positions
        if scores.ndim == 2:
            example_scores = scores.sum(dim=1)  # [n_examples]
        else:
            example_scores = scores  # 1D fallback

        print(f"\nPer-example scores (sum over tokens):")
        print(f"  Shape: {example_scores.shape}")
        print(
            f"  Range: [{example_scores.min().item():.6e}, "
            f"{example_scores.max().item():.6e}]"
        )
        print(
            f"  Negative: {(example_scores < 0).sum().item()}, "
            f"  Positive: {(example_scores > 0).sum().item()}, "
            f"  Zero: {(example_scores == 0).sum().item()}"
        )

        # ── Step 5: Save results ────────────────────────────────────────
        sorted_indices = example_scores.argsort()
        n_show = 50

        results_lowest = []
        print(f"\n{'='*60}")
        print(f"Top {n_show} sequences with LOWEST attribution:")
        print(f"{'='*60}")
        for rank_i, idx in enumerate(sorted_indices[:n_show]):
            idx_int = int(idx)
            text = wikitext[idx_int]["text"]
            results_lowest.append(
                {
                    "rank": rank_i,
                    "dataset_index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "text": text[:500],
                }
            )
            print(
                f"#{rank_i:3d}  idx={idx_int:4d}  "
                f"score={example_scores[idx_int]:.6e}"
            )
            print(f"      {text[:120]}...")
            print()

        results_highest = []
        print(f"\n{'='*60}")
        print(f"Top {n_show} sequences with HIGHEST attribution:")
        print(f"{'='*60}")
        for rank_i, idx in enumerate(
            reversed(sorted_indices[-n_show:])
        ):
            idx_int = int(idx)
            text = wikitext[idx_int]["text"]
            results_highest.append(
                {
                    "rank": rank_i,
                    "dataset_index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "text": text[:500],
                }
            )
            print(
                f"#{rank_i:3d}  idx={idx_int:4d}  "
                f"score={example_scores[idx_int]:.6e}"
            )
            print(f"      {text[:120]}...")
            print()

        # Per-token breakdowns for top/bottom 5 examples
        if scores.ndim == 2:
            n_detail = 5
            print(f"\n{'='*60}")
            print(f"Per-token breakdown: {n_detail} lowest examples")
            print(f"{'='*60}")
            for rank_i, idx in enumerate(
                sorted_indices[:n_detail]
            ):
                idx_int = int(idx)
                text = wikitext[idx_int]["text"]
                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                tok_scores = scores[idx_int, : len(tokens)]
                print(
                    f"\n#{rank_i} idx={idx_int}  "
                    f"sum={example_scores[idx_int]:.6e}"
                )
                for t, (tid, s) in enumerate(
                    zip(tokens, tok_scores.tolist())
                ):
                    tok_str = tokenizer.decode([tid])
                    print(f"  [{t:3d}] {s:+.4e}  {tok_str!r}")

            print(f"\n{'='*60}")
            print(f"Per-token breakdown: {n_detail} highest examples")
            print(f"{'='*60}")
            for rank_i, idx in enumerate(
                reversed(sorted_indices[-n_detail:])
            ):
                idx_int = int(idx)
                text = wikitext[idx_int]["text"]
                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                tok_scores = scores[idx_int, : len(tokens)]
                print(
                    f"\n#{rank_i} idx={idx_int}  "
                    f"sum={example_scores[idx_int]:.6e}"
                )
                for t, (tid, s) in enumerate(
                    zip(tokens, tok_scores.tolist())
                ):
                    tok_str = tokenizer.decode([tid])
                    print(f"  [{t:3d}] {s:+.4e}  {tok_str!r}")

        with open(
            os.path.join(OUTPUT_DIR, "lowest_attribution.json"), "w"
        ) as f:
            json.dump(results_lowest, f, indent=2)

        with open(
            os.path.join(OUTPUT_DIR, "highest_attribution.json"), "w"
        ) as f:
            json.dump(results_highest, f, indent=2)

        all_results = []
        for i in range(len(example_scores)):
            entry = {
                "index": i,
                "example_score": float(example_scores[i]),
                "text": wikitext[i]["text"][:500],
            }
            if scores.ndim == 2:
                entry["token_scores"] = scores[i].tolist()
            all_results.append(entry)
        with open(
            os.path.join(OUTPUT_DIR, "all_scores.json"), "w"
        ) as f:
            json.dump(all_results, f, indent=2)

        # Save full per-token tensor separately
        if scores.ndim == 2:
            tok_path = os.path.join(
                OUTPUT_DIR, "per_token_scores.pt"
            )
            torch.save(scores, tok_path)
            print(f"\nPer-token scores saved to {tok_path}")

        print(f"\nAll results saved to {OUTPUT_DIR}")

        print("Done.")


def main():
    # ── Load datasets before spawning workers ──────────────────────────
    print("Loading WikiText-103...")
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    wikitext = wikitext.filter(lambda x: len(x["text"].strip()) > 100)
    wikitext = wikitext.map(lambda x: {"length": len(x["text"])})
    wikitext = wikitext.sort("length")
    print(f"WikiText after filtering: {len(wikitext)} rows")

    n_examples = BATCH_SIZE * NUM_BATCHES
    start = len(wikitext) // 4
    wikitext = wikitext.select(range(start, start + n_examples))
    print(
        f"Selected {n_examples} training examples (indices {start}..{start + n_examples})"
    )
    print(f"Text lengths: {wikitext[0]['length']}..{wikitext[-1]['length']} chars")

    print("\nLoading WMDP-bio-robust...")
    configs = [
        "bioweapons_and_bioterrorism",
        "dual_use_virology",
        "enhanced_potential_pandemic_pathogens",
        "expanding_access_to_threat_vectors",
        "reverse_genetics_and_easy_editing",
        "viral_vector_research",
    ]
    wmdp_parts = []
    for c in configs:
        ds = load_dataset("EleutherAI/wmdp_bio_robust_mcqa", c, split="robust")
        wmdp_parts.append(ds)
    wmdp = concatenate_datasets(wmdp_parts)
    print(f"WMDP-bio-robust: {len(wmdp)} questions across {len(configs)} categories")

    launch_distributed_run("magic-wmdp", worker, [wikitext, wmdp])


if __name__ == "__main__":
    main()
