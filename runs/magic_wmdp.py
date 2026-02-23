#!/usr/bin/env python3
"""MAGIC attribution: fine-tune on WikiText with checkpoints, then backprop
through training to attribute WMDP-bio-robust evaluation loss to individual
training sequences.

Eval query: loss only on the correct answer letter token (everything else masked
with -100). Averaged over the full wmdp_bio_robust_mcqa dataset (868 questions).

Resume: skips completed phases automatically based on checkpoint/output existence.
"""

import gc
import json
import os
import shutil
import time

import torch
import torchopt
from datasets import concatenate_datasets, load_dataset
from torchopt.pytree import tree_iter
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.trainer import BackwardState, DataStream, Trainer, TrainerState, sorted_checkpoints
from bergson.utils.math import weighted_causal_lm_ce

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
DEVICE = "cuda:0"

# Training hyperparams
LR = 1e-4
BATCH_SIZE = 4
NUM_BATCHES = 250      # 1000 training examples
MAX_SEQ_LEN = 256

# Eval — process in small chunks to avoid OOM
EVAL_CHUNK_SIZE = 8

# Paths
CKPT_DIR = "/projects/a6a/public/lucia/magic_checkpoints"
OUTPUT_DIR = "/home/a6a/lucia.a6a/bergson3/runs/magic_wmdp_output"
EVAL_GRADS_PATH = os.path.join(OUTPUT_DIR, "eval_grads.pt")
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
        # The answer token goes right after the last real token (which is ":")
        pos = int(seq_len)  # position after "Answer:"
        if pos < input_ids.shape[1]:
            # Set the input to include the answer token and label it
            input_ids[i, pos] = ans_tok[0]
            labels[i, pos] = ans_tok[0]
            attention_mask[i, pos] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def backward_with_offload(
    trainer: Trainer,
    ckpt_dir: str,
    data: DataStream,
    bwd_state: BackwardState,
) -> BackwardState:
    """Like Trainer.backward but offloads bwd_state to CPU during the traced
    forward step to keep GPU memory stable."""
    ckpts = sorted_checkpoints(ckpt_dir)

    for step_idx, (_, path) in enumerate(reversed(ckpts)):
        if step_idx % 10 == 0:
            print(f"  Backward step {step_idx+1}/{len(ckpts)}, "
                  f"GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

        # Move bwd_state to CPU
        cpu_param_grads = {k: v.cpu() for k, v in bwd_state.param_grads.items()}
        cpu_opt_grads = [g.cpu() for g in bwd_state.opt_grads]
        cpu_weight_grads = bwd_state.weight_grads.cpu()
        del bwd_state

        # Load checkpoint and traced forward
        state_i = TrainerState.load(path)
        state_i.requires_grad = True
        data.requires_grad = True

        flat_i = state_i.differentiable_tensors()

        state_f = trainer.step(
            state_i,
            data[state_i.batch_index],
            trace=True,
        )

        flat_f = state_f.differentiable_tensors()

        # Move grads back to GPU for VJP
        p_keys = list(cpu_param_grads.keys())
        p_grads = [cpu_param_grads[k].to(DEVICE) for k in p_keys]
        o_grads = [g.to(DEVICE) for g in cpu_opt_grads]
        w_grads = cpu_weight_grads.to(DEVICE)
        del cpu_param_grads, cpu_opt_grads, cpu_weight_grads

        inps = flat_i + [data.weights]
        result = list(
            torch.autograd.grad(
                flat_f,
                inps,
                grad_outputs=p_grads + o_grads,
                allow_unused=True,
            )
        )
        del p_grads, o_grads, state_i, state_f, flat_i, flat_f, inps

        # Extract results to CPU
        n_params = len(p_keys)
        param_grads = {k: result[i].cpu() for i, k in enumerate(p_keys)}
        opt_grads = [r.cpu() if r is not None else torch.tensor(0.0)
                     for r in result[n_params:-1]]
        wg = result[-1]
        weight_grads = (wg + w_grads if wg is not None else w_grads).cpu()
        del result, w_grads, wg

        bwd_state = BackwardState(param_grads, opt_grads, weight_grads)

    # Move final result back to GPU
    bwd_state = BackwardState(
        {k: v.to(DEVICE) for k, v in bwd_state.param_grads.items()},
        [g.to(DEVICE) for g in bwd_state.opt_grads],
        bwd_state.weight_grads.to(DEVICE),
    )
    return bwd_state


def training_complete() -> bool:
    """Check if training checkpoints exist for all NUM_BATCHES steps."""
    if not os.path.isdir(CKPT_DIR):
        return False
    ckpts = sorted_checkpoints(CKPT_DIR)
    return len(ckpts) >= NUM_BATCHES


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem_gb:.1f} GB")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded in {time.time() - t0:.1f}s  ({n_params/1e9:.2f}B params)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    # ── Load WikiText training data ─────────────────────────────────────────
    print("\nLoading WikiText-103...")
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    wikitext = wikitext.filter(lambda x: len(x["text"].strip()) > 100)
    wikitext = wikitext.map(lambda x: {"length": len(x["text"])})
    wikitext = wikitext.sort("length")
    print(f"WikiText after filtering: {len(wikitext)} rows")

    n_examples = BATCH_SIZE * NUM_BATCHES
    start = len(wikitext) // 4
    wikitext = wikitext.select(range(start, start + n_examples))
    print(f"Selected {n_examples} training examples (indices {start}..{start + n_examples})")
    print(f"Text lengths: {wikitext[0]['length']}..{wikitext[-1]['length']} chars")

    # ── Load WMDP-bio-robust eval set ───────────────────────────────────────
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

    # ── Create optimizer + trainer ──────────────────────────────────────────
    print("\nInitializing SGD optimizer and trainer...")
    opt = torchopt.sgd(LR)
    trainer, state0 = Trainer.initialize(model, opt)
    print(f"Trainer initialized. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    stream = DataStream(
        wikitext,
        tokenizer,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        device=DEVICE,
        input_key="text",
    )
    print(f"DataStream: {NUM_BATCHES} batches x {BATCH_SIZE} = {n_examples} examples")

    # ── Step 1: Forward training with checkpoints ───────────────────────────
    if training_complete():
        print(f"\n{'='*60}")
        print("Step 1: SKIPPED (found {0} checkpoints in {1})".format(
            len(sorted_checkpoints(CKPT_DIR)), CKPT_DIR))
        print(f"{'='*60}")
        # Load final checkpoint as state
        ckpts = sorted_checkpoints(CKPT_DIR)
        _, last_path = ckpts[-1]
        print(f"Loading final checkpoint: {last_path}")
        state = TrainerState.load(last_path)
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state, state.buffers, state.batch_index,
        )
        del state0
    else:
        print(f"\n{'='*60}")
        print("Step 1: Fine-tuning with checkpoints...")
        print(f"{'='*60}")

        if os.path.exists(CKPT_DIR):
            shutil.rmtree(CKPT_DIR)
        os.makedirs(CKPT_DIR)

        t0 = time.time()
        state = trainer.train(state0, stream, save_dir=CKPT_DIR)
        del state0
        train_time = time.time() - t0

        # Detach params to free the autograd graph from training
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state, state.buffers, state.batch_index,
        )
        gc.collect()

        ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".ckpt")]
        ckpt_size = sum(
            os.path.getsize(os.path.join(CKPT_DIR, f)) for f in ckpt_files
        )
        print(f"Training done in {train_time:.1f}s")
        print(f"Saved {len(ckpt_files)} checkpoints ({ckpt_size / 1e9:.1f} GB total)")

    print(f"GPU after step 1: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # ── Step 2: Evaluate on WMDP-bio-robust and accumulate gradients ────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(EVAL_GRADS_PATH):
        print(f"\n{'='*60}")
        print(f"Step 2: SKIPPED (loading cached eval grads from {EVAL_GRADS_PATH})")
        print(f"{'='*60}")
        saved = torch.load(EVAL_GRADS_PATH, weights_only=True)
        param_grads = {k: v.to(DEVICE) for k, v in saved["param_grads"].items()}
        avg_loss_value = saved["avg_loss"]
        n_chunks = saved["n_chunks"]
        print(f"WMDP-bio-robust avg loss: {avg_loss_value:.4f}")
        del state  # free GPU for backward
    else:
        print(f"\n{'='*60}")
        print("Step 2: Evaluating on WMDP-bio-robust (answer-only loss)...")
        print(f"{'='*60}")

        wmdp_examples = [wmdp[i] for i in range(len(wmdp))]
        param_keys = list(state.params.keys())
        grad_accum = None
        total_loss_value = 0.0
        n_chunks = 0

        for chunk_start in range(0, len(wmdp_examples), EVAL_CHUNK_SIZE):
            chunk = wmdp_examples[chunk_start:chunk_start + EVAL_CHUNK_SIZE]
            eval_batch = build_eval_batch(chunk, tokenizer)
            eval_inputs = {k: v.to(DEVICE) for k, v in eval_batch.items()}

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

            if n_chunks % 20 == 0:
                print(f"  Eval chunk {n_chunks}, "
                      f"GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)

        avg_loss_value = total_loss_value / n_chunks
        for g in grad_accum:
            g.div_(n_chunks)
        param_grads = dict(zip(param_keys, grad_accum))

        # Save eval grads for resume
        torch.save({
            "param_grads": {k: v.cpu() for k, v in param_grads.items()},
            "avg_loss": avg_loss_value,
            "n_chunks": n_chunks,
        }, EVAL_GRADS_PATH)
        print(f"Saved eval grads to {EVAL_GRADS_PATH}")

        print(f"WMDP-bio-robust avg loss ({len(wmdp)} questions, {n_chunks} chunks): "
              f"{avg_loss_value:.4f}")
        del state

    print(f"GPU after step 2: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # ── Step 3: Backward through training ───────────────────────────────────
    if os.path.exists(SCORES_PATH):
        print(f"\n{'='*60}")
        print(f"Step 3: SKIPPED (scores already at {SCORES_PATH})")
        print(f"{'='*60}")
        scores = torch.load(SCORES_PATH, weights_only=True)
    else:
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
        bwd_state = BackwardState(param_grads, opt_grads, torch.zeros_like(stream.weights))
        del param_grads
        gc.collect()

        stream.requires_grad = True
        print(f"GPU before backward loop: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

        bwd_state = backward_with_offload(trainer, CKPT_DIR, stream, bwd_state)
        bwd_time = time.time() - t0
        print(f"Backward done in {bwd_time:.1f}s")

        scores = bwd_state.weight_grads.detach().cpu()
        torch.save(scores, SCORES_PATH)
        print(f"Saved scores to {SCORES_PATH}")
        del bwd_state

    # ── Step 4: Collect and analyze scores ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: Collecting attribution scores...")
    print(f"{'='*60}")

    print(f"Score tensor shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.6e}, {scores.max().item():.6e}]")
    print(f"Score mean:  {scores.mean().item():.6e}")
    print(f"Score std:   {scores.std().item():.6e}")
    print(f"Negative: {(scores < 0).sum().item()}, "
          f"Positive: {(scores > 0).sum().item()}, "
          f"Zero: {(scores == 0).sum().item()}")

    # ── Step 5: Save results ────────────────────────────────────────────────
    sorted_indices = scores.argsort()
    n_show = 50

    results_lowest = []
    print(f"\n{'='*60}")
    print(f"Top {n_show} sequences with LOWEST attribution scores:")
    print(f"{'='*60}")
    for rank_i, idx in enumerate(sorted_indices[:n_show]):
        idx_int = int(idx)
        text = wikitext[idx_int]["text"]
        results_lowest.append({
            "rank": rank_i,
            "dataset_index": idx_int,
            "score": float(scores[idx_int]),
            "text": text[:500],
        })
        print(f"#{rank_i:3d}  idx={idx_int:4d}  score={scores[idx_int]:.6e}")
        print(f"      {text[:120]}...")
        print()

    results_highest = []
    print(f"\n{'='*60}")
    print(f"Top {n_show} sequences with HIGHEST attribution scores:")
    print(f"{'='*60}")
    for rank_i, idx in enumerate(reversed(sorted_indices[-n_show:])):
        idx_int = int(idx)
        text = wikitext[idx_int]["text"]
        results_highest.append({
            "rank": rank_i,
            "dataset_index": idx_int,
            "score": float(scores[idx_int]),
            "text": text[:500],
        })
        print(f"#{rank_i:3d}  idx={idx_int:4d}  score={scores[idx_int]:.6e}")
        print(f"      {text[:120]}...")
        print()

    with open(os.path.join(OUTPUT_DIR, "lowest_attribution.json"), "w") as f:
        json.dump(results_lowest, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "highest_attribution.json"), "w") as f:
        json.dump(results_highest, f, indent=2)

    all_results = []
    for i in range(len(scores)):
        all_results.append({
            "index": i,
            "score": float(scores[i]),
            "text": wikitext[i]["text"][:500],
        })
    with open(os.path.join(OUTPUT_DIR, "all_scores.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")

    # Cleanup checkpoints
    print(f"\nCleaning up checkpoints at {CKPT_DIR}...")
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
