#!/usr/bin/env python3
"""Quick test: load a checkpoint and run the eval gradient accumulation
to verify it doesn't OOM."""

import time

import torch
import torchopt
from datasets import concatenate_datasets, load_dataset
from torchopt.pytree import tree_iter
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.trainer import BackwardState, Trainer, TrainerState
from bergson.utils.math import weighted_causal_lm_ce

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
EVAL_CHUNK_SIZE = 8
CKPT_DIR = "/projects/a6a/public/lucia/magic_checkpoints"
LR = 1e-4


def build_eval_batch(examples, tokenizer):
    letters = ["A", "B", "C", "D"]
    texts, answer_letters = [], []
    for ex in examples:
        q, choices, answer_idx = ex["question"], ex["choices"], ex["answer"]
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
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    labels = torch.full_like(input_ids, -100)

    for i, ans_str in enumerate(answer_letters):
        ans_tok = tokenizer.encode(ans_str, add_special_tokens=False)
        seq_len = attention_mask[i].sum().item()
        pos = int(seq_len)
        if pos < input_ids.shape[1]:
            input_ids[i, pos] = ans_tok[0]
            labels[i, pos] = ans_tok[0]
            attention_mask[i, pos] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(DEVICE)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Init trainer (moves model to meta)
    opt = torchopt.sgd(LR)
    trainer, _ = Trainer.initialize(model, opt)
    print(f"Trainer initialized. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # Load latest checkpoint as our "state after training"
    import os

    ckpts = sorted(
        os.listdir(CKPT_DIR), key=lambda f: int(f.split("_")[1].split(".")[0])
    )
    # Skip potentially corrupted last checkpoint
    latest = os.path.join(CKPT_DIR, ckpts[-2])
    print(f"Loading checkpoint: {latest}")
    state = TrainerState.load(latest)
    # Detach to simulate post-training state
    state = TrainerState(
        {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
        state.opt_state,
        state.buffers,
        state.batch_index,
    )
    print(f"State loaded. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # Load WMDP eval
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
    wmdp_examples = [wmdp[i] for i in range(len(wmdp))]
    print(f"WMDP: {len(wmdp)} questions")

    # === THE FIX: accumulate gradients per chunk ===
    print("\nRunning eval with gradient accumulation...")
    param_keys = list(state.params.keys())
    grad_accum = None
    total_loss_value = 0.0
    n_chunks = 0
    t0 = time.time()

    for chunk_start in range(0, len(wmdp_examples), EVAL_CHUNK_SIZE):
        chunk = wmdp_examples[chunk_start : chunk_start + EVAL_CHUNK_SIZE]
        eval_batch = build_eval_batch(chunk, tokenizer)
        eval_inputs = {k: v.to(DEVICE) for k, v in eval_batch.items()}

        chunk_loss = trainer.evaluate(state, eval_inputs)
        total_loss_value += chunk_loss.item()

        grads = torch.autograd.grad(chunk_loss, list(state.params.values()))
        if grad_accum is None:
            grad_accum = [g.detach().clone() for g in grads]
        else:
            for i, g in enumerate(grads):
                grad_accum[i] += g.detach()
        n_chunks += 1

        del chunk_loss, grads, eval_inputs

        if n_chunks % 20 == 0:
            print(
                f"  Chunk {n_chunks}/"
                f"{(len(wmdp_examples) + EVAL_CHUNK_SIZE - 1) // EVAL_CHUNK_SIZE}, "
                f"GPU: {torch.cuda.memory_allocated(0)/1e9:.1f} GB",
                flush=True,
            )

    for g in grad_accum:
        g.div_(n_chunks)
    param_grads = dict(zip(param_keys, grad_accum))

    eval_time = time.time() - t0
    avg_loss = total_loss_value / n_chunks
    print(f"\nEval done in {eval_time:.1f}s")
    print(f"Avg loss: {avg_loss:.4f}")
    print(f"GPU after eval: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    # Build BackwardState
    opt_grads = [
        torch.zeros_like(buf)
        for buf in tree_iter(state.opt_state)
        if isinstance(buf, torch.Tensor) and buf.is_floating_point()
    ]
    print(f"opt_grads count: {len(opt_grads)}")
    print(f"param_grads count: {len(param_grads)}")
    BackwardState(param_grads, opt_grads, torch.zeros(1000, device=DEVICE))
    print("BackwardState created successfully")
    print(f"GPU final: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    print("\nSUCCESS - eval fix works!")


if __name__ == "__main__":
    main()
