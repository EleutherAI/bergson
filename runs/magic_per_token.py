#!/usr/bin/env python3
"""MAGIC per-token attribution: fine-tune on a configurable training
dataset, then backprop through training to attribute eval loss to
individual tokens in training sequences.

Supports WMDP-bio-robust and MMLU as eval queries. Training checkpoints
are shared across eval tasks (same training trajectory).

Usage:
    # Training only (creates checkpoints):
    python -u runs/magic_per_token.py --dataset wmdp_forget --num_examples 10000 --batch_size 40 --train_only

    # Eval+backward only (requires checkpoints):
    python -u runs/magic_per_token.py --dataset wmdp_forget --num_examples 10000 --batch_size 40 --eval_task wmdp --eval_only
    python -u runs/magic_per_token.py --dataset wmdp_forget --num_examples 10000 --batch_size 40 --eval_task mmlu --eval_only

    # Full pipeline (train + eval):
    python -u runs/magic_per_token.py --dataset wmdp_forget --eval_task wmdp
"""

import argparse
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

LR = 1e-4
MAX_SEQ_LEN = 1024
EVAL_CHUNK_SIZE = 4

BASE_CKPT = "/projects/a6a/public/lucia"
BASE_OUT = "/home/a6a/lucia.a6a/bergson3/runs"


def get_paths(dataset_name: str, num_examples: int, eval_task: str):
    tag = f"{dataset_name}_{num_examples // 1000}k" if num_examples != 1000 else dataset_name
    ckpt_dir = os.path.join(BASE_CKPT, f"magic_{tag}_msl1024_ckpts")
    output_dir = os.path.join(BASE_OUT, f"magic_{tag}_{eval_task}_msl1024_output")
    eval_grads = os.path.join(output_dir, "eval_grads.pt")
    scores = os.path.join(output_dir, "attribution_scores.pt")
    return ckpt_dir, output_dir, eval_grads, scores


def build_eval_batch(
    examples: list[dict], tokenizer
) -> dict[str, torch.Tensor]:
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


def worker(
    global_rank,
    rank,
    world_size,
    train_data,
    eval_data,
    run_args,
):
    dataset_name = run_args.dataset
    text_key = run_args.text_key
    eval_task = run_args.eval_task
    batch_size = run_args.batch_size
    num_batches = run_args.num_examples // run_args.batch_size
    train_only = run_args.train_only
    eval_only = run_args.eval_only

    CKPT_DIR, OUTPUT_DIR, EVAL_GRADS_PATH, SCORES_PATH = get_paths(
        dataset_name, run_args.num_examples, eval_task
    )

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if global_rank == 0:
        print(f"Dataset: {dataset_name}")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(rank)}")

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

    n_params = sum(p.numel() for p in model.parameters())
    if global_rank == 0:
        print(
            f"Model loaded in {time.time() - t0:.1f}s  "
            f"({n_params/1e9:.2f}B params per shard)"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    opt = torchopt.sgd(LR)
    trainer, state0 = Trainer.initialize(model, opt)

    stream = DataStream(
        train_data,
        tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        device=device,
        input_key=text_key,
        per_token=True,
        max_seq_len=MAX_SEQ_LEN,
    )
    if global_rank == 0:
        n_ex = batch_size * num_batches
        print(f"DataStream: {num_batches} batches x {batch_size} = {n_ex}")

    # ── Step 1: Forward training ─────────────────────────────────────
    def training_complete():
        if not os.path.isdir(CKPT_DIR):
            return False
        return len(sorted_checkpoints(CKPT_DIR)) >= num_batches

    if eval_only:
        assert training_complete(), (
            f"--eval_only but checkpoints not found/incomplete in {CKPT_DIR}"
        )

    if training_complete():
        if global_rank == 0:
            ckpts = sorted_checkpoints(CKPT_DIR)
            print(f"\nStep 1: SKIPPED ({len(ckpts)} checkpoints)")
        ckpts = sorted_checkpoints(CKPT_DIR)
        _, last_path = ckpts[-1]
        state = TrainerState.load(last_path)
        state = TrainerState(
            {
                k: v.detach().requires_grad_(True)
                for k, v in state.params.items()
            },
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        del state0
    else:
        if global_rank == 0:
            print(f"\nStep 1: Fine-tuning with checkpoints...")
        if rank == 0 and os.path.exists(CKPT_DIR):
            shutil.rmtree(CKPT_DIR)
        if world_size > 1:
            dist.barrier()
        os.makedirs(CKPT_DIR, exist_ok=True)

        t0 = time.time()
        state = trainer.train(state0, stream, save_dir=CKPT_DIR)
        del state0
        if global_rank == 0:
            print(f"Training done in {time.time() - t0:.1f}s")

        state = TrainerState(
            {
                k: v.detach().requires_grad_(True)
                for k, v in state.params.items()
            },
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        gc.collect()

    if train_only:
        if global_rank == 0:
            print("Training complete. --train_only set, exiting.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ── Step 2: Evaluate on eval task ────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(EVAL_GRADS_PATH):
        if global_rank == 0:
            print(f"\nStep 2: SKIPPED (cached eval grads)")
        saved = torch.load(EVAL_GRADS_PATH, weights_only=True)
        param_grads = {
            k: v.to(device) for k, v in saved["param_grads"].items()
        }
        if global_rank == 0:
            print(f"{eval_task} avg loss: {saved['avg_loss']:.4f}")
        del state
    else:
        if global_rank == 0:
            print(f"\nStep 2: Evaluating on {eval_task}...")

        eval_examples = [eval_data[i] for i in range(len(eval_data))]
        param_keys = list(state.params.keys())
        grad_accum = None
        total_loss = 0.0
        n_chunks = 0

        for cs in range(0, len(eval_examples), EVAL_CHUNK_SIZE):
            chunk = eval_examples[cs : cs + EVAL_CHUNK_SIZE]
            eval_batch = build_eval_batch(chunk, tokenizer)
            eval_inputs = {
                k: v.to(device) for k, v in eval_batch.items()
            }
            chunk_loss = trainer.evaluate(state, eval_inputs)
            total_loss += chunk_loss.item()

            grads = torch.autograd.grad(
                chunk_loss, list(state.params.values())
            )
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

        avg_loss = total_loss / n_chunks
        for g in grad_accum:
            g.div_(n_chunks)
        param_grads = dict(zip(param_keys, grad_accum))

        if global_rank == 0:
            torch.save(
                {
                    "param_grads": {
                        k: v.cpu() for k, v in param_grads.items()
                    },
                    "avg_loss": avg_loss,
                    "n_chunks": n_chunks,
                },
                EVAL_GRADS_PATH,
            )
            print(f"{eval_task} avg loss: {avg_loss:.4f}")
        del state

    # ── Step 3: Backward through training ────────────────────────────
    if os.path.exists(SCORES_PATH):
        if global_rank == 0:
            print(f"\nStep 3: SKIPPED (scores cached)")
        scores = torch.load(SCORES_PATH, weights_only=True)
    else:
        if global_rank == 0:
            print(f"\nStep 3: Backprop through training...")

        t0 = time.time()
        last_ckpt = TrainerState.load(
            sorted_checkpoints(CKPT_DIR)[-1][1]
        )
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(last_ckpt.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        del last_ckpt

        bwd_state = BackwardState(
            param_grads, opt_grads, torch.zeros_like(stream.weights)
        )
        del param_grads
        gc.collect()

        stream.requires_grad = True
        if global_rank == 0:
            torch.cuda.reset_peak_memory_stats(rank)

        bwd_state = trainer.backward(CKPT_DIR, stream, bwd_state)

        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads)

        if global_rank == 0:
            peak = torch.cuda.max_memory_allocated(rank) / 1e9
            print(f"Backward done in {time.time() - t0:.1f}s")
            print(f"Peak GPU: {peak:.1f} GB")

        scores = bwd_state.weight_grads.detach().cpu()
        if global_rank == 0:
            torch.save(scores, SCORES_PATH)
        del bwd_state

    # ── Step 4: Analysis ─────────────────────────────────────────────
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"Results for {dataset_name}")
        print(f"{'='*60}")

        print(f"Per-token scores: {scores.shape}")
        print(
            f"Range: [{scores.min().item():.6e}, "
            f"{scores.max().item():.6e}]"
        )

        if scores.ndim == 2:
            example_scores = scores.sum(dim=1)
        else:
            example_scores = scores

        print(f"\nPer-example scores (sum over tokens):")
        print(f"  Shape: {example_scores.shape}")
        print(
            f"  Range: [{example_scores.min().item():.6e}, "
            f"{example_scores.max().item():.6e}]"
        )

        # Unlearning potential: sum of positive token attributions
        # per example (tokens that increase WMDP eval loss)
        if scores.ndim == 2:
            pos_mask = scores > 0
            unlearn_potential = (scores * pos_mask).sum(dim=1)
        else:
            unlearn_potential = scores.clamp(min=0)

        print(f"\nUnlearning potential (sum of positive token scores):")
        print(f"  Total: {unlearn_potential.sum().item():.6e}")
        print(f"  Mean per example: {unlearn_potential.mean().item():.6e}")
        print(f"  Max per example: {unlearn_potential.max().item():.6e}")
        print(
            f"  Examples with any positive tokens: "
            f"{(unlearn_potential > 0).sum().item()}/{len(unlearn_potential)}"
        )

        # Save summary
        sorted_idx = example_scores.argsort()
        n_show = 20
        results = {"dataset": dataset_name, "n_examples": len(scores)}
        results["unlearning_potential"] = {
            "total": float(unlearn_potential.sum()),
            "mean": float(unlearn_potential.mean()),
            "max": float(unlearn_potential.max()),
            "n_positive": int((unlearn_potential > 0).sum()),
        }

        results["lowest"] = []
        for rank_i, idx in enumerate(sorted_idx[:n_show]):
            idx_int = int(idx)
            text = train_data[idx_int][text_key]
            results["lowest"].append(
                {
                    "rank": rank_i,
                    "index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "unlearn_potential": float(
                        unlearn_potential[idx_int]
                    ),
                    "text": text[:500],
                }
            )

        results["highest"] = []
        for rank_i, idx in enumerate(
            reversed(sorted_idx[-n_show:])
        ):
            idx_int = int(idx)
            text = train_data[idx_int][text_key]
            results["highest"].append(
                {
                    "rank": rank_i,
                    "index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "unlearn_potential": float(
                        unlearn_potential[idx_int]
                    ),
                    "text": text[:500],
                }
            )

        with open(
            os.path.join(OUTPUT_DIR, "results.json"), "w"
        ) as f:
            json.dump(results, f, indent=2)

        if scores.ndim == 2:
            torch.save(
                scores,
                os.path.join(OUTPUT_DIR, "per_token_scores.pt"),
            )
            torch.save(
                unlearn_potential,
                os.path.join(OUTPUT_DIR, "unlearn_potential.pt"),
            )

        print(f"\nAll results saved to {OUTPUT_DIR}")
        print("Done.")


def load_wmdp_eval():
    configs = [
        "bioweapons_and_bioterrorism",
        "dual_use_virology",
        "enhanced_potential_pandemic_pathogens",
        "expanding_access_to_threat_vectors",
        "reverse_genetics_and_easy_editing",
        "viral_vector_research",
    ]
    parts = []
    for c in configs:
        ds = load_dataset(
            "EleutherAI/wmdp_bio_robust_mcqa", c, split="robust"
        )
        parts.append(ds)
    return concatenate_datasets(parts)


def load_mmlu_eval():
    ds = load_dataset("cais/mmlu", "all", split="test")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "wmdp_forget",
            "wmdp_retain",
            "ultrachat",
            "wmdp_lie_o",
        ],
    )
    parser.add_argument(
        "--eval_task",
        default="wmdp",
        choices=["wmdp", "mmlu"],
        help="Eval query for attribution (default: wmdp = WMDP-bio-robust)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of training examples (default: 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only run training phase (create checkpoints), skip eval/backward",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run eval/backward phases (checkpoints must exist)",
    )
    args = parser.parse_args()
    assert args.num_examples % args.batch_size == 0, (
        f"num_examples ({args.num_examples}) must be divisible by "
        f"batch_size ({args.batch_size})"
    )
    num_batches = args.num_examples // args.batch_size

    dataset_name = args.dataset
    n_examples = args.num_examples

    if dataset_name == "wmdp_forget":
        print("Loading WMDP bio forget corpus...")
        ds = load_dataset(
            "cais/wmdp-bio-forget-corpus", split="train"
        )
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        print(f"After filtering: {len(ds)} rows")
        ds = ds.select(range(min(n_examples, len(ds))))
        text_key = "text"

    elif dataset_name == "wmdp_retain":
        print("Loading WMDP bio retain corpus...")
        ds = load_dataset(
            "cais/wmdp-corpora", "bio-retain-corpus", split="train"
        )
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        print(f"After filtering: {len(ds)} rows")
        ds = ds.select(range(min(n_examples, len(ds))))
        text_key = "text"

    elif dataset_name == "wmdp_lie_o":
        print("Loading WMDP lie-o-deep-fried...")
        ds = load_dataset(
            "Unlearning/wmdp-lie-o-deep-fried", split="train"
        )
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        print(f"After filtering: {len(ds)} rows")
        ds = ds.select(range(min(n_examples, len(ds))))
        text_key = "text"

    elif dataset_name == "ultrachat":
        print("Loading UltraChat 200k...")
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft"
        )
        ds = ds.map(
            lambda x: {
                "text": "\n".join(
                    m["content"] for m in x["messages"]
                )
            }
        )
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        print(f"After filtering: {len(ds)} rows")
        ds = ds.select(range(min(n_examples, len(ds))))
        text_key = "text"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Selected {len(ds)} training examples")
    print(f"Batch size: {args.batch_size}, Num batches: {num_batches}")
    print(f"Sample: {ds[0][text_key][:200]}")

    # Load eval data (skip if train_only)
    eval_data = None
    if not args.train_only:
        if args.eval_task == "wmdp":
            print("\nLoading WMDP-bio-robust eval set...")
            eval_data = load_wmdp_eval()
            print(f"WMDP-bio-robust: {len(eval_data)} questions")
        elif args.eval_task == "mmlu":
            print("\nLoading MMLU eval set...")
            eval_data = load_mmlu_eval()
            print(f"MMLU: {len(eval_data)} questions")

    # Store args for worker access
    args.text_key = text_key

    launch_distributed_run(
        f"magic-{dataset_name}-{args.eval_task}",
        worker,
        [ds, eval_data, args],
    )


if __name__ == "__main__":
    main()
