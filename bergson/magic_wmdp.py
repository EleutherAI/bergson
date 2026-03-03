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
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
import torchopt
from datasets import concatenate_datasets, load_dataset
from simple_parsing import ArgumentParser, field
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


@dataclass
class MagicConfig:
    """MAGIC per-token attribution on WikiText -> eval task."""

    model: str = "EleutherAI/deep-ignorance-unfiltered"
    """Model to fine-tune and attribute."""

    lr: float = 1e-4
    """Learning rate for SGD fine-tuning."""

    batch_size: int = 4
    """Training batch size (must be divisible by world size)."""

    num_batches: int = 250
    """Number of training batches (total examples = batch_size * num_batches)."""

    max_seq_len: int = 1024
    """Maximum sequence length for tokenization."""

    eval_chunk_size: int = 4
    """Eval batch size (smaller to avoid OOM)."""

    per_token: bool = True
    """Use per-token attribution weights instead of per-example."""

    eval_task: Literal["wmdp", "mmlu"] = "wmdp"
    """Eval task for attribution: wmdp (WMDP-bio-robust) or mmlu."""

    ckpt_dir: str = "/projects/a6a/public/lucia/magic_wikitext_msl1024_ckpts"
    """Directory for FSDP training checkpoints."""

    output_dir: str = field(default="runs/magic_wikitext_msl1024_output")
    """Directory for output files (scores, eval grads, results)."""

    wandb_project: str = "magic-wmdp"
    """Wandb project name. Set to empty string to disable."""

    @property
    def n_examples(self) -> int:
        return self.batch_size * self.num_batches

    @property
    def eval_grads_path(self) -> str:
        return os.path.join(self.output_dir, "eval_grads.pt")

    @property
    def scores_path(self) -> str:
        return os.path.join(self.output_dir, "attribution_scores.pt")


def build_eval_batch(
    examples: list[dict], tokenizer, max_seq_len: int
) -> dict[str, torch.Tensor]:
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
        max_length=max_seq_len,
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


def worker(global_rank, rank, world_size, wikitext, wmdp, run_cfg: MagicConfig):
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

    n_params = sum(p.numel() for p in model.parameters())
    if global_rank == 0:
        print(
            f"Model loaded in {time.time() - t0:.1f}s  ({n_params/1e9:.2f}B params per shard)"
        )

    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = run_cfg.max_seq_len

    # ── Create optimizer + trainer ──────────────────────────────────────
    opt = torchopt.sgd(run_cfg.lr)
    trainer, state0 = Trainer.initialize(model, opt)
    if global_rank == 0:
        print(
            f"Trainer initialized. GPU: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB"
        )

    stream = DataStream(
        wikitext,
        tokenizer,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.num_batches,
        device=device,
        input_key="text",
        per_token=run_cfg.per_token,
        max_seq_len=run_cfg.max_seq_len,
    )
    if global_rank == 0:
        print(
            f"DataStream: {run_cfg.num_batches} batches x {run_cfg.batch_size} "
            f"= {run_cfg.n_examples} examples"
        )

    # ── Optional wandb init ──────────────────────────────────────────────
    log_fn = None
    if run_cfg.wandb_project and global_rank == 0:
        from bergson.utils.logging import wandb_log_fn

        log_fn = wandb_log_fn(
            run_cfg.wandb_project,
            config={
                "model": run_cfg.model,
                "lr": run_cfg.lr,
                "batch_size": run_cfg.batch_size,
                "num_batches": run_cfg.num_batches,
                "max_seq_len": run_cfg.max_seq_len,
                "eval_task": run_cfg.eval_task,
                "per_token": run_cfg.per_token,
            },
        )

    # ── Step 1: Forward training with checkpoints ───────────────────────
    def training_complete():
        if not os.path.isdir(run_cfg.ckpt_dir):
            return False
        return len(sorted_checkpoints(run_cfg.ckpt_dir)) >= run_cfg.num_batches

    if training_complete():
        if global_rank == 0:
            ckpts = sorted_checkpoints(run_cfg.ckpt_dir)
            print(f"\n{'='*60}")
            print(
                f"Step 1: SKIPPED (found {len(ckpts)} checkpoints in {run_cfg.ckpt_dir})"
            )
            print(f"{'='*60}")

        ckpts = sorted_checkpoints(run_cfg.ckpt_dir)
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

        if rank == 0 and os.path.exists(run_cfg.ckpt_dir):
            shutil.rmtree(run_cfg.ckpt_dir)
        if world_size > 1:
            dist.barrier()
        os.makedirs(run_cfg.ckpt_dir, exist_ok=True)

        t0 = time.time()
        state = trainer.train(state0, stream, save_dir=run_cfg.ckpt_dir, log_fn=log_fn)
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
    os.makedirs(run_cfg.output_dir, exist_ok=True)

    if os.path.exists(run_cfg.eval_grads_path):
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(
                f"Step 2: SKIPPED (loading cached eval grads from {run_cfg.eval_grads_path})"
            )
            print(f"{'='*60}")
        saved = torch.load(run_cfg.eval_grads_path, weights_only=True)
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

        for chunk_start in range(0, len(wmdp_examples), run_cfg.eval_chunk_size):
            chunk = wmdp_examples[chunk_start : chunk_start + run_cfg.eval_chunk_size]
            eval_batch = build_eval_batch(chunk, tokenizer, run_cfg.max_seq_len)
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
                run_cfg.eval_grads_path,
            )
            print(f"Saved eval grads to {run_cfg.eval_grads_path}")
            print(
                f"WMDP-bio-robust avg loss ({len(wmdp)} questions, {n_chunks} chunks): "
                f"{avg_loss_value:.4f}"
            )
        del state

    if global_rank == 0:
        print(f"GPU after step 2: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB")

    # ── Step 3: Backward through training ───────────────────────────────
    if os.path.exists(run_cfg.scores_path):
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Step 3: SKIPPED (scores already at {run_cfg.scores_path})")
            print(f"{'='*60}")
        scores = torch.load(run_cfg.scores_path, weights_only=True)
    else:
        if global_rank == 0:
            print(f"\n{'='*60}")
            print("Step 3: Backprop through training (MAGIC attribution)...")
            print(f"{'='*60}")

        t0 = time.time()
        # Load last checkpoint to get opt_state shape for zeros
        last_ckpt_state = TrainerState.load(sorted_checkpoints(run_cfg.ckpt_dir)[-1][1])
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
            print(
                f"GPU before backward: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB"
            )
            torch.cuda.reset_peak_memory_stats(rank)

        bwd_state = trainer.backward(run_cfg.ckpt_dir, stream, bwd_state)

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
            torch.save(scores, run_cfg.scores_path)
            print(f"Saved scores to {run_cfg.scores_path}")
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

        print("\nPer-example scores (sum over tokens):")
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
        for rank_i, idx in enumerate(reversed(sorted_indices[-n_show:])):
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
            for rank_i, idx in enumerate(sorted_indices[:n_detail]):
                idx_int = int(idx)
                text = wikitext[idx_int]["text"]
                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=run_cfg.max_seq_len,
                )
                tok_scores = scores[idx_int, : len(tokens)]
                print(
                    f"\n#{rank_i} idx={idx_int}  " f"sum={example_scores[idx_int]:.6e}"
                )
                for t, (tid, s) in enumerate(zip(tokens, tok_scores.tolist())):
                    tok_str = tokenizer.decode([tid])
                    print(f"  [{t:3d}] {s:+.4e}  {tok_str!r}")

            print(f"\n{'='*60}")
            print(f"Per-token breakdown: {n_detail} highest examples")
            print(f"{'='*60}")
            for rank_i, idx in enumerate(reversed(sorted_indices[-n_detail:])):
                idx_int = int(idx)
                text = wikitext[idx_int]["text"]
                tokens = tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=run_cfg.max_seq_len,
                )
                tok_scores = scores[idx_int, : len(tokens)]
                print(
                    f"\n#{rank_i} idx={idx_int}  " f"sum={example_scores[idx_int]:.6e}"
                )
                for t, (tid, s) in enumerate(zip(tokens, tok_scores.tolist())):
                    tok_str = tokenizer.decode([tid])
                    print(f"  [{t:3d}] {s:+.4e}  {tok_str!r}")

        with open(
            os.path.join(run_cfg.output_dir, "lowest_attribution.json"), "w"
        ) as f:
            json.dump(results_lowest, f, indent=2)

        with open(
            os.path.join(run_cfg.output_dir, "highest_attribution.json"), "w"
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
        with open(os.path.join(run_cfg.output_dir, "all_scores.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Save full per-token tensor separately
        if scores.ndim == 2:
            tok_path = os.path.join(run_cfg.output_dir, "per_token_scores.pt")
            torch.save(scores, tok_path)
            print(f"\nPer-token scores saved to {tok_path}")

        print(f"\nAll results saved to {run_cfg.output_dir}")

        print("Done.")


def main():
    parser = ArgumentParser()
    parser.add_arguments(MagicConfig, dest="run_cfg")
    run_cfg: MagicConfig = parser.parse_args().run_cfg

    # ── Load datasets before spawning workers ──────────────────────────
    print("Loading WikiText-103...")
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    wikitext = wikitext.filter(lambda x: len(x["text"].strip()) > 100)
    wikitext = wikitext.map(lambda x: {"length": len(x["text"])})
    wikitext = wikitext.sort("length")
    print(f"WikiText after filtering: {len(wikitext)} rows")

    start = len(wikitext) // 4
    wikitext = wikitext.select(range(start, start + run_cfg.n_examples))
    print(
        f"Selected {run_cfg.n_examples} training examples "
        f"(indices {start}..{start + run_cfg.n_examples})"
    )
    print(f"Text lengths: {wikitext[0]['length']}..{wikitext[-1]['length']} chars")

    if run_cfg.eval_task == "wmdp":
        print("\nLoading WMDP-bio-robust...")
        configs = [
            "bioweapons_and_bioterrorism",
            "dual_use_virology",
            "enhanced_potential_pandemic_pathogens",
            "expanding_access_to_threat_vectors",
            "reverse_genetics_and_easy_editing",
            "viral_vector_research",
        ]
        eval_parts = []
        for c in configs:
            ds = load_dataset("EleutherAI/wmdp_bio_robust_mcqa", c, split="robust")
            eval_parts.append(ds)
        eval_data = concatenate_datasets(eval_parts)
        print(
            f"WMDP-bio-robust: {len(eval_data)} questions across {len(configs)} categories"
        )
    elif run_cfg.eval_task == "mmlu":
        print("\nLoading MMLU...")
        eval_data = load_dataset("cais/mmlu", "all", split="test")
        print(f"MMLU: {len(eval_data)} questions")
    else:
        raise ValueError(f"Unknown eval_task: {run_cfg.eval_task}")

    job_name = f"magic-{run_cfg.eval_task}"
    launch_distributed_run(job_name, worker, [wikitext, eval_data, run_cfg])


if __name__ == "__main__":
    main()
