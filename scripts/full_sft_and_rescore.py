#!/usr/bin/env python3
"""
Full SFT (no LoRA) of OLMo-3-7B-Instruct on SmolLM2-135M-10B, then re-run
trackstar scoring against MMLU sociology to compare with the base model.

Training uses FSDP across 8 GPUs since the full 7B model doesn't fit on one
GPU with optimizer states at fp32.

Results are saved to runs/olmo3_7b_full_sft_mmlu_sociology_smollm2_10b/.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset


# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    base_model: str = "allenai/Olmo-3-7B-Instruct"
    dataset: str = "EleutherAI/SmolLM2-135M-10B"
    data_split: str = "train[:10000]"
    query_dataset: str = "cais/mmlu"
    query_subset: str = "sociology"
    query_split: str = "test"

    # SFT
    output_dir: str = "runs/olmo3_7b_full_sft_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 512

    # Trackstar scoring
    base_run: str = "runs/olmo3_7b_mmlu_sociology_smollm2_10b"
    sft_run: str = "runs/olmo3_7b_full_sft_mmlu_sociology_smollm2_10b"
    token_batch_size: int = 1024
    projection_dim: int = 16
    nproc_per_node: int = 8


def train_full_sft(cfg: ExperimentConfig):
    """Launch full SFT training via accelerate + FSDP as a subprocess."""
    print("=" * 60)
    print("Step 1: Full SFT on SmolLM2-135M-10B (10k docs)")
    print("=" * 60)

    output_path = Path(cfg.output_dir)
    if (output_path / "config.json").exists():
        print(f"  Full SFT model already exists at {output_path}, skipping training.")
        return

    # Run training as a subprocess via accelerate launch with FSDP
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "8",
        "--use_fsdp",
        "--fsdp_sharding_strategy",
        "FULL_SHARD",
        "--fsdp_auto_wrap_policy",
        "TRANSFORMER_BASED_WRAP",
        "--fsdp_backward_prefetch",
        "BACKWARD_PRE",
        "--fsdp_state_dict_type",
        "FULL_STATE_DICT",
        "--mixed_precision",
        "bf16",
        __file__,
        "--train-only",
        "--base_model",
        cfg.base_model,
        "--dataset",
        cfg.dataset,
        "--data_split",
        cfg.data_split,
        "--output_dir",
        cfg.output_dir,
        "--num_train_epochs",
        str(cfg.num_train_epochs),
        "--per_device_train_batch_size",
        str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(cfg.gradient_accumulation_steps),
        "--learning_rate",
        str(cfg.learning_rate),
        "--max_seq_length",
        str(cfg.max_seq_length),
    ]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_train_worker(args):
    """Full SFT training worker, called via accelerate launch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    )

    ds = load_dataset(args.dataset, split=args.data_split)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field="text",
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Model saved to {args.output_dir}")


def run_trackstar(cfg: ExperimentConfig):
    """Run trackstar scoring with the fine-tuned model."""
    print("=" * 60)
    print("Step 2: Trackstar scoring with full SFT model")
    print("=" * 60)

    scores_dir = Path(cfg.sft_run) / "scores"
    if scores_dir.exists():
        print(f"  Scores already exist at {scores_dir}, skipping.")
        return

    cmd = [
        "bergson",
        "trackstar",
        cfg.sft_run,
        "--model",
        cfg.output_dir,
        "--normalizer",
        "adafactor",
        "--stats_sample_size",
        "10000",
        "--index_cfg.precision",
        "fp32",
        "--fsdp",
        "--data.dataset",
        cfg.dataset,
        "--data.split",
        cfg.data_split,
        "--data.truncation",
        "--query.dataset",
        cfg.query_dataset,
        "--query.subset",
        cfg.query_subset,
        "--query.split",
        cfg.query_split,
        "--query.format_template",
        "bergson/templates/mcqa.yaml",
        "--query.truncation",
        "--unit_normalize",
        "--aggregation",
        "mean",
        "--normalize_aggregated_grad",
        "--projection_dim",
        str(cfg.projection_dim),
        "--token_batch_size",
        str(cfg.token_batch_size),
        "--nproc_per_node",
        str(cfg.nproc_per_node),
        "--resume",
    ]

    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def save_results(run_path: str, label: str):
    """Save top/bottom 20 texts and score summary."""
    print(f"  Saving results for {label}...")
    scores_dir = Path(run_path) / "scores"
    dtype = np.dtype(
        {
            "names": ["score_0", "written_0"],
            "formats": ["float32", "bool"],
            "offsets": [0, 4],
            "itemsize": 8,
        }
    )
    scores = np.memmap(str(scores_dir / "scores.bin"), dtype=dtype, mode="r")
    s = scores["score_0"]

    ds = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train[:10000]")

    results_dir = Path(run_path) / "results"
    results_dir.mkdir(exist_ok=True)

    # Save top 20
    top_idx = np.argsort(s)[-20:][::-1]
    with open(results_dir / "top_20.txt", "w") as f:
        for rank, idx in enumerate(top_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

    # Save bottom 20
    bot_idx = np.argsort(s)[:20]
    with open(results_dir / "bottom_20.txt", "w") as f:
        for rank, idx in enumerate(bot_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

    # Save summary stats
    with open(results_dir / "summary.txt", "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Run path: {run_path}\n")
        f.write(f"Total items: {len(s)}\n")
        f.write(f"Written: {int(scores['written_0'].sum())}\n")
        f.write(f"Score mean: {s.mean():.6f}\n")
        f.write(f"Score std: {s.std():.6f}\n")
        f.write(f"Score min: {s.min():.6f}\n")
        f.write(f"Score max: {s.max():.6f}\n\n")
        f.write("Top 20 indices and scores:\n")
        for rank, idx in enumerate(top_idx, 1):
            f.write(f"  #{rank}: idx={idx}, score={s[idx]:.6f}\n")
        f.write("\nBottom 20 indices and scores:\n")
        for rank, idx in enumerate(bot_idx, 1):
            f.write(f"  #{rank}: idx={idx}, score={s[idx]:.6f}\n")

    # Save raw scores
    np.save(results_dir / "scores.npy", np.array(s))

    print(f"  Results saved to {results_dir}")


def compare_results(base_run: str, sft_run: str):
    """Print a comparison of base vs SFT scores."""
    print("=" * 60)
    print("Step 4: Comparison")
    print("=" * 60)

    dtype = np.dtype(
        {
            "names": ["score_0", "written_0"],
            "formats": ["float32", "bool"],
            "offsets": [0, 4],
            "itemsize": 8,
        }
    )

    base_scores = np.memmap(f"{base_run}/scores/scores.bin", dtype=dtype, mode="r")[
        "score_0"
    ]
    sft_scores = np.memmap(f"{sft_run}/scores/scores.bin", dtype=dtype, mode="r")[
        "score_0"
    ]

    print(f"  {'':>30s} {'Base':>12s} {'SFT':>12s} {'Delta':>12s}")
    print(
        f"  {'Mean':>30s} {base_scores.mean():>12.6f} {sft_scores.mean():>12.6f} {sft_scores.mean()-base_scores.mean():>+12.6f}"
    )
    print(
        f"  {'Std':>30s} {base_scores.std():>12.6f} {sft_scores.std():>12.6f} {sft_scores.std()-base_scores.std():>+12.6f}"
    )
    print(
        f"  {'Min':>30s} {base_scores.min():>12.6f} {sft_scores.min():>12.6f} {sft_scores.min()-base_scores.min():>+12.6f}"
    )
    print(
        f"  {'Max':>30s} {base_scores.max():>12.6f} {sft_scores.max():>12.6f} {sft_scores.max()-base_scores.max():>+12.6f}"
    )

    # Rank correlation
    from scipy.stats import kendalltau, spearmanr

    rho, _ = spearmanr(base_scores, sft_scores)
    tau, _ = kendalltau(base_scores, sft_scores)
    print(f"  {'Spearman rho':>30s} {rho:>12.4f}")
    print(f"  {'Kendall tau':>30s} {tau:>12.4f}")

    # Top-20 overlap
    base_top = set(np.argsort(base_scores)[-20:])
    sft_top = set(np.argsort(sft_scores)[-20:])
    base_bot = set(np.argsort(base_scores)[:20])
    sft_bot = set(np.argsort(sft_scores)[:20])
    print(f"  {'Top-20 overlap':>30s} {len(base_top & sft_top):>12d}/20")
    print(f"  {'Bottom-20 overlap':>30s} {len(base_bot & sft_bot):>12d}/20")

    # Save comparison
    comp_dir = Path(sft_run) / "results"
    comp_dir.mkdir(exist_ok=True)
    with open(comp_dir / "comparison.txt", "w") as f:
        f.write(f"{'':>30s} {'Base':>12s} {'SFT':>12s} {'Delta':>12s}\n")
        f.write(
            f"{'Mean':>30s} {base_scores.mean():>12.6f} {sft_scores.mean():>12.6f} {sft_scores.mean()-base_scores.mean():>+12.6f}\n"
        )
        f.write(
            f"{'Std':>30s} {base_scores.std():>12.6f} {sft_scores.std():>12.6f} {sft_scores.std()-base_scores.std():>+12.6f}\n"
        )
        f.write(
            f"{'Min':>30s} {base_scores.min():>12.6f} {sft_scores.min():>12.6f} {sft_scores.min()-base_scores.min():>+12.6f}\n"
        )
        f.write(
            f"{'Max':>30s} {base_scores.max():>12.6f} {sft_scores.max():>12.6f} {sft_scores.max()-base_scores.max():>+12.6f}\n"
        )
        f.write(f"{'Spearman rho':>30s} {rho:>12.4f}\n")
        f.write(f"{'Kendall tau':>30s} {tau:>12.4f}\n")
        f.write(f"{'Top-20 overlap':>30s} {len(base_top & sft_top):>12d}/20\n")
        f.write(f"{'Bottom-20 overlap':>30s} {len(base_bot & sft_bot):>12d}/20\n")

    print(f"  Comparison saved to {comp_dir / 'comparison.txt'}")


def main():
    cfg = ExperimentConfig()

    # Step 1: Full SFT
    train_full_sft(cfg)

    # Step 2: Trackstar with full SFT model
    run_trackstar(cfg)

    # Step 3: Save results for both runs
    print("=" * 60)
    print("Step 3: Saving results")
    print("=" * 60)
    save_results(cfg.base_run, "base")
    save_results(cfg.sft_run, "full_sft")

    # Step 4: Compare
    compare_results(cfg.base_run, cfg.sft_run)

    print("\nDone!")


if __name__ == "__main__":
    if "--train-only" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train-only", action="store_true")
        parser.add_argument("--base_model", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--data_split", required=True)
        parser.add_argument("--output_dir", required=True)
        parser.add_argument("--num_train_epochs", type=int, required=True)
        parser.add_argument("--per_device_train_batch_size", type=int, required=True)
        parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        parser.add_argument("--max_seq_length", type=int, required=True)
        _run_train_worker(parser.parse_args())
    else:
        main()
