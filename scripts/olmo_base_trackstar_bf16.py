#!/usr/bin/env python3
"""
Run trackstar scoring with the base OLMo-3-7B-Instruct model (no SFT)
against MMLU sociology on SmolLM2-135M-10B.

Uses bf16, no FSDP, and times each trackstar step.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset


@dataclass
class ExperimentConfig:
    base_model: str = "allenai/Olmo-3-7B-Instruct"
    dataset: str = "EleutherAI/SmolLM2-135M-10B"
    data_split: str = "train[:10000]"
    query_dataset: str = "cais/mmlu"
    query_subset: str = "sociology"
    query_split: str = "test"

    # Trackstar scoring
    run_path: str = "runs/olmo3_7b_base_trackstar_bf16_timed"
    token_batch_size: int = 1024
    projection_dim: int = 16
    nproc_per_node: int = 4


def run_trackstar(exp_cfg: ExperimentConfig):
    """Run trackstar scoring with the base model, timing each step."""
    print("=" * 60)
    print("Trackstar scoring with base model (bf16, no FSDP)")
    print("=" * 60)

    run_path = Path(exp_cfg.run_path)

    base_cmd = [
        "bergson",
        "trackstar",
        str(run_path),
        "--model",
        exp_cfg.base_model,
        "--normalizer",
        "adafactor",
        "--stats_sample_size",
        "10000",
        "--index_cfg.precision",
        "bf16",
        # no --fsdp
        "--data.dataset",
        exp_cfg.dataset,
        "--data.split",
        exp_cfg.data_split,
        "--data.truncation",
        "--query.dataset",
        exp_cfg.query_dataset,
        "--query.subset",
        exp_cfg.query_subset,
        "--query.split",
        exp_cfg.query_split,
        "--query.format_template",
        "bergson/templates/mcqa.yaml",
        "--query.truncation",
        "--unit_normalize",
        "--aggregation",
        "mean",
        "--normalize_aggregated_grad",
        "--projection_dim",
        str(exp_cfg.projection_dim),
        "--token_batch_size",
        str(exp_cfg.token_batch_size),
        "--nproc_per_node",
        str(exp_cfg.nproc_per_node),
        "--resume",
    ]

    print(f"  {' '.join(base_cmd)}")

    step_times: dict[str, float] = {}
    current_step: str | None = None
    step_start: float | None = None

    proc = subprocess.Popen(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

        if line.startswith("Step ") and "/5:" in line:
            now = time.monotonic()
            if current_step is not None and step_start is not None:
                step_times[current_step] = now - step_start
            current_step = line.strip()
            step_start = now

    proc.wait()

    # Record the last step
    if current_step is not None and step_start is not None:
        step_times[current_step] = time.monotonic() - step_start

    # Print timing summary
    print("\n" + "=" * 60)
    print("Trackstar step timings")
    print("=" * 60)
    total = 0.0
    for step, duration in step_times.items():
        mins, secs = divmod(duration, 60)
        print(f"  {step:<55s}  {int(mins):3d}m {secs:05.2f}s")
        total += duration
    mins, secs = divmod(total, 60)
    print(f"  {'Total':<55s}  {int(mins):3d}m {secs:05.2f}s")

    # Save timings to file
    timings_path = run_path / "step_timings.txt"
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timings_path, "w") as f:
        for step, duration in step_times.items():
            mins, secs = divmod(duration, 60)
            f.write(f"{step:<55s}  {int(mins):3d}m {secs:05.2f}s\n")
        mins, secs = divmod(total, 60)
        f.write(f"{'Total':<55s}  {int(mins):3d}m {secs:05.2f}s\n")
    print(f"  Timings saved to {timings_path}")

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, base_cmd)


def save_results(exp_cfg: ExperimentConfig):
    """Save top/bottom 20 texts and score summary."""
    run_path = exp_cfg.run_path
    scores_dir = Path(run_path) / "scores"

    if not scores_dir.exists():
        print(f"  No scores found at {scores_dir}, skipping results.")
        return

    print("=" * 60)
    print("Saving results")
    print("=" * 60)

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

    ds = load_dataset(exp_cfg.dataset, split=exp_cfg.data_split)

    results_dir = Path(run_path) / "results"
    results_dir.mkdir(exist_ok=True)

    # Top 20
    top_idx = np.argsort(s)[-20:][::-1]
    with open(results_dir / "top_20.txt", "w") as f:
        for rank, idx in enumerate(top_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

    # Bottom 20
    bot_idx = np.argsort(s)[:20]
    with open(results_dir / "bottom_20.txt", "w") as f:
        for rank, idx in enumerate(bot_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

    # Summary
    with open(results_dir / "summary.txt", "w") as f:
        f.write(f"Model: {exp_cfg.base_model}\n")
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

    np.save(results_dir / "scores.npy", np.array(s))
    print(f"  Results saved to {results_dir}")


def main():
    exp_cfg = ExperimentConfig()
    run_trackstar(exp_cfg)
    save_results(exp_cfg)
    print("\nDone!")


if __name__ == "__main__":
    main()
