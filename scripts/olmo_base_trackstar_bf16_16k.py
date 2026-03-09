#!/usr/bin/env python3
"""
Run trackstar scoring with the base OLMo-3-7B-Instruct model (no SFT)
against MMLU sociology on SmolLM2-135M-10B.

Uses bf16, no FSDP, token_batch_size=16384, GPU memory logging, and times each step.
"""

import subprocess
import threading
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
    run_path: str = "runs/olmo3_7b_base_trackstar_bf16_16k"
    token_batch_size: int = 16384
    projection_dim: int = 16
    nproc_per_node: int = 4


def gpu_logger(
    log_path: Path, interval: float = 10.0, stop_event: threading.Event | None = None
):
    """Background thread that logs nvidia-smi output periodically."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(
            "timestamp,gpu_id,utilization_gpu%,memory_used_MiB,memory_total_MiB,temperature_C\n"
        )
        while stop_event is None or not stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                ts = time.strftime("%H:%M:%S")
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        f.write(f"{ts},{line.strip()}\n")
                f.flush()
            except Exception as e:
                f.write(f"# error: {e}\n")
                f.flush()
            if stop_event and stop_event.wait(interval):
                break
            elif stop_event is None:
                time.sleep(interval)


def run_trackstar(exp_cfg: ExperimentConfig):
    """Run trackstar scoring with the base model, timing each step."""
    print("=" * 60)
    print("Trackstar scoring with base model (bf16, no FSDP, batch=16384)")
    print("=" * 60)

    run_path = Path(exp_cfg.run_path)

    # Start GPU logger
    stop_event = threading.Event()
    log_path = run_path / "gpu_log.csv"
    logger_thread = threading.Thread(
        target=gpu_logger, args=(log_path, 10.0, stop_event), daemon=True
    )
    logger_thread.start()
    print(f"  GPU logging to {log_path}")

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

    # Stop GPU logger
    stop_event.set()
    logger_thread.join(timeout=5)

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

    # Print GPU memory summary
    print("\n" + "=" * 60)
    print("GPU memory summary")
    print("=" * 60)
    try:
        import csv

        with open(log_path) as f:
            reader = csv.DictReader(f)
            mem_by_gpu: dict[str, list[float]] = {}
            util_by_gpu: dict[str, list[float]] = {}
            for row in reader:
                gpu_id = row["gpu_id"].strip()
                mem_by_gpu.setdefault(gpu_id, []).append(
                    float(row["memory_used_MiB"].strip())
                )
                util_by_gpu.setdefault(gpu_id, []).append(
                    float(row["utilization_gpu%"].strip())
                )
            for gpu_id in sorted(mem_by_gpu.keys()):
                mem = mem_by_gpu[gpu_id]
                util = util_by_gpu[gpu_id]
                print(
                    f"  GPU {gpu_id}: mem avg={np.mean(mem):.0f} MiB, max={max(mem):.0f} MiB | util avg={np.mean(util):.0f}%, max={max(util):.0f}%"
                )
    except Exception as e:
        print(f"  Could not parse GPU log: {e}")

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

    top_idx = np.argsort(s)[-20:][::-1]
    with open(results_dir / "top_20.txt", "w") as f:
        for rank, idx in enumerate(top_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

    bot_idx = np.argsort(s)[:20]
    with open(results_dir / "bottom_20.txt", "w") as f:
        for rank, idx in enumerate(bot_idx, 1):
            text = ds[int(idx)]["text"]
            f.write(f"--- #{rank} | idx={idx} | score={s[idx]:.6f} ---\n")
            f.write(text)
            f.write("\n\n")

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
