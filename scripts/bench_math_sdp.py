"""Benchmark --force_math_sdp overhead on bergson build.

Runs bergson build with and without --force_math_sdp for each model,
measuring wall-clock time. Uses a small subset of pile-10k to keep
runs short while still being representative.
"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchResult:
    model: str
    force_math_sdp: bool
    precision: str
    use_tf32_matmuls: bool
    build_seconds: float
    token_batch_size: int
    dataset_split: str
    projection_dim: int


MODELS = [
    "EleutherAI/pythia-160m",
    "allenai/OLMo-2-0425-1B",
]

CONFIGS = [
    {"force_math_sdp": False, "precision": "bf16", "use_tf32_matmuls": False},
    {"force_math_sdp": True, "precision": "bf16", "use_tf32_matmuls": False},
    {"force_math_sdp": False, "precision": "fp32", "use_tf32_matmuls": True},
    {"force_math_sdp": True, "precision": "fp32", "use_tf32_matmuls": True},
    {"force_math_sdp": False, "precision": "fp32", "use_tf32_matmuls": False},
    {"force_math_sdp": True, "precision": "fp32", "use_tf32_matmuls": False},
]

SPLIT = "train[:500]"
TOKEN_BATCH_SIZE = 2048
PROJECTION_DIM = 16
OUTPUT_DIR = Path("runs/bench_math_sdp")


def run_build(
    model: str,
    run_path: str,
    force_math_sdp: bool,
    precision: str,
    use_tf32_matmuls: bool = False,
):
    cmd = [
        "bergson",
        "build",
        run_path,
        "--model",
        model,
        "--dataset",
        "NeelNanda/pile-10k",
        "--split",
        SPLIT,
        "--truncation",
        "--projection_dim",
        str(PROJECTION_DIM),
        "--token_batch_size",
        str(TOKEN_BATCH_SIZE),
        "--precision",
        precision,
        "--skip_preconditioners",
        "--nproc_per_node",
        "1",
        "--overwrite",
    ]
    if force_math_sdp:
        cmd.append("--force_math_sdp")
    if use_tf32_matmuls:
        cmd.append("--use_tf32_matmuls")

    print(f"\n{'=' * 60}")
    print(f"  Model: {model}")
    print(
        f"  precision={precision}, force_math_sdp={force_math_sdp},"
        f" tf32={use_tf32_matmuls}"
    )
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 60)

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    print(f"  Completed in {elapsed:.1f}s")
    return elapsed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[BenchResult] = []

    for model in MODELS:
        model_slug = model.split("/")[-1]
        for cfg in CONFIGS:
            force_math_sdp = cfg["force_math_sdp"]
            precision = cfg["precision"]
            use_tf32_matmuls = cfg["use_tf32_matmuls"]
            parts = []
            parts.append("math" if force_math_sdp else "default")
            parts.append(precision)
            if use_tf32_matmuls:
                parts.append("tf32")
            suffix = "_".join(parts)
            run_path = str(OUTPUT_DIR / f"{model_slug}_{suffix}")

            elapsed = run_build(
                model, run_path, force_math_sdp, precision, use_tf32_matmuls
            )
            if elapsed is not None:
                results.append(
                    BenchResult(
                        model=model,
                        force_math_sdp=force_math_sdp,
                        precision=precision,
                        use_tf32_matmuls=use_tf32_matmuls,
                        build_seconds=elapsed,
                        token_batch_size=TOKEN_BATCH_SIZE,
                        dataset_split=SPLIT,
                        projection_dim=PROJECTION_DIM,
                    )
                )

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"{'Model':<25s} {'Settings':<35s}" f" {'Time (s)':>9s} {'vs bf16':>9s}")
    print("-" * 80)

    # Compute overhead vs bf16 baseline
    bf16_baseline = {}
    for r in results:
        if r.precision == "bf16" and not r.force_math_sdp:
            bf16_baseline[r.model] = r.build_seconds

    for r in results:
        model_short = r.model.split("/")[-1]
        parts = [r.precision]
        if r.use_tf32_matmuls:
            parts.append("tf32_matmuls")
        if r.force_math_sdp:
            parts.append("force_math_sdp")
        settings = " + ".join(parts)

        baseline = bf16_baseline.get(r.model)
        if (
            baseline
            and r == results[0]
            or (r.precision == "bf16" and not r.force_math_sdp)
        ):
            overhead_str = "—"
        elif baseline:
            overhead = (r.build_seconds / baseline - 1) * 100
            overhead_str = f"{overhead:+.1f}%"
        else:
            overhead_str = "N/A"
        print(
            f"{model_short:<25s} {settings:<35s}"
            f" {r.build_seconds:>9.1f} {overhead_str:>9s}"
        )

    # Save JSON
    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
