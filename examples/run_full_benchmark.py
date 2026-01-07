"""Coordinate running dattri and bergson benchmarks and generate comparison plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

# Import from same directory
# try:
from benchmark_common import MODEL_SPECS
from benchmark_dattri import load_records as load_dattri_records
from benchmark_bergson import load_records as load_bergson_records
from benchmark_dattri import RunRecord as DattriRecord
from benchmark_bergson import RunRecord as BergsonRecord
# except ImportError:
    # from examples.benchmark_common import MODEL_SPECS
    # from examples.benchmark_dattri import load_records as load_dattri_records
    # from examples.benchmark_bergson import load_records as load_bergson_records
    # from examples.benchmark_dattri import RunRecord as DattriRecord
    # from examples.benchmark_bergson import RunRecord as BergsonRecord


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace(",", "")
    if text.endswith("tokens"):
        text = text[:-6]
    if not text:
        raise ValueError("empty token spec")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    unit = 1
    if text[-1] in suffixes:
        unit = suffixes[text[-1]]
        text = text[:-1]
    number = float(text)
    return int(number * unit)


def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        value = tokens / 1_000_000_000
        suffix = "B"
    elif tokens >= 1_000_000:
        value = tokens / 1_000_000
        suffix = "M"
    elif tokens >= 1_000:
        value = tokens / 1_000
        suffix = "K"
    else:
        return str(tokens)
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.2f}{suffix}"


def run_benchmark(
    method: str,
    model: str,
    train_tokens: int,
    eval_tokens: int,
    run_root: str,
    **kwargs: Any,
) -> bool:
    """Run a single benchmark."""
    if method == "dattri":
        cmd = [
            sys.executable,
            "-m",
            "examples.benchmark_dattri",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run-root",
            run_root,
        ]
    elif method == "bergson":
        cmd = [
            sys.executable,
            "-m",
            "examples.benchmark_bergson",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run-root",
            run_root,
        ]
        if "max_eval_examples" in kwargs:
            cmd.extend(["--max-eval-examples", str(kwargs["max_eval_examples"])])
    else:
        raise ValueError(f"Unknown method: {method}")

    if "batch_size" in kwargs:
        cmd.extend(["--batch-size", str(kwargs["batch_size"])])
    if "max_length" in kwargs:
        cmd.extend(["--max-length", str(kwargs["max_length"])])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {method} benchmark:")
        print(result.stderr)
        return False
    
    print(f"Successfully ran {method} benchmark")
    return True


def load_all_records(
    dattri_root: Path,
    bergson_root: Path,
) -> tuple[list[DattriRecord], list[BergsonRecord]]:
    """Load all benchmark records."""
    dattri_records = load_dattri_records(dattri_root) if dattri_root.exists() else []
    bergson_records = load_bergson_records(bergson_root) if bergson_root.exists() else []
    return dattri_records, bergson_records


def create_comparison_dataframe(
    dattri_records: list[DattriRecord],
    bergson_records: list[BergsonRecord],
) -> pd.DataFrame:
    """Create a combined dataframe for comparison."""
    rows = []
    
    # Add dattri records
    for r in dattri_records:
        if r.status == "success" and r.runtime_seconds is not None:
            rows.append({
                "method": "dattri",
                "model_key": r.model_key,
                "model_params": r.params,
                "train_tokens": r.train_tokens,
                "eval_tokens": r.eval_tokens,
                "runtime_seconds": r.runtime_seconds,
                "reduce_seconds": None,  # Dattri doesn't separate reduce/score
                "score_seconds": None,
            })
    
    # Add bergson records
    for r in bergson_records:
        if r.status == "success" and r.total_runtime_seconds is not None:
            rows.append({
                "method": "bergson",
                "model_key": r.model_key,
                "model_params": r.params,
                "train_tokens": r.train_tokens,
                "eval_tokens": r.eval_tokens,
                "runtime_seconds": r.total_runtime_seconds,
                "reduce_seconds": r.reduce_seconds,
                "score_seconds": r.score_seconds,
            })
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots."""
    if df.empty:
        print("No data to plot")
        return
    
    # Filter successful runs
    df = df[df["runtime_seconds"].notna()]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Runtime vs train tokens (by model)
    ax1 = axes[0, 0]
    for method in df["method"].unique():
        for model_key in df["model_key"].unique():
            subset = df[(df["method"] == method) & (df["model_key"] == model_key)]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                ax1.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker="o",
                    label=f"{method}-{model_key}",
                    linewidth=1.5,
                )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens")
    ax1.set_ylabel("Total Runtime (seconds)")
    ax1.set_title("Runtime Scaling: Total Time")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize="small", ncol=2)
    
    # Plot 2: Runtime vs model params (by token scale)
    ax2 = axes[0, 1]
    for method in df["method"].unique():
        for train_tokens in sorted(df["train_tokens"].unique())[:5]:  # Top 5 token scales
            subset = df[(df["method"] == method) & (df["train_tokens"] == train_tokens)]
            if not subset.empty:
                subset = subset.sort_values("model_params")
                ax2.plot(
                    subset["model_params"],
                    subset["runtime_seconds"],
                    marker="o",
                    label=f"{method}-{format_tokens(train_tokens)}",
                    linewidth=1.5,
                )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model Parameters")
    ax2.set_ylabel("Total Runtime (seconds)")
    ax2.set_title("Runtime Scaling: Model Size")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize="small", ncol=2)
    
    # Plot 3: Bergson reduce vs score breakdown
    ax3 = axes[1, 0]
    bergson_df = df[df["method"] == "bergson"]
    if not bergson_df.empty and bergson_df["reduce_seconds"].notna().any():
        for model_key in bergson_df["model_key"].unique():
            subset = bergson_df[bergson_df["model_key"] == model_key].sort_values("train_tokens")
            if subset["reduce_seconds"].notna().any():
                ax3.plot(
                    subset["train_tokens"],
                    subset["reduce_seconds"],
                    marker="s",
                    label=f"{model_key} (reduce)",
                    linewidth=1.5,
                    linestyle="-",
                )
            if subset["score_seconds"].notna().any():
                ax3.plot(
                    subset["train_tokens"],
                    subset["score_seconds"],
                    marker="^",
                    label=f"{model_key} (score)",
                    linewidth=1.5,
                    linestyle="--",
                )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training Tokens")
    ax3.set_ylabel("Runtime (seconds)")
    ax3.set_title("Bergson: Reduce vs Score Breakdown")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize="small")
    
    # Plot 4: Speedup comparison (dattri / bergson)
    ax4 = axes[1, 1]
    speedup_data = []
    for model_key in df["model_key"].unique():
        for train_tokens in df["train_tokens"].unique():
            dattri_subset = df[(df["method"] == "dattri") & (df["model_key"] == model_key) & (df["train_tokens"] == train_tokens)]
            bergson_subset = df[(df["method"] == "bergson") & (df["model_key"] == model_key) & (df["train_tokens"] == train_tokens)]
            
            if not dattri_subset.empty and not bergson_subset.empty:
                dattri_time = dattri_subset["runtime_seconds"].iloc[0]
                bergson_time = bergson_subset["runtime_seconds"].iloc[0]
                speedup = dattri_time / bergson_time if bergson_time > 0 else None
                if speedup is not None:
                    speedup_data.append({
                        "model_key": model_key,
                        "train_tokens": train_tokens,
                        "speedup": speedup,
                    })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        for model_key in speedup_df["model_key"].unique():
            subset = speedup_df[speedup_df["model_key"] == model_key].sort_values("train_tokens")
            ax4.plot(
                subset["train_tokens"],
                subset["speedup"],
                marker="o",
                label=model_key,
                linewidth=1.5,
            )
        ax4.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax4.set_xscale("log")
        ax4.set_xlabel("Training Tokens")
        ax4.set_ylabel("Speedup (dattri / bergson)")
        ax4.set_title("Relative Performance: Dattri vs Bergson")
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax4.legend(fontsize="small")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks for specified models and token scales."""
    models = args.models or ["pythia-14m", "pythia-70m"]
    token_scales = [parse_tokens(ts) for ts in args.token_scales]
    eval_tokens = parse_tokens(args.eval_tokens)
    
    dattri_root = Path(args.run_root) / "dattri-scaling"
    bergson_root = Path(args.run_root) / "bergson-scaling"
    
    # Check existing runs
    dattri_records, bergson_records = load_all_records(dattri_root, bergson_root)
    existing_dattri = {
        (r.model_key, r.train_tokens, r.eval_tokens)
        for r in dattri_records
        if r.status == "success"
    }
    existing_bergson = {
        (r.model_key, r.train_tokens, r.eval_tokens)
        for r in bergson_records
        if r.status == "success"
    }
    
    # Run benchmarks
    for model in models:
        if model not in MODEL_SPECS:
            print(f"Warning: Unknown model {model}, skipping")
            continue
        
        for train_tokens in token_scales:
            # Run dattri
            if not args.skip_dattri:
                key = (model, train_tokens, eval_tokens)
                if key not in existing_dattri or args.force:
                    print(f"\n{'='*60}")
                    print(f"Running Dattri: {model}, {format_tokens(train_tokens)} train tokens")
                    print(f"{'='*60}")
                    success = run_benchmark(
                        "dattri",
                        model,
                        train_tokens,
                        eval_tokens,
                        str(dattri_root),
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                    )
                    if not success:
                        print(f"Failed to run dattri benchmark for {model} {format_tokens(train_tokens)}")
                else:
                    print(f"Skipping dattri {model} {format_tokens(train_tokens)} (already exists)")
            
            # Run bergson
            if not args.skip_bergson:
                key = (model, train_tokens, eval_tokens)
                if key not in existing_bergson or args.force:
                    print(f"\n{'='*60}")
                    print(f"Running Bergson: {model}, {format_tokens(train_tokens)} train tokens")
                    print(f"{'='*60}")
                    success = run_benchmark(
                        "bergson",
                        model,
                        train_tokens,
                        eval_tokens,
                        str(bergson_root),
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        max_eval_examples=args.num_test,
                    )
                    if not success:
                        print(f"Failed to run bergson benchmark for {model} {format_tokens(train_tokens)}")
                else:
                    print(f"Skipping bergson {model} {format_tokens(train_tokens)} (already exists)")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate comparison plots from existing benchmark results."""
    dattri_root = Path(args.run_root) / "dattri-scaling"
    bergson_root = Path(args.run_root) / "bergson-scaling"
    
    dattri_records, bergson_records = load_all_records(dattri_root, bergson_root)
    
    df = create_comparison_dataframe(dattri_records, bergson_records)
    
    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison data to {csv_path}")
    
    # Create plots
    if not args.skip_plots:
        plot_path = Path(args.plot_output)
        plot_comparison(df, plot_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate dattri and bergson benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--models", nargs="*", help="Models to benchmark")
    run_parser.add_argument(
        "--token-scales",
        nargs="*",
        default=["1M", "2M", "5M", "10M"],
        help="Token scales to test (e.g. 1M 10M)",
    )
    run_parser.add_argument("--eval-tokens", default="100K", help="Evaluation tokens")
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--max-length", type=int, default=512)
    run_parser.add_argument("--num-test", type=int, default=10, help="Number of test examples for bergson")
    run_parser.add_argument("--run-root", default="runs")
    run_parser.add_argument("--skip-dattri", action="store_true")
    run_parser.add_argument("--skip-bergson", action="store_true")
    run_parser.add_argument("--force", action="store_true", help="Re-run existing benchmarks")
    run_parser.set_defaults(func=cmd_run)
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate comparison plots")
    plot_parser.add_argument("--run-root", default="runs")
    plot_parser.add_argument("--output-csv", default="data/benchmark_comparison.csv")
    plot_parser.add_argument("--plot-output", default="figures/benchmark_comparison.png")
    plot_parser.add_argument("--skip-plots", action="store_true")
    plot_parser.set_defaults(func=cmd_plot)
    
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

