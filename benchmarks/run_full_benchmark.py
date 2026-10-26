"""Coordinate running dattri and bergson benchmarks and generate comparison plots."""

# python -m benchmarks.run_full_benchmark plot runs/benchmarks figures
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser, field

# Import from same directory
from benchmarks.benchmark_bergson import RunRecord as BergsonRecord
from benchmarks.benchmark_bergson import load_records as load_bergson_records
from benchmarks.benchmark_dattri import RunRecord as DattriRecord
from benchmarks.benchmark_dattri import load_records as load_dattri_records
from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    extract_gpu_info,
    format_tokens,
    parse_tokens,
)


def run_benchmark(
    method: str,
    model: str,
    train_tokens: int,
    eval_tokens: int,
    run_root: str,
    num_gpus: int = 1,
    **kwargs: Any,
) -> bool:
    """Run a single benchmark."""
    if method == "dattri":
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_dattri",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run_root",
            run_root,
            "--num_gpus",
            str(num_gpus),
        ]
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "max_length" in kwargs:
            cmd.extend(["--max_length", str(kwargs["max_length"])])
    elif method == "bergson":
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_bergson",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run_root",
            run_root,
            "--num_gpus",
            str(num_gpus),
        ]
        if "max_eval_examples" in kwargs:
            cmd.extend(["--max_eval_examples", str(kwargs["max_eval_examples"])])
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "max_length" in kwargs:
            cmd.extend(["--max_length", str(kwargs["max_length"])])
        # Enable FSDP for larger models (>= 1B parameters)
        if model in MODEL_SPECS and MODEL_SPECS[model].params >= 1_000_000_000:
            cmd.append("--fsdp")
    else:
        raise ValueError(f"Unknown method: {method}")

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
    bergson_records = (
        load_bergson_records(bergson_root) if bergson_root.exists() else []
    )
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
            # Handle records that may not have num_gpus (backwards compatibility)
            num_gpus = getattr(r, "num_gpus", 1)
            rows.append(
                {
                    "method": "dattri",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "num_gpus": num_gpus,
                    "runtime_seconds": r.runtime_seconds,
                    "reduce_seconds": None,  # Dattri doesn't separate reduce/score
                    "score_seconds": None,
                }
            )

    # Add bergson records
    for r in bergson_records:
        if r.status == "success":
            # Calculate total runtime for in-memory bergson (query + build + score)
            total_runtime = None
            if all(
                x is not None
                for x in [r.query_seconds, r.build_seconds, r.score_seconds]
            ):
                total_runtime = (
                    (r.query_seconds or 0)
                    + (r.build_seconds or 0)
                    + (r.score_seconds or 0)
                )

            if total_runtime is not None:
                # Handle records that may not have num_gpus (backwards compatibility)
                num_gpus = getattr(r, "num_gpus", 1)
                rows.append(
                    {
                        "method": "bergson-inmem",
                        "model_key": r.model_key,
                        "model_params": r.params,
                        "train_tokens": r.train_tokens,
                        "eval_tokens": r.eval_tokens,
                        "num_gpus": num_gpus,
                        "runtime_seconds": total_runtime,
                        "reduce_seconds": None,  # No separate reduce step in in-memory
                        "score_seconds": r.score_seconds,
                    }
                )

    return pd.DataFrame(rows)


def plot_comparison_ax(
    df: pd.DataFrame,
    ax: plt.Axes,
    title_suffix: str = "",
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot runtime vs training tokens onto a given axes."""
    df = df[df["runtime_seconds"].notna()].copy()
    for method in df["method"].unique():
        for model_key in df["model_key"].unique():
            subset = df[(df["method"] == method) & (df["model_key"] == model_key)]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                label = f"{method}-{model_key}"
                kwargs: dict = {}
                if color_map and label in color_map:
                    kwargs["color"] = color_map[label]
                ax.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker="o",
                    label=label,
                    linewidth=1.5,
                    **kwargs,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Tokens")
    ax.set_ylabel("Score Runtime (seconds)")
    ax.set_title(f"Score Runtime by Training Tokens{title_suffix}")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.legend(fontsize="small", ncol=2)


def plot_comparison(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create standalone comparison plot."""
    if df.empty:
        print("No data to plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_comparison_ax(df, ax, title_suffix)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


@dataclass
class RunBenchmarkConfig:
    """Configuration for running benchmarks."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    models: list[str] | None = None
    """Models to benchmark (default: pythia-14m, pythia-70m)."""

    token_scales: list[str] = field(default_factory=lambda: ["1M", "2M", "5M", "10M"])
    """Token scales to test (e.g. 1M 10M)."""

    eval_tokens: str = "100K"
    """Evaluation tokens."""

    batch_size: int = 4
    """Batch size."""

    max_length: int = 512
    """Maximum sequence length."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    num_test: int = 10
    """Number of test examples for bergson."""

    skip_dattri: bool = False
    """Skip dattri benchmarks."""

    skip_bergson: bool = False
    """Skip bergson benchmarks."""

    force: bool = False
    """Re-run existing benchmarks."""


@dataclass
class BenchmarkRun:
    """Run benchmarks for specified models and token scales."""

    run_cfg: RunBenchmarkConfig

    def execute(self) -> None:
        """Run benchmarks for specified models and token scales."""
        models = self.run_cfg.models or ["pythia-14m", "pythia-70m"]
        token_scales = [parse_tokens(ts) for ts in self.run_cfg.token_scales]
        eval_tokens = parse_tokens(self.run_cfg.eval_tokens)
        num_gpus = self.run_cfg.num_gpus

        dattri_root = Path(self.run_cfg.run_root) / "dattri-scaling"
        bergson_root = Path(self.run_cfg.run_root) / "bergson-scaling"

        # Check existing runs
        dattri_records, bergson_records = load_all_records(dattri_root, bergson_root)
        existing_dattri = {
            (r.model_key, r.train_tokens, r.eval_tokens, getattr(r, "num_gpus", 1))
            for r in dattri_records
            if r.status == "success"
        }
        existing_bergson = {
            (r.model_key, r.train_tokens, r.eval_tokens, getattr(r, "num_gpus", 1))
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
                if not self.run_cfg.skip_dattri:
                    key = (model, train_tokens, eval_tokens, num_gpus)
                    if key not in existing_dattri or self.run_cfg.force:
                        print(f"\n{'='*60}")
                        print(
                            f"Running Dattri: {model}, {format_tokens(train_tokens)} "
                            f"train tokens, {num_gpus} GPU(s)"
                        )
                        print(f"{'='*60}")
                        success = run_benchmark(
                            "dattri",
                            model,
                            train_tokens,
                            eval_tokens,
                            str(dattri_root),
                            num_gpus=num_gpus,
                            batch_size=self.run_cfg.batch_size,
                            max_length=self.run_cfg.max_length,
                        )
                        if not success:
                            print(
                                f"Failed to run dattri benchmark for {model} "
                                f"{format_tokens(train_tokens)}"
                            )
                    else:
                        print(
                            f"Skipping dattri {model} {format_tokens(train_tokens)} "
                            f"{num_gpus}gpu (already exists)"
                        )

                # Run bergson
                if not self.run_cfg.skip_bergson:
                    key = (model, train_tokens, eval_tokens, num_gpus)
                    if key not in existing_bergson or self.run_cfg.force:
                        print(f"\n{'='*60}")
                        print(
                            f"Running Bergson: {model}, {format_tokens(train_tokens)} "
                            f"train tokens, {num_gpus} GPU(s)"
                        )
                        print(f"{'='*60}")
                        success = run_benchmark(
                            "bergson",
                            model,
                            train_tokens,
                            eval_tokens,
                            str(bergson_root),
                            num_gpus=num_gpus,
                            batch_size=self.run_cfg.batch_size,
                            max_length=self.run_cfg.max_length,
                            max_eval_examples=self.run_cfg.num_test,
                        )
                        if not success:
                            print(
                                f"Failed to run bergson benchmark for {model} "
                                f"{format_tokens(train_tokens)}"
                            )
                    else:
                        print(
                            f"Skipping bergson {model} {format_tokens(train_tokens)} "
                            f"{num_gpus}gpu (already exists)"
                        )


@dataclass
class PlotConfig:
    """Configuration for generating comparison plots."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    output_path: str = field(positional=True)
    """Path to save benchmark results."""

    num_gpus: int | None = None
    """Filter plots to specific GPU count (default: generate all)."""

    skip_plots: bool = False
    """Skip generating plots."""


@dataclass
class Plot:
    """Generate comparison plots from existing benchmark results."""

    plot_cfg: PlotConfig

    def load_projection_comparison_data(
        self, csv_path: Path, projection_type: str
    ) -> pd.DataFrame:
        """Load comparison data from projection comparison CSV."""
        df = pd.read_csv(csv_path)

        # Filter by projection type
        if projection_type == "without":
            df_filtered = df[df["projection"] == "without"].copy()
        elif projection_type == "with":
            df_filtered = df[df["projection"] == "with (16)"].copy()
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # Keep existing column names (they already match)

        # Add missing columns for compatibility
        model_params_map = {
            "pythia-14m": 14000000,
            "pythia-70m": 70000000,
            "pythia-160m": 160000000,
            "pythia-410m": 410000000,
            "pythia-1b": 1000000000,
        }
        df_filtered["model_params"] = df_filtered["model_key"].apply(
            lambda x: model_params_map.get(x, 0)
        )
        df_filtered["eval_tokens"] = 1024  # Standard eval tokens
        df_filtered["num_gpus"] = 1  # All projection data is 1 GPU
        df_filtered["reduce_seconds"] = None
        df_filtered["score_seconds"] = None

        return df_filtered

    def execute(self) -> None:
        """Generate comparison plots from existing benchmark results."""
        # Use projection comparison data for fair in-memory comparison
        projection_csv = Path("runs/benchmarks/projection_comparison_1gpu.csv")

        if projection_csv.exists():
            projection_types = [
                ("with", "with_projection"),
                ("without", "without_projection"),
            ]

            # Load and save CSVs for each projection type
            loaded: dict[str, pd.DataFrame] = {}
            out = Path(self.plot_cfg.output_path)
            for proj_type, file_suffix in projection_types:
                df = self.load_projection_comparison_data(projection_csv, proj_type)
                if df.empty:
                    print(f"No data for {proj_type} projection, " "skipping")
                    continue
                csv_path = out / f"{file_suffix}_comparison.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"Saved {proj_type} projection data to " f"{csv_path}")
                loaded[proj_type] = df

            # Combined plot with both projection types
            if not self.plot_cfg.skip_plots and loaded:
                num_gpus = self.plot_cfg.num_gpus or 1

                # Extract GPU tag from data if available
                combined = pd.concat(loaded.values())
                gpu_names = combined.get("gpu_name")
                if gpu_names is not None:
                    names = gpu_names.dropna().unique()
                    if len(names) == 1:
                        gpu_tag = f"{num_gpus}x {names[0]}"
                    elif len(names) > 1:
                        gpu_tag = "mixed hardware"
                    else:
                        gpu_tag = f"{num_gpus} GPU"
                else:
                    # Fallback to hardware string
                    hw_col = combined.get("hardware")
                    if hw_col is not None:
                        hw_vals = hw_col.dropna().unique()
                        if len(hw_vals) == 1:
                            info = extract_gpu_info(hw_vals[0])
                            gpu_tag = info or f"{num_gpus} GPU"
                        elif len(hw_vals) > 1:
                            gpu_tag = "mixed hardware"
                        else:
                            gpu_tag = f"{num_gpus} GPU"
                    else:
                        gpu_tag = f"{num_gpus} GPU"

                panels = []
                for proj_type, _ in projection_types:
                    if proj_type not in loaded:
                        continue
                    df = loaded[proj_type]
                    if "num_gpus" in df.columns:
                        df = df[df["num_gpus"] == num_gpus]
                    if not df.empty:
                        suffix = f" ({proj_type} projection," f" {gpu_tag})"
                        panels.append((df, suffix))

                if panels:
                    # Build a shared color map so the same
                    # series gets the same color in each panel.
                    all_labels: list[str] = []
                    for pdf, _ in panels:
                        for method in pdf["method"].unique():
                            for mk in pdf["model_key"].unique():
                                label = f"{method}-{mk}"
                                if label not in all_labels:
                                    all_labels.append(label)
                    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                    color_map = {
                        label: prop_cycle[i % len(prop_cycle)]
                        for i, label in enumerate(all_labels)
                    }

                    n = len(panels)
                    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
                    if n == 1:
                        axes = [axes]
                    for ax, (pdf, suffix) in zip(axes, panels):
                        plot_comparison_ax(pdf, ax, suffix, color_map)
                    plt.tight_layout()
                    plot_path = out / "projection_comparison.png"
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(plot_path, dpi=200)
                    plt.close()
                    print(f"Saved projection comparison to " f"{plot_path}")

        else:
            # Fallback to directory loading (original behavior)
            dattri_root = Path(self.plot_cfg.run_root) / "dattri-scaling"
            bergson_root = (
                Path(self.plot_cfg.run_root) / "proj_comparison" / "bergson_noproj"
            )
            dattri_records, bergson_records = load_all_records(
                dattri_root, bergson_root
            )
            df = create_comparison_dataframe(dattri_records, bergson_records)

            # Save CSV
            out = Path(self.plot_cfg.output_path)
            csv_path = out / "comparison.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved comparison data to {csv_path}")

            # Create plots
            if not self.plot_cfg.skip_plots:
                # Get unique GPU counts in the data
                gpu_counts = (
                    sorted(df["num_gpus"].unique()) if "num_gpus" in df.columns else [1]
                )

                if self.plot_cfg.num_gpus is not None:
                    # Filter to specific GPU count
                    gpu_counts = [self.plot_cfg.num_gpus]

                for num_gpus in gpu_counts:
                    if "num_gpus" in df.columns:
                        gpu_df = df[df["num_gpus"] == num_gpus]
                    else:
                        gpu_df = df

                    if gpu_df.empty:
                        print(f"No data for {num_gpus} GPU(s), skipping")
                        continue

                    title_suffix = f" ({num_gpus} GPU{'s' if num_gpus > 1 else ''})"
                    plot_path = out / f"comparison_{num_gpus}gpu.png"
                    plot_comparison(gpu_df, plot_path, title_suffix)


@dataclass
class Main:
    """Coordinate dattri and bergson benchmarks."""

    command: BenchmarkRun | Plot

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(description="Coordinate dattri and bergson benchmarks")
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
