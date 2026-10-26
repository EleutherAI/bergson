"""Generate a combined multi-GPU plot from CLI benchmarks.

Produces a single 1xN figure where each subplot shows build and
score runtime by training tokens, broken down by model. One
subplot per GPU count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson_cli import (
    CLIRunRecord,
    load_records,
)
from benchmarks.benchmark_utils import extract_gpu_info
from benchmarks.plot_styles import model_color


def create_cli_dataframe(
    records: list[CLIRunRecord],
) -> pd.DataFrame:
    """Create a dataframe from CLI benchmark records."""
    rows = []
    for r in records:
        if r.status == "success" and r.total_runtime_seconds is not None:
            rows.append(
                {
                    "model_key": r.model_key,
                    "model_name": r.model_name,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "dataset": r.dataset,
                    "batch_size": r.batch_size,
                    "build_seconds": r.build_seconds,
                    "reduce_seconds": r.reduce_seconds,
                    "score_seconds": r.score_seconds,
                    "total_runtime_seconds": (r.total_runtime_seconds),
                    "run_path": r.run_path,
                    "num_gpus": r.num_gpus,
                    "hardware": r.hardware,
                }
            )
    return pd.DataFrame(rows)


def _plot_build_score(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
) -> None:
    """Plot build and score runtime on a single axes."""
    for model_key in sorted(df["model_key"].unique()):
        color = model_color(model_key)
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if subset["build_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["build_seconds"],
                marker="s",
                color=color,
                label=f"{model_key} (build)",
                linewidth=2,
                linestyle="-",
            )
        if subset["score_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["score_seconds"],
                marker="D",
                color=color,
                label=f"{model_key} (score)",
                linewidth=2,
                linestyle=":",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Tokens", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.legend(fontsize=8, ncol=2)


def plot_cli_benchmark(
    panels: list[tuple[pd.DataFrame, str]],
    figure_path: Path,
    suptitle: str,
) -> None:
    """Create a combined 1xN plot, one subplot per GPU count."""
    if not panels:
        print("No data to plot")
        return

    ncols = len(panels)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(7 * ncols, 6),
        sharey=True,
    )
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        suptitle,
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for ax, (df, title) in zip(axes, panels):
        _plot_build_score(ax, df, title)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved CLI benchmark plot to {figure_path}")


def _gpu_name_from_df(
    df: pd.DataFrame | None,
) -> str:
    """Extract GPU name (without Nx prefix) from a df."""
    if df is None or df.empty:
        return "unknown"
    hw_col = df["hardware"].dropna()
    if hw_col.empty:
        return "unknown"
    gpu_info = extract_gpu_info(hw_col.iloc[0])
    if gpu_info:
        parts = gpu_info.split(" ", 1)
        if len(parts) == 2 and parts[0].endswith("x"):
            return parts[1]
        return gpu_info
    return "unknown"


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    if not path.exists():
        print(f"CSV not found: {path}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def _load_run_root(run_root: Path, num_gpus: int | None = None) -> pd.DataFrame:
    """Load records from a run root directory."""
    if not run_root.exists():
        print(
            f"Run root not found: {run_root}",
            file=sys.stderr,
        )
        return pd.DataFrame()
    records = load_records(run_root)
    if not records:
        print(f"No records in {run_root}", file=sys.stderr)
        return pd.DataFrame()
    df = create_cli_dataframe(records)
    if num_gpus is not None:
        df = df[df["num_gpus"] == num_gpus]

    # Keep only the latest run per (model, train_tokens)
    if not df.empty:
        df = df.sort_values("run_path").drop_duplicates(
            subset=["model_key", "train_tokens"],
            keep="last",
        )

    print(f"Loaded {len(df)} runs from {run_root}")
    return df


def _parse_source(
    value: str,
) -> tuple[str, int | None, Path]:
    """Parse a SOURCE argument: 'NUM_GPUS:PATH'.

    Returns (type, num_gpus, path) where type is 'csv' or 'dir'.
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Expected NUM_GPUS:PATH, got '{value}'")
    gpu_str, path_str = value.split(":", 1)
    num_gpus = int(gpu_str)
    path = Path(path_str)
    return (
        "csv" if path.suffix == ".csv" else "dir",
        num_gpus,
        path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined multi-GPU CLI "
        "benchmark plot. Each --source is NUM_GPUS:PATH "
        "where PATH is a CSV or run root directory.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "GPU count and data source as NUM_GPUS:PATH. "
            "PATH can be a .csv file or a run root dir. "
            "Repeat for multiple GPU configs."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save figure and CSVs",
    )

    args = parser.parse_args(argv)
    output_path = Path(args.output_path)

    # Load data for each GPU config
    gpu_data: dict[int, pd.DataFrame] = {}
    for source in args.source:
        src_type, num_gpus, path = _parse_source(source)
        if src_type == "csv":
            df = _load_csv(path)
        else:
            df = _load_run_root(path, num_gpus=num_gpus)
        if df is not None and not df.empty:
            gpu_data[num_gpus] = df

    if not gpu_data:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    # Save CSVs and build panels
    panels: list[tuple[pd.DataFrame, str]] = []
    hw_names: set[str] = set()

    for num_gpus in sorted(gpu_data):
        df = gpu_data[num_gpus]
        hw = _gpu_name_from_df(df)
        if hw != "unknown":
            hw_names.add(hw)

        label = hw.replace(" ", "_")
        csv_path = output_path / f"cli_benchmark_{num_gpus}x_{label}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {num_gpus}-GPU CSV to {csv_path}")

        title = f"{num_gpus} GPU: Build vs Score"
        panels.append((df, title))

    # Figure title
    title_hw = " & ".join(sorted(hw_names)) if hw_names else "unknown"
    file_hw = "_".join(sorted(hw_names)).replace(" ", "_")

    figure_path = output_path / f"cli_benchmark_{file_hw}.png"
    plot_cli_benchmark(
        panels,
        figure_path,
        f"Bergson CLI Benchmark ({title_hw})",
    )


if __name__ == "__main__":
    main()
