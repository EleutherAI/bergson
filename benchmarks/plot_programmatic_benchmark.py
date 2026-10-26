"""Generate a build-vs-score plot from programmatic benchmarks.

Produces a single figure showing build and score runtime by
training tokens, broken down by model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import RunRecord, load_records
from benchmarks.benchmark_utils import (
    extract_gpu_info,
    get_hardware_info,
)
from benchmarks.plot_styles import model_color


def create_dataframe(
    records: list[RunRecord],
) -> pd.DataFrame:
    """Create a dataframe from benchmark records."""
    rows = []

    for r in records:
        if r.status == "success":
            hw = getattr(r, "hardware", None) or get_hardware_info()
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
                    "score_seconds": r.score_seconds,
                    "run_path": r.run_path,
                    "num_gpus": r.num_gpus,
                    "hardware": hw,
                    "gpu_name": getattr(r, "gpu_name", None),
                    "num_gpus_available": getattr(r, "num_gpus_available", None),
                    "gpu_vram_gb": getattr(r, "gpu_vram_gb", None),
                    "token_batch_size": r.token_batch_size,
                    "projection_dim": r.projection_dim,
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
                marker="^",
                color=color,
                label=f"{model_key} (build)",
                linewidth=2,
                linestyle="--",
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


def plot_programmatic_benchmark(
    df: pd.DataFrame,
    figure_path: Path,
    title: str,
) -> None:
    """Create a single build-vs-score plot."""
    if df.empty:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_build_score(ax, df, title)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved programmatic benchmark plot to" f" {figure_path}")


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
    df = create_dataframe(records)
    if num_gpus is not None:
        df = df[df["num_gpus"] == num_gpus]
    print(f"Loaded {len(df)} runs from {run_root}")
    return df


def _parse_source(
    value: str,
) -> tuple[str, int | None, Path]:
    """Parse a SOURCE argument: 'NUM_GPUS:PATH'.

    Returns (type, num_gpus, path) where type is
    'csv' or 'dir'.
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
        description="Generate programmatic benchmark "
        "plot. --source is NUM_GPUS:PATH where PATH is "
        "a CSV or run root directory.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "GPU count and data source as NUM_GPUS:PATH. "
            "PATH can be a .csv file or a run root dir."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save figure and CSV",
    )

    args = parser.parse_args(argv)
    output_path = Path(args.output_path)

    src_type, num_gpus, path = _parse_source(args.source)
    if src_type == "csv":
        df = _load_csv(path)
    else:
        df = _load_run_root(path, num_gpus=num_gpus)

    if df is None or df.empty:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    hw = _gpu_name_from_df(df)
    label = hw.replace(" ", "_")

    csv_path = (
        output_path / "archive" / f"programmatic_benchmark" f"_{num_gpus}x_{label}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    gpu_tag = f"{num_gpus}x {hw}" if hw != "unknown" else f"{num_gpus} GPU"
    title = f"{gpu_tag}: Build vs Score"

    figure_path = output_path / f"programmatic_benchmark_{num_gpus}x_{label}.png"
    plot_programmatic_benchmark(df, figure_path, title)


if __name__ == "__main__":
    main()
