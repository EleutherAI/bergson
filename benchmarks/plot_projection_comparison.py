"""Generate comparison plots for projection vs no-projection benchmarks.

4-panel layout (1 row x 4 columns):
  1. With projection + fixed batch size
  2. Without projection + fixed batch size
  3. With projection + optimal batch size
  4. Without projection + optimal batch size

Each panel shows bergson vs dattri across models and token scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import (
    RunRecord as BergsonRecord,
)
from benchmarks.benchmark_bergson import (
    load_records as load_bergson_records,
)
from benchmarks.benchmark_dattri import (
    RunRecord as DattriRecord,
)
from benchmarks.benchmark_dattri import (
    load_records as load_dattri_records,
)
from benchmarks.benchmark_utils import (
    extract_gpu_info,
    format_tokens,
)
from benchmarks.plot_styles import model_color

BENCH_ROOT = Path("/projects/a6a/public/lucia/proj_bench")
FIXED_BERGSON_ROOT = Path("/projects/a6a/public/lucia/proj_bench_bergson_fixedbatch")

# method -> (marker, linestyle)
METHOD_STYLES = {
    "bergson": ("o", "-"),
    "dattri": ("^", "--"),
}


def _hw_fields(r: object) -> dict:
    """Extract hardware fields from a record."""
    return {
        "hardware": getattr(r, "hardware", None),
        "gpu_name": getattr(r, "gpu_name", None),
        "num_gpus_available": getattr(r, "num_gpus_available", None),
        "gpu_vram_gb": getattr(r, "gpu_vram_gb", None),
    }


def _bergson_rows(
    records: list[BergsonRecord],
    projection: str,
) -> list[dict]:
    rows = []
    for r in records:
        if r.status != "success":
            continue
        total = (r.query_seconds or 0) + (r.score_seconds or 0)
        rows.append(
            {
                "method": "bergson",
                "projection": projection,
                "model_key": r.model_key,
                "model_params": r.params,
                "train_tokens": r.train_tokens,
                "runtime_seconds": total,
                **_hw_fields(r),
            }
        )
    return rows


def _dattri_rows(
    records: list[DattriRecord],
    projection: str,
) -> list[dict]:
    rows = []
    for r in records:
        if r.status != "success":
            continue
        if r.runtime_seconds is None:
            continue
        rows.append(
            {
                "method": "dattri",
                "projection": projection,
                "model_key": r.model_key,
                "model_params": r.params,
                "train_tokens": r.train_tokens,
                "runtime_seconds": r.runtime_seconds,
                **_hw_fields(r),
            }
        )
    return rows


def _load_dir(path: Path, loader):
    if path.exists():
        return loader(path)
    return []


def load_condition(
    bergson_proj_root: Path,
    bergson_noproj_root: Path,
    dattri_proj_root: Path,
    dattri_noproj_root: Path,
) -> pd.DataFrame:
    """Load all four method x projection combos."""
    rows: list[dict] = []
    rows += _bergson_rows(
        _load_dir(bergson_proj_root, load_bergson_records),
        "with",
    )
    rows += _bergson_rows(
        _load_dir(bergson_noproj_root, load_bergson_records),
        "without",
    )
    rows += _dattri_rows(
        _load_dir(dattri_proj_root, load_dattri_records),
        "with",
    )
    rows += _dattri_rows(
        _load_dir(dattri_noproj_root, load_dattri_records),
        "without",
    )
    return pd.DataFrame(rows)


def _plot_panel(ax, df: pd.DataFrame, title: str):
    """Plot one panel: bergson + dattri, all models."""
    if df.empty:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    for model_key in sorted(df["model_key"].unique()):
        color = model_color(model_key)
        for method in ["bergson", "dattri"]:
            subset = df[(df["method"] == method) & (df["model_key"] == model_key)]
            if subset.empty:
                continue
            subset = subset.sort_values("train_tokens")
            marker, linestyle = METHOD_STYLES[method]
            ax.plot(
                subset["train_tokens"],
                subset["runtime_seconds"],
                marker=marker,
                color=color,
                linestyle=linestyle,
                label=f"{method} ({model_key})",
                linewidth=1.5,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Tokens", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.legend(fontsize=7, ncol=1)


def plot_comparison(
    fixed_df: pd.DataFrame,
    optimal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create 1x4 comparison plot."""
    panels = [
        (
            "With Projection\n(Fixed Batch)",
            fixed_df[fixed_df["projection"] == "with"],
        ),
        (
            "Without Projection\n(Fixed Batch)",
            fixed_df[fixed_df["projection"] == "without"],
        ),
        (
            "With Projection\n(Optimal Batch)",
            optimal_df[optimal_df["projection"] == "with"],
        ),
        (
            "Without Projection\n(Optimal Batch)",
            optimal_df[optimal_df["projection"] == "without"],
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    fig.suptitle(
        "Projection Comparison: Bergson vs Dattri",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for ax, (title, panel_df) in zip(axes, panels):
        _plot_panel(ax, panel_df, title)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate projection comparison plots",
    )
    parser.add_argument(
        "--output_dir",
        default="docs/benchmarks",
    )
    args = parser.parse_args(argv)

    # Fixed batch: bergson tbs=4096, dattri bs=4
    print("Loading fixed-batch data...")
    fixed_df = load_condition(
        bergson_proj_root=FIXED_BERGSON_ROOT / "bergson_proj",
        bergson_noproj_root=FIXED_BERGSON_ROOT / "bergson_noproj",
        dattri_proj_root=BENCH_ROOT / "dattri_proj_old_bs4",
        dattri_noproj_root=BENCH_ROOT / "dattri_noproj_old_bs4",
    )

    # Optimal batch: auto-tuned batch sizes
    print("Loading optimal-batch data...")
    optimal_df = load_condition(
        bergson_proj_root=BENCH_ROOT / "bergson_proj",
        bergson_noproj_root=BENCH_ROOT / "bergson_noproj",
        dattri_proj_root=BENCH_ROOT / "dattri_proj",
        dattri_noproj_root=BENCH_ROOT / "dattri_noproj",
    )

    print(f"Fixed batch: {len(fixed_df)} runs")
    print(f"Optimal batch: {len(optimal_df)} runs")

    if fixed_df.empty and optimal_df.empty:
        print("No benchmark records found")
        return

    # Summary
    for label, df in [
        ("Fixed", fixed_df),
        ("Optimal", optimal_df),
    ]:
        print(f"\n{label} batch data:")
        for method in sorted(df["method"].unique()):
            for proj in ["with", "without"]:
                subset = df[(df["method"] == method) & (df["projection"] == proj)]
                if subset.empty:
                    continue
                models = sorted(subset["model_key"].unique())
                tokens = sorted(subset["train_tokens"].unique())
                print(f"  {method} ({proj} proj):" f" {len(subset)} runs")
                print(f"    Models: {models}")
                print(f"    Tokens:" f" {[format_tokens(t) for t in tokens]}")

    # Derive hardware suffix from data
    combined = pd.concat(
        [
            fixed_df.assign(batch_strategy="fixed"),
            optimal_df.assign(batch_strategy="optimal"),
        ],
        ignore_index=True,
    )
    hw_sample = combined["hardware"].dropna().iloc[0]
    gpu_info = extract_gpu_info(hw_sample)
    hw_suffix = f"_{gpu_info.replace(' ', '_')}" if gpu_info else ""

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save combined CSV
    csv_path = out / "archive" / f"projection_comparison{hw_suffix}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # Plot
    plot_path = out / f"projection_comparison{hw_suffix}.png"
    plot_comparison(fixed_df, optimal_df, plot_path)


if __name__ == "__main__":
    main()
