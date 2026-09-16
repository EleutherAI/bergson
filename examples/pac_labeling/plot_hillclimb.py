"""Plot a hill-climb leaderboard written by ``hillclimb.py``.

Usage:
    python examples/pac_labeling/plot_hillclimb.py --leaderboard <out>/leaderboard.csv \
        --out <dir>

One panel per task and ``eps``: certified save (net of the expert samples)
against the expert sample size ``m`` for each uncertainty score under the
betting bound, with the oracle save as a dashed line for the best score.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaderboard", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.leaderboard)
    df = df[df.method == "betting"]
    cells = sorted(set(zip(df.task, df.eps)))
    fig, axes = plt.subplots(
        1, len(cells), figsize=(3.4 * len(cells), 3.2), squeeze=False
    )
    for ax, (task, eps) in zip(axes[0], cells):
        cell = df[(df.task == task) & (df.eps == eps)]
        names = (
            cell.groupby("uncertainty")
            .pac_save.mean()
            .sort_values(ascending=False)
            .index
        )
        for color, name in zip(PALETTE, names):
            rows = cell[cell.uncertainty == name].sort_values("m")
            ax.plot(
                rows.m, 100 * rows.pac_save, "-o", color=color, ms=4, lw=1.5, label=name
            )
        best = cell[cell.uncertainty == names[0]].oracle_save.mean()
        ax.axhline(100 * best, color="#555555", linestyle="--", lw=1)
        ax.set_title(f"{task}, eps={eps:g}", fontsize=9)
        ax.set_xlabel("expert samples m")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=6, frameon=False)
    axes[0, 0].set_ylabel("certified cheap-labeled (%)")
    fig.suptitle(
        "PAC save against expert sample size; dashed = oracle save of the best score",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out / "hillclimb.pdf")
    fig.savefig(args.out / "hillclimb.png", dpi=150)


if __name__ == "__main__":
    main()
