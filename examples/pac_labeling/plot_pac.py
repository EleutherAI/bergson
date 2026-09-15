"""Plot PAC labeling trials written by ``pac_attribution.py``.

Usage:
    python examples/pac_labeling/plot_pac.py --trials <out>/trials.csv [...] \
        --out <dir>

``pac_labeling.pdf`` shows realized loss against the fraction of documents
that kept the cheap score, one panel per task and ``eps``, fifty trials per
uncertainty score, with ``eps`` as a dashed line. ``pac_hybrid.pdf`` shows the
hybrid score vector's top 1% overlap with the expert and its LDS against the
fraction saved, with the cheap-only and expert-only baselines as lines.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

COLORS = {
    "proponent": "#2a78d6",
    "boundary": "#eb6834",
    "magnitude": "#eb6834",
    "learned": "#1baf7a",
    "random": "#eda100",
}
ORDER = ["proponent", "boundary", "magnitude", "learned", "random"]
TASK_LABEL = {
    "proponent": "top-1% membership (0-1)",
    "recall": "missed proponents (0-1)",
    "score": "score error / cutoff² (clipped)",
}


def style(ax):
    ax.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-uncertainty", type=int, default=50)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(p) for p in args.trials], ignore_index=True)
    base = df[df.trial < 0].groupby("uncertainty")[["top1_overlap", "lds"]].mean()
    pac = df[df.trial >= 0]
    tasks = [t for t in TASK_LABEL if t in set(pac.task)]
    n_eps = max(pac[pac.task == t].eps.nunique() for t in tasks)

    fig, axes = plt.subplots(
        len(tasks), n_eps, figsize=(3.2 * n_eps, 2.9 * len(tasks)), squeeze=False
    )
    for i, task in enumerate(tasks):
        sub = pac[pac.task == task]
        for j, eps in enumerate(sorted(sub.eps.unique())):
            ax = axes[i, j]
            cell = sub[sub.eps == eps]
            for name in ORDER:
                rows = cell[cell.uncertainty == name]
                if rows.empty:
                    continue
                rows = rows.sample(min(args.per_uncertainty, len(rows)), random_state=0)
                ax.scatter(
                    rows.error,
                    100 * rows.budget_save,
                    s=9,
                    color=COLORS[name],
                    alpha=0.7,
                    linewidths=0,
                    label=name,
                )
            ax.axvline(eps, color="#555555", linestyle="--", linewidth=1)
            ax.axvline(
                cell.cheap_error.mean(), color="#999999", linestyle=":", linewidth=1
            )
            ax.set_xlim(left=0)
            ax.set_ylim(-3, 103)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
            ax.set_title(f"{TASK_LABEL[task]}, eps={eps:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("cheap-labeled (%)")
            if i == len(tasks) - 1:
                ax.set_xlabel("realized loss")
            style(ax)
        for j in range(sub.eps.nunique(), n_eps):
            axes[i, j].axis("off")
    seen: dict[str, object] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    labels = [n for n in ORDER if n in seen]
    handles = [seen[n] for n in labels]
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9
    )
    fig.suptitle(
        "PAC labeling: dashed = eps, dotted = cheap-only loss; "
        "every point is one trial (20 queries x 100 trials)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(args.out / "pac_labeling.pdf")
    fig.savefig(args.out / "pac_labeling.png", dpi=150)

    agg = (
        pac.groupby(["task", "uncertainty", "eps"])[
            ["budget_save", "top1_overlap", "lds"]
        ]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, metric, label in zip(
        axes,
        ["top1_overlap", "lds"],
        ["top-1% overlap with MAGIC", "LDS (100 subsets)"],
    ):
        for name in ORDER:
            rows = agg[agg.uncertainty == name]
            if rows.empty:
                continue
            ax.scatter(
                100 * rows.budget_save,
                rows[metric],
                s=18,
                color=COLORS[name],
                linewidths=0,
                label=name,
            )
        ax.axhline(base.loc["cheap_only", metric], color="#999999", linestyle=":")
        ax.axhline(base.loc["expert_only", metric], color="#555555", linestyle="--")
        ax.set_xlabel("cheap-labeled (%)")
        ax.set_ylabel(label)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Hybrid score vector: dotted = EK-FAC only, dashed = MAGIC only "
        "(means over queries and trials, all tasks and eps)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out / "pac_hybrid.pdf")
    fig.savefig(args.out / "pac_hybrid.png", dpi=150)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
