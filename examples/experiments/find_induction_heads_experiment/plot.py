"""Plotting utilities for the induction head experiment."""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch  # noqa: F401

from bergson.config import FaissConfig  # noqa: F401
from bergson.query.attributor import Attributor  # noqa: F401


def plot_influence_scores(data: pd.DataFrame, unit_norm: bool):
    """Plot influence scores of training examples vs training step."""
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.scatter(
        data["global_step"],
        data["score"],
        alpha=0.6,
        s=20,
    )
    plt.xlabel("Cumulative Training Steps")
    plt.ylabel("Influence Score")
    plt.title(
        f"Most Influential Training Examples "
        f"({'Normalized' if unit_norm else 'Unnormalized'})"
    )
    plt.grid(True, alpha=0.3)
    fig_name = f'figures/scores{"_norm" if unit_norm else ""}.pdf'
    plt.savefig(
        fig_name,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    print("Module-wise scores not yet supported for FAISS index")
    return

    # Produce the same plot but split out by module (i.e. key in the grads mmap)
    df_path = f"figures/module_scores{'_norm' if unit_norm else ''}.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print(f"Loaded module scores from {df_path}")
    else:
        data = []
        for epoch_idx in range(num_train_epochs):  # noqa: F821
            attributor = Attributor(  # noqa: F841
                index_path=f"{trainer.args.output_dir}/gradients/train/epoch_{epoch_idx}",  # noqa: F821
                device="cpu",
                unit_norm=args.unit_norm,  # noqa: F821
                dtype=torch.float32,
                faiss_cfg=FaissConfig(
                    mmap_index=True, index_factory="IVF1,SQfp16", num_shards=10
                ),
            )

            for name, grad in mean_module_induction_gradients.items():  # noqa: F821
                if "attention" not in name and "attn" not in name:
                    print(f"Skipping {name}")
                    continue
                else:
                    print(f"Processing {name}")

                mod_inner_products, _ = attributor.search(
                    {name: grad}, k=None, modules=[name]
                )

                for i, score in enumerate(mod_inner_products.squeeze()):
                    training_metadata = training_order[  # noqa: F821
                        (training_order["_idx"] == i)  # noqa: F821
                        & (training_order["epoch"] == epoch_idx)  # noqa: F821
                    ]
                    if len(training_metadata) != 1:
                        continue
                    for row in training_metadata.itertuples(index=False):
                        data.append(
                            {
                                "global_step": row.global_step,
                                "epoch": epoch_idx,
                                "module": name,
                                "score": score.item(),
                            }
                        )

        df = pd.DataFrame(data)
        df.to_csv(df_path, index=False)

    attn_modules = [name for name in df["module"].unique() if "attn" in name]
    non_attn_modules = [name for name in df["module"].unique() if "attn" not in name]

    for module in non_attn_modules:
        name = module
        module_data = df[df["module"] == module]

        plt.figure(figsize=(12, 8))

        plt.scatter(
            module_data["global_step"],
            module_data["score"],
            # c=module_data["epoch"],
            alpha=0.6,
            s=20,
            label=f"Module {name}",
        )
        plt.xlabel("Training Step")
        plt.ylabel("Influence Score")
        plt.title(
            f"Most Influential Training Examples for {name} "
            f"({'Normalized' if unit_norm else 'Unnormalized'})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_name = f'figures/module_scores{"_norm" if unit_norm else ""}_{name}.pdf'
        plt.savefig(
            fig_name,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

        # Add a line plot with the sum of the gradients for each module
        # Sum points at each global step
        module_data = module_data.groupby(["global_step", "epoch"], as_index=False).agg(
            score=("score", "sum")
        )
        plt.figure(figsize=(12, 8))
        plt.plot(
            module_data["global_step"],
            module_data["score"],
            label=f"Module {name}",  # c=module_data["epoch"]
        )
        plt.xlabel("Training Step")
        plt.ylabel("Sum of Gradients")
        plt.title(f"Sum of Gradients for {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_name = f'figures/sum{"_norm" if unit_norm else ""}_{name}.pdf'
        plt.savefig(
            fig_name,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Plot all attention heads in one file
    n = len(attn_modules)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True
    )

    for ax, module in zip(axes.flatten(), attn_modules):
        module_data = df[df["module"] == module]
        ax.scatter(
            module_data["global_step"],
            module_data["score"],
            alpha=0.6,
            s=20,
        )
        ax.set_title(module)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"figures/all_heads_scores{'_norm' if unit_norm else ''}.pdf")
    plt.close(fig)

    # Single figure with each attention modules' sum-of-scores over steps
    fig, ax = plt.subplots(figsize=(6, 4))

    for module in attn_modules:
        module_data = df[df["module"] == module]
        summed = module_data.groupby("global_step")["score"].sum().reset_index()
        ax.plot(summed["global_step"], summed["score"], label=module, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Sum of Scores")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.legend().remove()

    plt.tight_layout()
    fig.savefig(f"figures/all_heads_sum_scores{'_norm' if unit_norm else ''}.pdf")
    plt.close(fig)

    # Single figure with each attention modules' sum-of-scores summed over steps
    sums = [df[df["module"] == m]["score"].sum() for m in attn_modules]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(attn_modules)), sums)
    ax.set_xticks(range(len(attn_modules)))
    ax.set_xticklabels(attn_modules, rotation=90)
    ax.set_ylabel("Sum of Scores")
    ax.set_xlabel("Module")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"figures/all_heads_sum_scores_bar{'_norm' if unit_norm else ''}.pdf")
    plt.close(fig)
