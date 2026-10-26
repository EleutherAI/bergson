"""Main experiment orchestration for semantic experiments."""

import subprocess
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from .data import create_data
from .hessians import (
    build_style_indices,
    compute_between_hessian_means,
    compute_mixed_hessian,
)
from .metrics import compute_metrics
from .scoring import compute_scores_fast


def create_index(dataset_name: str, analysis_model_name: str) -> None:
    """Create a bergson index for a dataset.

    Args:
        dataset_name: Name or path of the dataset.
        analysis_model_name: Model to use for gradient collection.
    """
    run_path = Path(f"runs/{dataset_name}")
    cmd = [
        "bergson",
        "build",
        str(run_path / "index"),
        "--model",
        analysis_model_name,
        "--dataset",
        dataset_name,
        "--drop_columns",
        "False",
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
        "--fsdp",
        "--projection_dim",
        "16",
    ]

    print(" ".join(cmd))
    if not run_path.exists():
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)


def finetune(
    dataset_path: str, analysis_model_name: str, finetuned_model_path: str
) -> None:
    """Finetune a model on a dataset using LoRA.

    Args:
        dataset_path: Path to the training dataset.
        analysis_model_name: Base model to finetune.
        finetuned_model_path: Path to save the finetuned model.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29500",
        "--standalone",
        "examples/train_lora.py",
        "--dataset_name",
        dataset_path,
        "--finetuned_model_path",
        finetuned_model_path,
        "--model_name",
        analysis_model_name,
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
    ]
    print(" ".join(cmd))
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:  # type: ignore
            print(line.strip())

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def run_hessian_comparison() -> dict[str, dict[str, float]]:
    """Compare three preconditioning strategies on pirate+shakespeare data.

    Strategies:
    1. baseline: Hessian computed on whole combined dataset
    2. mixed: 0.5 * R_pirate + 0.5 * R_shakespeare
    3. r_between: R_pirate + R_shakespeare - R_combined (isolates style direction)
    4. no_hess: No preconditioning (control)

    Returns:
        Dictionary mapping strategy names to their computed statistics.
    """
    base_path = Path("runs/hess_comparison")

    # 1. Build indices if needed
    print("\n" + "=" * 60)
    print("STEP 1: Building indices")
    print("=" * 60)
    build_style_indices()

    # 2. Compute derived hessians
    print("\n" + "=" * 60)
    print("STEP 2: Computing derived hessians")
    print("=" * 60)
    compute_mixed_hessian(
        base_path / "pirate",
        base_path / "shakespeare",
        base_path / "mixed_50_50",
    )
    # Use means-based approach (more targeted at style direction)
    compute_between_hessian_means(
        base_path / "pirate",
        base_path / "shakespeare",
        base_path / "between",
    )

    # 3. Score with each hessian strategy (using fast index-vs-index scoring)
    print("\n" + "=" * 60)
    print("STEP 3: Computing scores with each strategy")
    print("=" * 60)
    strategies: list[tuple[str | None, str]] = [
        ("combined", "baseline"),  # Standard: precondition with combined R
        ("mixed_50_50", "mixed"),  # 50-50 mix of style-specific Rs
        ("between", "r_between"),  # Between-group hessian
        (None, "no_hess"),  # No preconditioning (control)
    ]

    for prec_path, name in strategies:
        print(f"\n--- Strategy: {name} ---")
        output_path = base_path / f"scores_{name}"
        compute_scores_fast(
            base_path / "combined",  # Use precomputed gradients from combined index
            output_path,
            hessian_path=(base_path / prec_path if prec_path else None),
        )

    # 4. Compare metrics across strategies
    print("\n" + "=" * 60)
    print("STEP 4: Comparing metrics across strategies")
    print("=" * 60)

    all_stats: dict[str, dict[str, float]] = {}
    for _, name in strategies:
        print(f"\n{'#' * 60}")
        print(f"# Strategy: {name}")
        print(f"{'#' * 60}")
        stats = compute_metrics(
            base_path / "combined",
            scores_path=base_path / f"scores_{name}",
            exclude_llama=True,
        )
        all_stats[name] = stats

    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Style vs Fact Discrimination")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Style Diff':<12} {'Fact Diff':<12} {'Subject Diff':<12}")
    print("-" * 51)
    for name in ["no_hess", "baseline", "mixed", "r_between"]:
        if name in all_stats and all_stats[name]:
            s = all_stats[name]
            style_diff = s.get("intra_style", 0) - s.get("inter_style", 0)
            fact_diff = s.get("intra_fact", 0) - s.get("inter_fact_same_subject", 0)
            subj_diff = s.get("intra_subject", 0) - s.get("inter_subject", 0)
            print(
                f"{name:<15} {style_diff:<12.4f} {fact_diff:<12.4f} {subj_diff:<12.4f}"
            )

    return all_stats


def main() -> None:
    """Main entry point for semantic experiments."""
    create_data()  # Skips if style datasets already exist
    dataset_paths = [
        "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf",
        "data/facts_dataset_pirate-Qwen3-8B-Base.hf",
        "data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf",
        "data/facts_dataset_pirate-Meta-Llama-3-8B.hf",
    ]

    final_dataset_path = "data/facts_dataset_reworded.hf"

    if not Path(final_dataset_path).exists():
        original = load_from_disk("data/facts_dataset.hf")
        if isinstance(original, DatasetDict):
            original = original["train"]

        merged_datasets: list[Dataset] = []

        for path in dataset_paths:
            ds = load_from_disk(path)
            if isinstance(ds, DatasetDict):
                ds = ds["train"]

            # Add back any dropped columns from original
            for col in original.column_names:
                if col not in ds.column_names:
                    # Align ds length with original by matching on "fact"
                    # Create a mapping from fact -> row
                    orig_map = {row["fact"]: row for row in original}  # type: ignore[index]

                    # Build list for restored column
                    restored_col = [orig_map[row["fact"]][col] for row in ds]  # type: ignore[index]

                    ds = ds.add_column(col, restored_col)

            merged_datasets.append(ds)

        final_dataset = concatenate_datasets(merged_datasets)
        final_dataset = final_dataset.shuffle(seed=42)

        final_dataset.save_to_disk(final_dataset_path)
        print(f"Merged dataset saved to: {final_dataset_path}")

    # Run the hessian comparison experiment
    run_hessian_comparison()


if __name__ == "__main__":
    main()
