"""Ablation: compute PCA style subspace from semantic (Q&A) gradients.

Hypothesis: the PCA style subspace computed from full gradients may overlap
with semantic directions in the answer-only gradient space. Computing PCA
from semantic gradients should give a cleaner style subspace for the
semantic eval conditions.

Steps:
1. Add question/answer columns to pirate/shakespeare datasets
2. Build semantic gradient indices (gradients only from answer tokens)
3. Compute PCA style subspace from semantic gradient differences
4. Re-run semantic PCA conditions with the new subspace
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DatasetDict, load_dataset, load_from_disk

from examples.semantic.asymmetric import (
    AsymmetricConfig,
    AsymmetricMetrics,
    compute_asymmetric_metrics_with_pca,
)
from examples.semantic.data import HF_ANALYSIS_MODEL
from examples.semantic.preconditioners import compute_pca_style_subspace

BASE_PATH = Path("runs/asymmetric_style")
SEMANTIC_IDX_PATH = Path("runs/precond_comparison_semantic")
DAMPING = 0.1


def create_semantic_style_datasets() -> tuple[Path, Path]:
    """Add question/answer columns to pirate and shakespeare datasets."""
    # Load HF data for Q&A mapping
    hf_ds = load_dataset("EleutherAI/bergson-asymmetric-style")
    fact_to_qa: dict[str, tuple[str, str]] = {}
    for split in hf_ds:
        for row in hf_ds[split]:
            fact_to_qa[row["fact"]] = (row["question"], row["answer"])

    for style in ["pirate", "shakespeare"]:
        out_path = Path(f"data/facts_dataset_{style}-Qwen3-8B-Base-qa.hf")
        if out_path.exists():
            print(f"Semantic dataset already exists at {out_path}, skipping...")
            continue

        ds = load_from_disk(f"data/facts_dataset_{style}-Qwen3-8B-Base.hf")
        if isinstance(ds, DatasetDict):
            ds = ds["train"]

        def add_qa(example: dict) -> dict:
            q, a = fact_to_qa[example["fact"]]
            return {"question": q, "answer": a}

        ds = ds.map(add_qa)
        ds.save_to_disk(str(out_path))
        print(f"Saved semantic dataset to {out_path} ({len(ds)} rows)")
        print(f"  Columns: {ds.column_names}")
        print(f"  Example: question='{ds[0]['question']}' answer='{ds[0]['answer']}'")

    pirate_path = Path("data/facts_dataset_pirate-Qwen3-8B-Base-qa.hf")
    shakespeare_path = Path("data/facts_dataset_shakespeare-Qwen3-8B-Base-qa.hf")
    return pirate_path, shakespeare_path


def build_semantic_indices(
    pirate_data: Path,
    shakespeare_data: Path,
    analysis_model: str = HF_ANALYSIS_MODEL,
) -> tuple[Path, Path]:
    """Build gradient indices using question/answer columns (semantic only)."""
    SEMANTIC_IDX_PATH.mkdir(parents=True, exist_ok=True)

    for dataset_path, style_name in [
        (pirate_data, "pirate"),
        (shakespeare_data, "shakespeare"),
    ]:
        run_path = SEMANTIC_IDX_PATH / style_name
        if run_path.exists():
            print(f"Semantic index already exists at {run_path}, skipping...")
            continue

        print(f"Building semantic index for {style_name}...")
        cmd = [
            "bergson",
            "build",
            str(run_path),
            "--model",
            analysis_model,
            "--dataset",
            str(dataset_path),
            "--drop_columns",
            "False",
            "--prompt_column",
            "question",
            "--completion_column",
            "answer",
            "--fsdp",
            "--projection_dim",
            "16",
            "--token_batch_size",
            "6000",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"bergson build failed for {style_name}")

    pirate_idx = SEMANTIC_IDX_PATH / "pirate"
    shakespeare_idx = SEMANTIC_IDX_PATH / "shakespeare"
    return pirate_idx, shakespeare_idx


def main():
    print("=" * 70)
    print("SEMANTIC PCA ABLATION")
    print("=" * 70)
    print()
    print("Hypothesis: PCA style subspace computed from full gradients may")
    print("overlap with semantic directions in answer-only gradient space.")
    print("Computing PCA from semantic gradients should give a cleaner subspace.")
    print()

    # Step 1: Create Q&A-augmented datasets
    print("-" * 60)
    print("STEP 1: Creating semantic style datasets")
    print("-" * 60)
    pirate_data, shakespeare_data = create_semantic_style_datasets()

    # Step 2: Build semantic gradient indices
    print()
    print("-" * 60)
    print("STEP 2: Building semantic gradient indices")
    print("-" * 60)
    pirate_idx, shakespeare_idx = build_semantic_indices(pirate_data, shakespeare_data)

    # Step 3: Load eval facts to exclude (prevent data leakage)
    eval_ds = load_from_disk(str(BASE_PATH / "data" / "eval.hf"))
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]
    eval_facts_to_exclude: set[str] = set(eval_ds["fact"])

    # Step 4: Compute PCA subspaces from semantic gradients
    print()
    print("-" * 60)
    print("STEP 3: Computing PCA style subspaces from semantic gradients")
    print("-" * 60)

    config = AsymmetricConfig(hf_dataset="EleutherAI/bergson-asymmetric-style")

    k_values = [10, 100]
    preconditioners = [None, "index"]  # None = no precond, "index" = H_train

    all_metrics: dict[str, AsymmetricMetrics] = {}

    for k in k_values:
        print(f"\n--- Computing semantic style subspace for k={k} ---")
        semantic_subspace = compute_pca_style_subspace(
            pirate_idx,
            shakespeare_idx,
            SEMANTIC_IDX_PATH / "pca_subspace",
            top_k=k,
            exclude_facts=eval_facts_to_exclude,
        )

        for precond_name in preconditioners:
            precond_display = precond_name if precond_name else "no_precond"
            strategy_name = f"semantic_pca_k{k}_{precond_display}_semantic_basis"

            print(f"\n--- Strategy: {strategy_name} ---")
            metrics = compute_asymmetric_metrics_with_pca(
                config,
                BASE_PATH,
                semantic_subspace,
                top_k=k,
                preconditioner_name=precond_name,
                damping_factor=DAMPING,
                eval_prompt_column="question",
                eval_completion_column="answer",
            )
            print(f"  Top-1: {metrics.top1_semantic_accuracy:.2%}")
            print(f"  Top-5: {metrics.top5_semantic_recall:.2%}")
            print(f"  Style Leak: {metrics.top1_style_leakage:.2%}")
            all_metrics[strategy_name] = metrics

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: Full-gradient PCA vs Semantic-gradient PCA")
    print("=" * 70)

    # Load original results for comparison
    with open(BASE_PATH / "experiment_results.json") as f:
        orig_results = json.load(f)

    comparisons = [
        (
            "k=10, no precond",
            "semantic_pca_projection_k10",
            "semantic_pca_k10_no_precond_semantic_basis",
        ),
        (
            "k=10, H_train",
            "semantic_pca_k10_index",
            "semantic_pca_k10_index_semantic_basis",
        ),
        (
            "k=100, no precond",
            "semantic_pca_projection_k100",
            "semantic_pca_k100_no_precond_semantic_basis",
        ),
        (
            "k=100, H_train",
            "semantic_pca_k100_index",
            "semantic_pca_k100_index_semantic_basis",
        ),
    ]

    header = (
        f"{'Condition':<25} {'Full-grad PCA Top-1':<22} "
        f"{'Semantic PCA Top-1':<22} {'Full Leak':<12} {'Sem Leak':<12}"
    )
    print(header)
    print("-" * len(header))

    for label, orig_key, new_key in comparisons:
        orig = orig_results.get(orig_key, {})
        new = all_metrics.get(new_key)

        orig_top1 = orig.get("top1_semantic", 0)
        orig_leak = orig.get("top1_leak", 0)
        new_top1 = new.top1_semantic_accuracy if new else 0
        new_leak = new.top1_style_leakage if new else 0

        print(
            f"{label:<25} {orig_top1:<22.2%} {new_top1:<22.2%} "
            f"{orig_leak:<12.2%} {new_leak:<12.2%}"
        )


if __name__ == "__main__":
    main()
