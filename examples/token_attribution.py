"""In-memory per-token gradient attribution.

Demonstrates how to:
1. Reduce query gradients to a single vector via mean reduction
2. Score training per-token gradients against the reduced query
3. Access token-level attribution scores

Usage:
    python examples/token_attribution.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset

from bergson import (
    CollectorComputer,
    DataConfig,
    GradientProcessor,
    IndexConfig,
    InMemoryCollector,
    PreprocessConfig,
    Scorer,
)
from bergson.data import allocate_batches
from bergson.score.score_writer import (
    InMemoryTokenScoreWriter,
)
from bergson.utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)

MODEL = "EleutherAI/pythia-14m"
TRAIN_SPLIT = "train[:20]"
QUERY_SPLIT = "train[20:22]"


def main():
    index_cfg = IndexConfig(
        run_path="/tmp/bergson_token_attr",
        model=MODEL,
        data=DataConfig(
            truncation=True,
            split=TRAIN_SPLIT,
        ),
        precision="fp32",
        token_batch_size=2048,
        attribute_tokens=True,
        skip_preconditioners=True,
    )

    # Load model
    model, target_modules = setup_model_and_peft(index_cfg)
    processor = GradientProcessor()

    # Ensure partial_run_paths exist (CollectorComputer
    # saves total_processed.pt there)
    Path(index_cfg.partial_run_path).mkdir(parents=True, exist_ok=True)

    # Prepare training data
    train_ds = setup_data_pipeline(index_cfg)
    assert isinstance(train_ds, Dataset)
    train_batches = allocate_batches(
        train_ds["length"][:],
        index_cfg.token_batch_size,
    )
    print(
        f"Training set: {len(train_ds)} examples, " f"{sum(train_ds['length'])} tokens"
    )

    # Prepare query data (sequence-level for reduce)
    query_cfg = IndexConfig(
        run_path="/tmp/bergson_token_attr_query",
        model=MODEL,
        data=DataConfig(
            truncation=True,
            split=QUERY_SPLIT,
        ),
        precision="fp32",
        token_batch_size=2048,
        skip_preconditioners=True,
    )
    Path(query_cfg.partial_run_path).mkdir(parents=True, exist_ok=True)
    query_ds = setup_data_pipeline(query_cfg)
    assert isinstance(query_ds, Dataset)
    query_batches = allocate_batches(
        query_ds["length"][:],
        query_cfg.token_batch_size,
    )
    print(f"Query set: {len(query_ds)} examples, " f"{sum(query_ds['length'])} tokens")

    # Step 1: Reduce query gradients to a single vector
    print("\nCollecting query gradients (reduce)...")
    query_collector = InMemoryCollector(
        model=model.base_model,
        processor=processor,
        data=query_ds,
        cfg=query_cfg,
        target_modules=target_modules,
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )
    query_computer = CollectorComputer(
        model=model,
        data=query_ds,
        collector=query_collector,
        batches=query_batches,
        cfg=query_cfg,
    )
    query_computer.run_with_collector_hooks(desc="query gradients")

    print(f"Reduced query to 1 vector across {len(target_modules)} modules")

    # Step 2: Create scorer with InMemoryTokenScoreWriter
    writer = InMemoryTokenScoreWriter(
        data=train_ds,
        num_scores=1,
    )
    scorer = Scorer(
        query_grads=query_collector.gradients,
        modules=list(target_modules),
        writer=writer,
        device=torch.device("cuda"),
        dtype=torch.float32,
        attribute_tokens=True,
    )

    # Step 3: Score training data per-token
    print("Scoring training gradients...")
    score_collector = InMemoryCollector(
        model=model.base_model,
        processor=processor,
        data=train_ds,
        cfg=index_cfg,
        target_modules=target_modules,
        scorer=scorer,
    )
    score_computer = CollectorComputer(
        model=model,
        data=train_ds,
        collector=score_collector,
        batches=train_batches,
        cfg=index_cfg,
    )
    score_computer.run_with_collector_hooks(desc="scoring")

    # Step 4: Access per-token scores
    print("\nPer-token attribution scores:")
    for i in range(len(train_ds)):
        scores_i = score_collector.scores[i]
        if scores_i.shape[0] == 0:
            continue
        print(
            f"  Example {i}: "
            f"{scores_i.shape[0]} tokens, "
            f"scores shape {tuple(scores_i.shape)}, "
            f"mean={scores_i.mean():.4e}, "
            f"max={scores_i.max():.4e}"
        )


if __name__ == "__main__":
    main()
