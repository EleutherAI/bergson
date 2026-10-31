"""In-memory per-token attribution, the way the leaderboard scores tokens.

Demonstrates how to:
1. Reduce query gradients to a single vector via mean reduction
2. Score every training loss term by its rate of change along that query
   (``token_influence: output``, the mode of the per-token leaderboard in
   ``examples/gpt2_wikitext_leaderboard/tokens``)
3. Access token-level attribution scores

Row ``t`` of a document is the score of the loss on token ``t + 1``, the
output token influence of Grosse et al. (2023); a document's rows sum to
its gradient's dot product with the query. One forward-mode pass scores a
batch against the reduced query.

Usage:
    python examples/token_attribution.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

from bergson import (
    CollectorComputer,
    DataConfig,
    GradientCollector,
    GradientProcessor,
    IndexConfig,
    InMemoryCollector,
    PreprocessConfig,
    Scorer,
)
from bergson.data import allocate_batches, pad_and_tensor
from bergson.score.output_influence import output_token_influence, query_directions
from bergson.score.score_writer import InMemoryTokenScoreWriter
from bergson.utils.utils import get_device
from bergson.utils.worker_utils import setup_data_pipeline, setup_model_and_peft

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
    )

    # The fused attention kernels don't implement forward-mode derivatives
    model, target_modules = setup_model_and_peft(index_cfg, attn_implementation="eager")
    # Dropout would give the query its own random model
    model.eval()
    device = torch.device(get_device())
    processor = GradientProcessor()

    # Ensure partial_run_paths exist (CollectorComputer
    # saves total_processed.pt there)
    Path(index_cfg.partial_run_path).mkdir(parents=True, exist_ok=True)

    # Prepare training data
    train_ds, _ = setup_data_pipeline(index_cfg)
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
    )
    Path(query_cfg.partial_run_path).mkdir(parents=True, exist_ok=True)
    query_ds, _ = setup_data_pipeline(query_cfg)
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

    modules = list(query_collector.gradients)
    print(f"Reduced query to 1 vector across {len(modules)} modules")

    # Step 2: Create scorer with InMemoryTokenScoreWriter and map the query
    # onto the parameters it moves
    writer = InMemoryTokenScoreWriter(
        data=train_ds,
        num_scores=1,
    )
    scorer = Scorer(
        query_grads=query_collector.gradients,
        modules=modules,
        writer=writer,
        device=device,
        dtype=torch.float32,
        attribute_tokens=True,
    )
    directions = query_directions(
        model,
        scorer.query_grads_t,
        GradientCollector.discover_targets(
            model.base_model,
            target_modules,
            index_cfg.include_bias,
            index_cfg.filter_modules,
        ),
    )

    # Step 3: Score every training loss term along the query
    print("Scoring output token influence...")
    for indices in tqdm(train_batches, desc="scoring"):
        batch = train_ds[indices]
        x, y, _, collection_mask = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=device,
            sync_max_len=False,
        )
        rates, _ = output_token_influence(model, x, y, directions, index_cfg)
        # Row t of the token store is the loss on token t + 1
        rates = torch.cat([rates, rates.new_zeros(rates[:, :1].shape)], dim=1)
        scorer.write_scores(indices, rates[collection_mask])
    writer.flush()

    # Step 4: Access per-token scores
    print("\nPer-token attribution scores:")
    for i in range(len(train_ds)):
        scores_i = writer.scores[i]
        if scores_i.shape[0] == 0:
            continue
        print(
            f"  Example {i}: "
            f"{scores_i.shape[0]} tokens, "
            f"scores shape {tuple(scores_i.shape)}, "
            f"sum={scores_i.sum():.4e}, "
            f"max={scores_i.max():.4e}"
        )


if __name__ == "__main__":
    main()
