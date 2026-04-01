#!/usr/bin/env python3
"""
Pretrain a two-layer transformer and try to identify the formation of induction heads
from the influence functions with respect to simple induction head completion gradients.

This script:
1. Creates a 2-layer attention-only transformer
2. Trains using the HF Trainer with the Bergson callback to collect gradients
3. Builds a static query Bergson index using synthetic induction head data
4. Plots the influence of the training examples on the induction heads
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from bergson import CollectorComputer, InMemoryCollector
from bergson.config import FaissConfig, IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.gradients import GradientProcessor
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.query.attributor import Attributor
from bergson.utils import assert_type
from examples.induction.attn_only_transformer import AttnOnlyForCausalLM
from examples.induction.plot import plot_influence_scores
from examples.induction.setup_utils import (
    HEAD_CFGS,
    check_logins,
    create_induction_ds,
    create_model,
    load_data,
    upload_to_hub,
)


def train_with_gradients(
    tokenizer,
    eval_dataset,
    output_dir: str,
    projection_dim: int,
    dataset_name: str,
    special_pos_embed: bool,
    debug: bool = False,
    wandb: bool = True,
):
    """Train model with Bergson gradient collection and save to output_dir."""
    if debug:
        train_dataset, _ = load_data(tokenizer, name=dataset_name, N=20_000)
    else:
        train_dataset, _ = load_data(tokenizer, name=dataset_name)

    model = create_model(tokenizer, special_pos_embed=special_pos_embed)

    pad_id = -100

    def compute_metrics(eval_preds):
        # predictions: (B, T, V)
        # label_ids: with your collator, this equals input_ids: (B, T)
        preds = eval_preds.predictions
        input_ids = eval_preds.label_ids

        correct = 0
        total = 0
        # for each sequence, evaluate the final next-token prediction
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            # last non-pad index j
            non_pad = np.where(seq != pad_id)[0]
            if len(non_pad) == 0:
                continue
            j = non_pad[-1]
            if j == 0:
                continue  # nothing to predict
            pred_tok = preds[i, j - 1].argmax(-1)
            tgt_tok = seq[j]
            correct += int(pred_tok == tgt_tok)
            total += 1

        # avoid div-by-zero
        acc = (correct / total) if total > 0 else 0.0
        return {"accuracy": acc}

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        warmup_steps=1000,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10_000,
        # save_total_limit=3,
        report_to="wandb" if wandb else None,
        run_name="2-layer-transformer-SmolLM2-corpus",
        seed=42,
        fp16=False,
        dataloader_drop_last=False,
    )

    bergson_callback = GradientCollectorCallback(
        path=Path(f"{output_dir}/gradients"),
        attention_cfgs=HEAD_CFGS,
        projection_dim=projection_dim,
        dtype=np.float32,
        accumulate_grads=False,
        track_order=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[bergson_callback],
        compute_metrics=compute_metrics,
    )

    # Prepare for gradient collection
    trainer = prepare_for_gradient_collection(trainer)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    upload_to_hub(model, tokenizer)


def reduce_query_gradients(
    model,
    induction_dataset,
    projection_dim,
    output_dir: str,
):
    """Reduce induction head gradients to a mean gradient vector in memory."""
    query_path = f"{output_dir}/query"
    Path(query_path + ".part").mkdir(parents=True, exist_ok=True)
    cfg = IndexConfig(
        run_path=query_path,
        projection_dim=projection_dim,
        skip_preconditioners=True,
    )
    processor = GradientProcessor(
        {},
        projection_dim=projection_dim or None,
    )

    collector = InMemoryCollector(
        model=model.base_model,
        data=induction_dataset,
        cfg=cfg,
        processor=processor,
        attention_cfgs=HEAD_CFGS,
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )

    doc_lengths = [len(ids) for ids in induction_dataset["input_ids"]]
    batches = allocate_batches(doc_lengths, cfg.token_batch_size)

    computer = CollectorComputer(
        model=model,
        data=induction_dataset,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="Reducing induction head gradients")

    return collector.gradients


def main(args):
    check_logins()

    dataset_name = "EleutherAI/SmolLM2-135M-10B"
    output_dir = "runs/two_layer_transformer"
    device = torch.device("cuda")

    print(
        "Starting 2-layer transformer pretraining with Bergson gradient collection..."
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    induction_dataset = create_induction_ds(tokenizer, seed=args.seed, num_prompts=100)

    model_path = Path(output_dir) / "model.safetensors"
    if not model_path.exists() or args.overwrite:
        train_with_gradients(
            tokenizer,
            eval_dataset=induction_dataset,
            output_dir=output_dir,
            projection_dim=args.projection_dim,
            dataset_name=dataset_name,
            special_pos_embed=not args.no_special_pos_embed,
            debug=args.debug,
            wandb=False,
        )

    # Reduce induction head gradients to a mean query vector
    model = AttnOnlyForCausalLM.from_pretrained(output_dir).to(device)
    mean_module_induction_gradients = reduce_query_gradients(
        model,
        induction_dataset,
        args.projection_dim,
        output_dir,
    )
    del model

    # Load parquet table containing training order
    training_order_ds = assert_type(
        Dataset, load_from_disk(str(Path(output_dir) / "gradients" / "order.hf"))
    )
    training_order = assert_type(pd.DataFrame, training_order_ds.to_pandas())

    # Calculate the mean query gradients' inner products with the training gradients
    attributor = Attributor(
        str(Path(output_dir) / "gradients" / "train" / "epoch_0"),
        device="cpu",
        unit_norm=args.unit_norm,
        dtype=torch.float32,
        faiss_cfg=FaissConfig(
            mmap_index=True, index_factory="IVF1,SQfp16", num_shards=10
        ),
    )

    # Ordered from largest to smallest like (3 2 1 ...)
    inner_products, indices = attributor.search(mean_module_induction_gradients, k=None)
    # Restore original order
    inner_products = torch.gather(inner_products, -1, indices.argsort(dim=-1))

    data = []
    for i, score in enumerate(inner_products.squeeze()):
        training_metadata = training_order[
            (training_order["_idx"] == i) & (training_order["epoch"] == 0)
        ]
        if len(training_metadata) != 1:
            continue

        for row in training_metadata.itertuples(index=False):
            data.append(
                {
                    "global_step": row[
                        training_metadata.columns.get_loc("global_step")
                    ],
                    "index": i,
                    "score": score.item(),
                }
            )
    data = pd.DataFrame(data)

    plot_influence_scores(data, args.unit_norm)

    # Test whether later training steps have higher influence scores
    from scipy.stats import spearmanr

    filtered = data[data["global_step"] > 100]
    corr, pvalue = spearmanr(filtered["global_step"], filtered["score"])
    print(
        f"Spearman correlation (step vs score, steps > 100): "
        f"r={corr:.4f}, p={pvalue:.2e}"
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--unit_norm", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_special_pos_embed", action="store_false")
    args = parser.parse_args()
    main(args)
