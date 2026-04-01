import gc
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from simple_parsing import parse
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bergson.config import DataConfig
from bergson.data import load_gradient_dataset, load_scores, tokenize
from bergson.process_grads import get_trackstar_preconditioner, precondition_flat_grads
from bergson.utils.utils import assert_type


@dataclass
class FilterConfig:
    """Config for building the index and running the model/dataset pipeline."""

    filter: Literal["classification", "attribution", "trackstar", "loss", "random"] = (
        "attribution"
    )
    """Filter to apply to the training set before finetuning."""

    model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    """Name of the model to load."""

    dataset: str = "argilla/magpie-ultra-v0.1"
    """Dataset identifier to finetune on."""

    index_dataset: str = ""
    """Bergson index to use for attribution and loss filtering."""

    split: str = "train"

    query_dataset: str = ""
    """
    Use the mean of this dataset's gradients as the query for attribution
    filtering. If unspecified the query is calculated over the index dataset.
    """

    query_scores: bool = False
    """Use the top-scored dataset items for the attribution query."""

    precondition: bool = False
    """Whether to use preconditioner for attribution filtering."""

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    max_samples: int = 25_000
    """Maximum number of samples to use from the dataset. 0 for all."""

    num_examples: int = 10_000
    """Number of items to select from the training set after filtering."""

    prompt_column: str = ""
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = "messages"
    """Optional column in the dataset that contains the conversation."""

    batch_size: int = 512
    """Batch size for processing the dataset."""

    seed: int = 42
    """Seed for reproducibility."""

    lowest: bool = False
    """Select the lowest scores."""

    sample: bool = False
    """Filter by sampling from the dataset without replacement with
    probability proportional to the filtering criteria."""

    temperature: float = 0.1
    """Temperature for sampling, used to control the distribution of
    the sampling probabilities. Lower values make the distribution more
    uniform, while higher values make it more peaked."""

    num_epochs: int = 1
    """Number of epochs to train for."""

    hf_token: str | None = None
    """Hugging Face token to use for the dataset."""

    dry_run: bool = False
    """Whether to run the script in dry run mode."""

    revision: str | None = None
    """Revision of the model to use."""

    query_method: Literal["mean", "nearest"] = "mean"
    """Method to use for computing the query."""

    use_lora: bool = False
    """Use LoRA for finetuning instead of full SFT."""

    lora_rank: int = 16
    """LoRA rank (only used when use_lora=True)."""

    projection_dim: int = 16
    """Projection dimension for gradient index."""


def run_sft(
    cfg: FilterConfig,
    train: Dataset,
    eval: Dataset,
    output_dir: str,
    model_name_or_path: str | None = None,
) -> dict:
    """Run SFT. Uses HF Trainer which handles DDP automatically.
    Returns the final eval metrics."""
    if model_name_or_path is None:
        model_name_or_path = cfg.model

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        revision=cfg.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=8192)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    num_train_steps = (len(train) // 32) * cfg.num_epochs
    eval_steps = max(1, num_train_steps // 10)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=eval,
        args=SFTConfig(
            max_length=2048,
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            learning_rate=3e-4,
            num_train_epochs=cfg.num_epochs,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=100,
            save_total_limit=3,
            group_by_length=True,
            ddp_find_unused_parameters=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=cfg.seed,
        ),
    )

    if cfg.dry_run:
        print("Dry run mode, exiting...")
        return {}

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save eval metrics
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Free GPU memory before next step
    del trainer, model
    gc.collect()

    return metrics


def build_index(args: FilterConfig, index_path: str, model: str) -> None:
    """Build a bergson gradient index if it doesn't already exist."""
    if Path(index_path).exists():
        return

    split = args.split
    if args.max_samples:
        split = f"{split}[:{args.max_samples}]"

    cmd = [
        "bergson",
        "build",
        index_path,
        "--model",
        model,
        "--dataset",
        args.dataset,
        "--split",
        split,
        "--truncation",
        "--projection_dim",
        str(args.projection_dim),
        "--token_batch_size",
        "8192",
        "--precision",
        "auto",
        "--overwrite",
    ]
    if args.prompt_column:
        cmd += ["--prompt_column", args.prompt_column]
    if args.completion_column:
        cmd += ["--completion_column", args.completion_column]
    if args.conversation_column:
        cmd += ["--conversation_column", args.conversation_column]

    print(f"Building index: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"bergson build failed with exit code {result.returncode}")


def run_trackstar(
    args: FilterConfig, trackstar_path: str, model: str, num_gpus: int = 1
) -> None:
    """Run the bergson trackstar pipeline to score the dataset."""
    scores_path = Path(trackstar_path) / "scores"
    if scores_path.exists():
        print(f"Trackstar scores already exist at {scores_path}, skipping.")
        return

    split = args.split
    if args.max_samples:
        split = f"{split}[:{args.max_samples}]"

    cmd = [
        "bergson",
        "trackstar",
        trackstar_path,
        "--model",
        model,
        "--normalizer",
        "adafactor",
        "--stats_sample_size",
        "10000",
        # Value dataset (the training data to score)
        "--data.dataset",
        args.dataset,
        "--data.split",
        split,
        "--data.truncation",
        # Query dataset (same data — self-attribution)
        "--query.dataset",
        args.dataset,
        "--query.split",
        split,
        "--query.truncation",
        # Score settings
        "--unit_normalize",
        "--aggregation",
        "mean",
        "--normalize_aggregated_grad",
        "--projection_dim",
        str(args.projection_dim),
        "--token_batch_size",
        "8192",
        "--nproc_per_node",
        str(num_gpus),
        "--overwrite",
    ]
    # PEFT models need explicit tokenizer since adapter dir has no tokenizer config
    if args.use_lora:
        cmd += ["--tokenizer", args.model]
    if args.conversation_column:
        cmd += ["--data.conversation_column", args.conversation_column]
        cmd += ["--query.conversation_column", args.conversation_column]
    if args.prompt_column:
        cmd += ["--data.prompt_column", args.prompt_column]
        cmd += ["--query.prompt_column", args.prompt_column]
    if args.completion_column:
        cmd += ["--data.completion_column", args.completion_column]
        cmd += ["--query.completion_column", args.completion_column]

    print(f"Running trackstar: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"bergson trackstar failed with exit code {result.returncode}"
        )


def sft_full(args: FilterConfig, output_dir: str) -> str:
    """SFT on the full dataset. Returns the checkpoint path.

    This is step 1 of the DA workflow: finetune on the data you want to
    attribute so that the gradients are meaningful.
    """
    # Check for actual model files, not just directory existence
    output_path = Path(output_dir)
    has_model = (output_path / "config.yaml").exists() or (
        output_path / "adapter_config.yaml"
    ).exists()
    if has_model:
        print(f"SFT checkpoint already exists at {output_dir}, skipping.")
        return output_dir

    dataset = assert_type(Dataset, load_dataset(args.dataset, split=args.split))
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)
    data_config = DataConfig(
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        conversation_column=args.conversation_column,
    )
    train_ds = train_ds.map(
        tokenize, batched=True, fn_kwargs=dict(args=data_config, tokenizer=tokenizer)
    )
    eval_ds = eval_ds.map(
        tokenize, batched=True, fn_kwargs=dict(args=data_config, tokenizer=tokenizer)
    )

    print(f"SFT on full dataset ({len(train_ds)} examples)...")
    run_sft(args, train_ds, eval_ds, output_dir)
    print(f"SFT checkpoint saved to {output_dir}")

    return output_dir


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _get_attribution_indices(
    args: FilterConfig,
    train: Dataset,
    run_name: str,
    query_method: Literal["mean", "nearest"] = "mean",
) -> Tensor:
    """Score gradient dataset and return selected indices."""
    if args.query_scores:
        query_dataset = train.filter(lambda x: x["quality"] == "excellent")
    elif args.query_dataset:
        query_dataset = load_gradient_dataset(
            Path(args.query_dataset), structured=False
        ).with_format("torch")
    else:
        query_dataset = train

    # Compute the mean of the normalized gradients in the query index
    if query_method == "mean":
        acc = {"sum": torch.zeros_like(query_dataset[0]["gradients"], device="cuda")}

        def sum_(col):
            acc["sum"] += col.cuda().sum(0)

        # RAM usage climbs here; it's intentionally only evicted under pressure
        # Do not use num_proc because we are accumulating in a single variable
        # nproc solution must use reduce as in
        # https://colab.research.google.com/drive/1jCLv31Y4cDfqD0lhO0AnqEv3Or-LLvWe?usp=sharing
        query_dataset.map(
            sum_, input_columns="gradients", batched=True, batch_size=args.batch_size
        )

        query = acc["sum"] / len(query_dataset)
    elif query_method == "nearest":
        query = assert_type(Tensor, query_dataset["gradients"]).cuda()

    if args.precondition:
        index_ds_path = Path(args.index_dataset)
        preconditioner_path = (
            args.query_dataset if args.query_dataset else args.index_dataset
        )
        h_inv = get_trackstar_preconditioner(
            preconditioner_path, device=torch.device("cuda"), power=-1
        )

        # Get ordered module names from info.json
        with open(index_ds_path / "info.json") as f:
            ordered_modules = json.load(f)["dtype"]["names"]

        query = precondition_flat_grads(
            query.unsqueeze(0), h_inv, ordered_modules
        ).squeeze(0)
    else:
        h_inv = {}
        ordered_modules = []

    query /= query.norm()

    del query_dataset

    # Score the training set
    acc = {"scores": []}

    def score(batch):
        gradients_batch = batch.cuda()

        if h_inv:
            gradients_batch = precondition_flat_grads(
                gradients_batch, h_inv, ordered_modules
            )

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query

        acc["scores"].append(batch_scores)

    def score_nearest(batch):
        gradients_batch = batch.cuda()

        if h_inv:
            gradients_batch = precondition_flat_grads(
                gradients_batch, h_inv, ordered_modules
            )

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query.T

        # Take the maximum batch score for each item in the batch
        # (query has multiple rows)
        batch_scores = batch_scores.max(dim=-1).values

        acc["scores"].append(batch_scores)

    score_fn = score_nearest if query_method == "nearest" else score
    train.map(
        score_fn, input_columns="gradients", batched=True, batch_size=args.batch_size
    )
    importance_scores = torch.cat(acc["scores"], dim=0).cuda()

    print(
        f"Score stats: min={importance_scores.min():.4f}, "
        f"max={importance_scores.max():.4f}, "
        f"mean={importance_scores.mean():.4f}, "
        f"std={importance_scores.std():.4f}"
    )

    print("Saving importance scores to disk.")
    os.makedirs(f"examples/runs/{run_name}", exist_ok=True)
    torch.save(importance_scores, f"examples/runs/{run_name}/importance_scores.pt")

    if args.sample:
        probs = torch.softmax(importance_scores / args.temperature, dim=0)
        selected_indices = torch.multinomial(
            probs, args.num_examples, replacement=False
        )
    else:
        # Select the top-k items
        sorted_scores = torch.argsort(importance_scores)
        selected_indices = (
            sorted_scores[: args.num_examples]
            if args.lowest
            else sorted_scores[-args.num_examples :]
        )

    return selected_indices


def main(
    args: FilterConfig,
):
    set_seeds(args.seed)
    print("Running")

    if args.name is None:
        run_name = (
            f"{args.model.split('/')[-1]}-{args.dataset.split('/')[-1]}-{args.filter}"
            f"{'-lora' if args.use_lora else ''}"
            f"{'-lowest' if args.lowest else ''}"
            f"-n={args.num_examples}"
            f"-s={args.seed}"
        )
    else:
        run_name = args.name

    # Always load the original text dataset for training.
    # Don't shuffle here — order must match the gradient index built by bergson.
    orig_dataset = assert_type(Dataset, load_dataset(args.dataset, split=args.split))
    if args.max_samples:
        orig_dataset = orig_dataset.select(
            range(min(args.max_samples, len(orig_dataset)))
        )

    # Add original index column so we can map back after train_test_split shuffles
    orig_dataset = orig_dataset.add_column("_orig_idx", list(range(len(orig_dataset))))

    # Split original dataset (same seed ensures consistent eval set)
    print("Splitting...")
    orig_split = orig_dataset.train_test_split(test_size=0.05, seed=args.seed)
    orig_train, orig_eval = orig_split["train"], orig_split["test"]

    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]

    lora_suffix = "_lora" if args.use_lora else ""
    proj_suffix = f"_p{args.projection_dim}" if args.projection_dim != 16 else ""

    if args.filter in ("attribution", "loss"):
        # Step 1: SFT on the full dataset so gradients are meaningful
        sft_dir = f"examples/runs/{model_name}_{dataset_name}_sft{lora_suffix}"
        sft_model_path = sft_full(args, sft_dir)

        # Step 2: Build gradient index using the finetuned checkpoint
        if not args.index_dataset:
            args.index_dataset = (
                f"examples/runs/{model_name}_{dataset_name}"
                f"_index{lora_suffix}{proj_suffix}"
            )

        build_index(args, args.index_dataset, model=sft_model_path)
        grad_dataset = load_gradient_dataset(Path(args.index_dataset), structured=False)

        # Split gradient dataset the same way
        grad_split = grad_dataset.train_test_split(test_size=0.05, seed=args.seed)
        grad_train = grad_split["train"]
        grad_train.set_format("torch")

    elif args.filter == "trackstar":
        # Step 1: SFT on the full dataset so gradients are meaningful
        sft_dir = f"examples/runs/{model_name}_{dataset_name}_sft{lora_suffix}"
        sft_model_path = sft_full(args, sft_dir)

        # Step 2: Run trackstar pipeline for scoring
        trackstar_path = (
            f"examples/runs/{model_name}_{dataset_name}"
            f"_trackstar{lora_suffix}{proj_suffix}"
        )
        run_trackstar(args, trackstar_path, model=sft_model_path)

    # Step 3: Filter the training set
    print("Filtering...")
    if args.num_examples == 0:
        train = orig_train
    elif args.filter == "trackstar":
        scores_path = Path(
            f"examples/runs/{model_name}_{dataset_name}_trackstar{lora_suffix}{proj_suffix}/scores"
        )
        scores = load_scores(scores_path)
        # Scores are in original dataset order (before train/test split).
        # Use _orig_idx to map train split positions to original indices.
        all_scores = torch.from_numpy(scores[:].flatten().astype("float32"))
        train_orig_indices = torch.tensor(orig_train["_orig_idx"])
        train_scores = all_scores[train_orig_indices]

        print(
            f"Trackstar score stats: min={train_scores.min():.4f}, "
            f"max={train_scores.max():.4f}, "
            f"mean={train_scores.mean():.4f}, "
            f"std={train_scores.std():.4f}"
        )

        sorted_local = torch.argsort(train_scores)
        selected_indices = (
            sorted_local[: args.num_examples]
            if args.lowest
            else sorted_local[-args.num_examples :]
        )
        train = orig_train.select(selected_indices)
    elif args.filter == "attribution":
        selected_indices = _get_attribution_indices(
            args, grad_train, run_name, query_method=args.query_method
        )
        train = orig_train.select(selected_indices)
    elif args.filter == "classification":
        if "score" in orig_train.column_names:
            train = orig_train.sort("score", reverse=not args.lowest)
            train = train.select(range(min(args.num_examples, len(train))))
        else:
            ranks = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "very poor": 0}

            def add_rank(ex):
                q = ex.get("quality")
                return {"_q": ranks.get(q, -1)}

            train = (
                orig_train.map(add_rank)
                .filter(lambda x: x["_q"] >= 0)
                .sort("_q", reverse=not args.lowest)
                .select(range(min(args.num_examples, len(orig_train))))
                .remove_columns("_q")
            )
    elif args.filter == "loss":
        grad_train_loss = grad_train.map(
            lambda x: {"loss_val": x["loss"].item()},
        )
        sorted_scores = torch.argsort(torch.tensor(grad_train_loss["loss_val"]))
        selected_indices = (
            sorted_scores[: args.num_examples]
            if args.lowest
            else sorted_scores[-args.num_examples :]
        )
        train = orig_train.select(selected_indices)
    elif args.filter == "random":
        train = orig_train.select(range(min(args.num_examples, len(orig_train))))
    else:
        raise ValueError(f"Invalid filter: {args.filter}")

    eval_ds = orig_eval

    # Remove internal index column before training
    if "_orig_idx" in train.column_names:
        train = train.remove_columns("_orig_idx")
    if "_orig_idx" in eval_ds.column_names:
        eval_ds = eval_ds.remove_columns("_orig_idx")

    # Step 4: SFT from the base model on the filtered subset
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)
    data_config = DataConfig(
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        conversation_column=args.conversation_column,
    )

    train = train.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )
    eval_ds = eval_ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )

    print(f"Training on {len(train)} examples, evaluating on {len(eval_ds)} examples.")
    metrics = run_sft(args, train, eval_ds, f"examples/runs/{run_name}")

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Filter: {args.filter}")
    print(f"Num training examples: {len(train)}")
    if metrics:
        print(f"Final eval loss: {metrics.get('eval_loss', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse(FilterConfig)

    main(args)
