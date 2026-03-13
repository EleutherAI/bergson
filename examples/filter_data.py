"""Launch this script with torchrun."""

import gc
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from simple_parsing import parse
from torch import Tensor
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bergson.config import DataConfig
from bergson.data import load_gradient_dataset, load_scores, tokenize
from bergson.utils.utils import assert_type

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class FilterConfig:
    """Config for building the index and running the model/dataset pipeline."""

    filter: Literal["classification", "attribution", "trackstar", "loss", "random"] = (
        "trackstar"
    )
    """Filter to apply to the training set before finetuning."""

    model: str = "Qwen/Qwen2.5-1.5B"
    """Name of the model to load."""

    dataset: str = "sander-wood/irishman"
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

    max_samples: int = 50_000
    """Maximum number of samples to use from the dataset. 0 for all."""

    num_examples: int = 5_000
    """Number of items to select from the training set after filtering."""

    prompt_column: str = "abc notation"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    map_batch_size: int = 512
    """Batch size for processing the dataset."""

    seed: int = 42
    """Seed for reproducibility."""

    lowest: bool = False
    """Select the lowest scores."""

    ordered: bool = True
    """Replay the original full-dataset shuffle order during the filtered
    retrain. Selected items appear in the same sequence as the original SFT
    run, with removed items simply skipped."""

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

    overwrite: bool = False
    """Overwrite existing trackstar scores."""

    revision: str | None = None
    """Revision of the model to use."""

    query_method: Literal["mean", "nearest"] = "mean"
    """Method to use for computing the query."""

    use_lora: bool = True
    """Use LoRA for finetuning instead of full SFT."""

    lora_rank: int = 64
    """LoRA rank (only used when use_lora=True)."""

    projection_dim: int = 16
    """Projection dimension for gradient index."""

    test_size: float = 0.05

    tag: str = ""

    pdbs: int = 1
    "Per-device batch size"

    learning_rate: float = 5e-5

    precision: str = "fp32"

    subset: str = "default"


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class OrderedFilterSampler(Sampler[int]):
    """Replays the full-dataset shuffle, yielding only selected items.

    Generates ``randperm(full_dataset_size)`` each epoch using a seeded
    generator, then yields only positions that correspond to items kept
    after filtering.  This ensures the filtered retrain sees items in the
    same order the original full-dataset SFT run would have used, with
    removed items simply skipped.

    Each call to ``__iter__`` advances the generator state, so multi-epoch
    training produces different (but deterministic) orderings per epoch,
    matching the behaviour of ``RandomSampler``.
    """

    def __init__(
        self,
        full_dataset_size: int,
        orig_to_pos: dict[int, int],
        seed: int,
    ):
        self.full_size = full_dataset_size
        self.orig_to_pos = orig_to_pos
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        perm = torch.randperm(self.full_size, generator=self.generator)
        for idx in perm.tolist():
            if idx in self.orig_to_pos:
                yield self.orig_to_pos[idx]

    def __len__(self) -> int:
        return len(self.orig_to_pos)


def run_sft(
    cfg: FilterConfig,
    train: Dataset,
    eval_ds: Dataset,
    output_path: Path,
    model_name: str | None = None,
    run_name: str | None = None,
    sampler: Sampler[int] | None = None,
) -> dict:
    """SFT with HF Trainer, which handles DDP automatically.
    Returns the final eval metrics."""
    model_name = model_name or cfg.model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        revision=cfg.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=8192)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)  # type: ignore
        model.print_trainable_parameters()  # type: ignore

    effective_batch_size = 128
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_acc_steps = effective_batch_size // world_size // cfg.pdbs
    num_train_steps = (len(train) // effective_batch_size) * cfg.num_epochs
    eval_steps = max(1, num_train_steps // 5)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=eval_ds,
        args=SFTConfig(
            # train_sampling_strategy="group_by_length",
            max_length=2048,
            output_dir=str(output_path),
            per_device_train_batch_size=cfg.pdbs,
            per_device_eval_batch_size=cfg.pdbs,
            gradient_accumulation_steps=grad_acc_steps,
            # gradient_checkpointing=True,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_epochs,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False,
            report_to="wandb",
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=cfg.seed,
            run_name=run_name,
        ),
    )

    # Override the Trainer's default RandomSampler to replay the
    # full-dataset shuffle order with unselected items removed.
    if sampler is not None:
        trainer._get_train_sampler = lambda _ds=None: sampler  # type: ignore[assignment]

    if cfg.dry_run:
        print("Dry run mode, exiting...")
        return {}

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)

    # Save eval metrics
    with open(output_path / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Free GPU memory before next step
    del trainer, model
    gc.collect()

    return metrics


_TORCHRUN_ENV_VARS = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
}


def _file_barrier(
    sentinel_dir: Path,
    run_id: str,
    local_rank: int,
    world_size: int,
):
    """File-based barrier with cleanup.

    Protocol:
    1. Rank 0 creates the sentinel after finishing its subprocess.
    2. Other ranks poll until the sentinel exists.
    3. Each rank touches an ack file.
    4. Rank 0 waits for all acks, then cleans up everything.
    """
    sentinel = sentinel_dir / f".barrier_{run_id}"
    ack = sentinel_dir / f".ack_{run_id}_{local_rank}"

    if local_rank == 0:
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
    else:
        while not sentinel.exists():
            time.sleep(1)

    ack.touch()

    if local_rank == 0:
        acks = [sentinel_dir / f".ack_{run_id}_{r}" for r in range(world_size)]
        while not all(p.exists() for p in acks):
            time.sleep(0.5)
        sentinel.unlink(missing_ok=True)
        for p in acks:
            p.unlink(missing_ok=True)


def _clean_dist_env() -> dict[str, str]:
    """Return os.environ without torchrun distributed variables.

    Subprocesses that manage their own distributed setup (bergson build,
    bergson trackstar) must not inherit the parent torchrun's env vars.
    """
    return {k: v for k, v in os.environ.items() if k not in _TORCHRUN_ENV_VARS}


def build_index(
    cfg: FilterConfig, index_path: Path, train_split: str, model: str, rank: int = 0
) -> None:
    """Build a bergson gradient index if it doesn't already exist."""
    if index_path.exists():
        return

    cmd = [
        "bergson",
        "build",
        str(index_path),
        "--model",
        model,
        "--dataset",
        cfg.dataset,
        "--split",
        train_split,
        "--truncation",
        "--projection_dim",
        str(cfg.projection_dim),
        "--token_batch_size",
        "4096",
        "--precision",
        "auto",
        "--overwrite",
    ]
    if cfg.subset:
        cmd += ["--subset", cfg.subset]
    if cfg.prompt_column:
        cmd += ["--prompt_column", cfg.prompt_column]
    if cfg.completion_column:
        cmd += ["--completion_column", cfg.completion_column]
    if cfg.conversation_column:
        cmd += ["--conversation_column", cfg.conversation_column]

    print(f"Building index: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=_clean_dist_env())
    if result.returncode != 0:
        raise RuntimeError(f"bergson build failed with exit code {result.returncode}")


def collect_trackstar_scores(
    cfg: FilterConfig,
    trackstar_path: Path,
    train_split: str,
    eval_split: str,
    model: str,
) -> None:
    """Run the bergson trackstar pipeline to score the dataset."""
    scores_path = trackstar_path / "scores"
    if scores_path.exists() and not cfg.overwrite:
        print(f"Trackstar scores already exist at {scores_path}, skipping.")
        return

    cmd = [
        "bergson",
        "trackstar",
        str(trackstar_path),
        "--model",
        model,
        "--normalizer",
        "adafactor",
        "--stats_sample_size",
        "10000",
        # Value dataset (the training data to score)
        "--data.dataset",
        cfg.dataset,
        "--data.split",
        train_split,
        "--data.truncation",
        "--query.dataset",
        cfg.dataset,
        "--query.split",
        eval_split,
        "--query.truncation",
        # Score settings
        "--unit_normalize",
        "--aggregation",
        "mean",
        "--normalize_aggregated_grad",
        "--projection_dim",
        str(cfg.projection_dim),
        "--token_batch_size",
        "4096",
        "--stats_token_batch_size",
        "4096",
        "--nproc_per_node",
        str(torch.cuda.device_count()),
        "--overwrite",
        "--index_cfg.precision",
        cfg.precision,
    ]
    if cfg.subset:
        cmd += ["--data.subset", cfg.subset]
        cmd += ["--query.subset", cfg.subset]
    # PEFT models need explicit tokenizer since adapter dir has no tokenizer config
    if cfg.use_lora:
        cmd += ["--tokenizer", cfg.model]
    if cfg.conversation_column:
        cmd += ["--data.conversation_column", cfg.conversation_column]
        cmd += ["--query.conversation_column", cfg.conversation_column]
    if cfg.prompt_column:
        cmd += ["--data.prompt_column", cfg.prompt_column]
        cmd += ["--query.prompt_column", cfg.prompt_column]
    if cfg.completion_column:
        cmd += ["--data.completion_column", cfg.completion_column]
        cmd += ["--query.completion_column", cfg.completion_column]

    print(f"Running trackstar: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=_clean_dist_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"bergson trackstar failed with exit code {result.returncode}"
        )


def sft_full(
    ds,
    eval_ds,
    cfg: FilterConfig,
    tokenizer,
    data_config,
    output_path: Path,
    initial_sft_run_name: str,
) -> Path:
    """SFT on the full dataset. Returns the checkpoint path.

    This is a step in the DA workflow: fine-tune on the data you want to
    attribute so that the gradients are meaningful.
    """
    # Check for actual model files, not just directory existence
    has_model = (output_path / "config.json").exists() or (
        output_path / "adapter_config.json"
    ).exists()
    if has_model:
        print(f"SFT checkpoint already exists at {output_path}, skipping.")
        return output_path

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer, max_length=2048),
    )
    eval_ds = eval_ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer, max_length=2048),
    )

    print(f"SFT on ({len(ds)} examples)...")
    metrics = run_sft(
        cfg,
        ds,
        eval_ds,
        output_path,
        run_name=initial_sft_run_name,
    )
    print(f"SFT checkpoint saved to {output_path}")
    print(f"\n{'='*60}")
    if metrics:
        print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
        print(f"Final eval loss: {metrics.get('eval_loss', 'N/A')}")
    print(f"{'='*60}\n")

    return output_path


def _get_attribution_indices(
    cfg: FilterConfig, train: Dataset, run_name: str
) -> Tensor:
    """Score gradient dataset using Hessian-free cosine similarity
    scores and return selected indices."""
    if cfg.query_scores:
        query_dataset = train.filter(lambda x: x["quality"] == "excellent")
    elif cfg.query_dataset:
        query_dataset = load_gradient_dataset(
            Path(cfg.query_dataset), structured=False
        ).with_format("torch")
    else:
        query_dataset = train

    # Compute the mean of the normalized gradients in the query index
    if cfg.query_method == "mean":
        acc = {"sum": torch.zeros_like(query_dataset[0]["gradients"], device="cuda")}

        def sum_(col):
            acc["sum"] += col.cuda().sum(0)

        # RAM usage climbs here; it's intentionally only evicted under pressure
        # Do not use num_proc because we are accumulating in a single variable
        # nproc solution must use reduce as in
        # https://colab.research.google.com/drive/1jCLv31Y4cDfqD0lhO0AnqEv3Or-LLvWe?usp=sharing
        query_dataset.map(
            sum_,
            input_columns="gradients",
            batched=True,
            batch_size=cfg.map_batch_size,
        )

        query = acc["sum"] / len(query_dataset)
    elif cfg.query_method == "nearest":
        query = assert_type(Tensor, query_dataset["gradients"]).cuda()
    else:
        raise NotImplementedError

    query /= query.norm(dim=-1, keepdim=True)

    del query_dataset

    # Score the training set
    acc = {"scores": []}

    def score(batch):
        gradients_batch = batch.cuda()

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query

        acc["scores"].append(batch_scores)

    def score_nearest(batch):
        gradients_batch = batch.cuda()

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query.T

        # Take the maximum batch score for each item in the batch
        # (query has multiple rows)
        batch_scores = batch_scores.max(dim=-1).values

        acc["scores"].append(batch_scores)

    train.map(
        score_nearest if cfg.query_method == "nearest" else score,
        input_columns="gradients",
        batched=True,
        batch_size=cfg.map_batch_size,
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

    if cfg.sample:
        probs = torch.softmax(importance_scores / cfg.temperature, dim=0)
        selected_indices = torch.multinomial(probs, cfg.num_examples, replacement=False)
    else:
        # Select the indices of the top-k (or bottom-k) scored items
        sorted_scores = torch.argsort(importance_scores)
        selected_indices = (
            sorted_scores[: cfg.num_examples]
            if cfg.lowest
            else sorted_scores[-cfg.num_examples :]
        )

    # Sort so the filtered dataset is in original order (required for the
    # OrderedFilterSampler mapping; harmless otherwise since the Trainer shuffles).
    selected_indices, _ = selected_indices.sort()
    return selected_indices


def load_ds(cfg: FilterConfig):
    train_split = f"{cfg.split}[:95%]"
    eval_split = f"{cfg.split}[-1:]"

    if cfg.subset:
        train_ds = assert_type(
            Dataset, load_dataset(cfg.dataset, cfg.subset, split=train_split)
        )
    else:
        train_ds = assert_type(Dataset, load_dataset(cfg.dataset, split=train_split))

    if cfg.subset:
        eval_ds = assert_type(
            Dataset, load_dataset(cfg.dataset, cfg.subset, split=eval_split)
        )
    else:
        eval_ds = assert_type(Dataset, load_dataset(cfg.dataset, split=eval_split))

    if cfg.max_samples < len(train_ds):
        train_split = f"{cfg.split}[:{cfg.max_samples}]"
        train_ds = train_ds.select(range(min(cfg.max_samples, len(train_ds))))

    # Add an index column so any shuffles can be undone.
    train_ds = train_ds.add_column("_orig_idx", list(range(len(train_ds))))

    return train_ds, eval_ds, train_split, eval_split


def build_paths(cfg: FilterConfig):
    # Set up data paths
    model_name = cfg.model.split("/")[-1]
    dataset_name = cfg.dataset.split("/")[-1]
    lora_suffix = "_lora" if cfg.use_lora else ""
    proj_suffix = f"_p{cfg.projection_dim}" if cfg.projection_dim != 16 else ""

    trackstar_path = (
        f"examples/runs/{model_name}_{dataset_name}"
        f"_trackstar{lora_suffix}{proj_suffix}{cfg.tag}_{cfg.precision}_{cfg.seed}"
        f"{cfg.ordered}{cfg.max_samples}"
    )
    index_path = cfg.index_dataset or (
        f"examples/runs/{model_name}_{dataset_name}"
        f"_index{lora_suffix}{proj_suffix}_{cfg.seed}_{cfg.learning_rate}"
        f"_{cfg.ordered}{cfg.max_samples}"
    )
    run_name = cfg.name or (
        f"{cfg.model.split('/')[-1]}-{cfg.dataset.split('/')[-1]}-{cfg.filter}"
        f"{cfg.max_samples}{'-lora' if cfg.use_lora else ''}"
        f"{'-lowest' if cfg.lowest else ''}"
        f"-n={cfg.num_examples}"
        f"-s={cfg.seed}"
        f"-p{cfg.precision}"
    )

    sft_path = Path(
        f"examples/runs/{model_name}_{dataset_name}_sft{lora_suffix}_"
        f"{cfg.learning_rate}_{cfg.seed}_{cfg.max_samples}"
    )
    initial_sft_run_name = (
        f"Full SFT {cfg.dataset} {cfg.learning_rate} "
        f"{cfg.model} lora={cfg.use_lora}"
    )

    return (
        sft_path,
        initial_sft_run_name,
        run_name,
        Path(trackstar_path),
        Path(index_path),
    )


def main(
    cfg: FilterConfig,
):
    set_seeds(cfg.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("Running")
        # Set by torchrun
        print("local_rank", local_rank)

    is_distributed = "RANK" in os.environ
    # MASTER_PORT is unique per torchrun invocation so stale sentinel
    # files from previous runs don't cause false-positive barrier passes.
    run_id = os.environ.get("MASTER_PORT", "0")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set up data paths
    sft_path, initial_sft_run_name, run_name, trackstar_path, index_path = build_paths(
        cfg
    )

    # Load the dataset for training.
    ds, eval_ds, train_split, eval_split = load_ds(cfg)

    # Define data config
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, max_length=8192)
    data_config = DataConfig(
        prompt_column=cfg.prompt_column,
        completion_column=cfg.completion_column,
        conversation_column=cfg.conversation_column,
        truncation=True,
    )

    # SFT on the dataset if training statistics are required
    sft_model_path = None
    if cfg.filter in ("attribution", "loss", "trackstar"):
        sft_model_path = sft_full(
            ds, eval_ds, cfg, tokenizer, data_config, sft_path, initial_sft_run_name
        )

        # Destroy NCCL process group before launching multi-GPU subprocesses
        # to free GPU NCCL resources. The subprocess manages its own dist.
        if dist.is_initialized():
            dist.destroy_process_group()

    # Collect artifacts for filtering using evaluation set
    grad_ds = None
    if cfg.filter in ("attribution", "loss"):
        # Collect gradients and final losses from the fine-tuned model
        if local_rank == 0:
            build_index(
                cfg, index_path, train_split, str(sft_model_path), rank=local_rank
            )
        if is_distributed:
            _file_barrier(index_path, run_id, local_rank, world_size)

        grad_ds = load_gradient_dataset(index_path, structured=False)
        grad_ds.set_format("torch")
    elif cfg.filter == "trackstar":
        if local_rank == 0:
            collect_trackstar_scores(
                cfg, trackstar_path, train_split, eval_split, model=str(sft_model_path)
            )
        if is_distributed:
            _file_barrier(trackstar_path, run_id, local_rank, world_size)

    # Filter the training set
    if local_rank == 0:
        print("Filtering...")

    if cfg.filter == "trackstar":
        scores = load_scores(trackstar_path / "scores")
        scores = torch.from_numpy(scores[:].flatten().astype("float32"))

        if local_rank == 0:
            print(
                f"Trackstar score stats: min={scores.min():.4f}, "
                f"max={scores.max():.4f}, "
                f"mean={scores.mean():.4f}, "
                f"std={scores.std():.4f}"
            )

        sorted_local = torch.argsort(scores)
        selected_indices = (
            sorted_local[: cfg.num_examples]
            if cfg.lowest
            else sorted_local[-cfg.num_examples :]
        )
        selected_indices, _ = selected_indices.sort()
        train = ds.select(selected_indices)
    elif cfg.filter == "attribution":
        selected_indices = _get_attribution_indices(cfg, grad_ds, run_name)
        train = ds.select(selected_indices)
    elif cfg.filter == "classification":
        if "score" in ds.column_names:
            train = ds.sort("score", reverse=not cfg.lowest)
            train = train.select(range(min(cfg.num_examples, len(train))))
        else:
            ranks = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "very poor": 0}

            def add_rank(ex):
                q = ex.get("quality")
                return {"_q": ranks.get(q, -1)}

            train = (
                ds.map(add_rank)
                .filter(lambda x: x["_q"] >= 0)
                .sort("_q", reverse=not cfg.lowest)
            )
            train = train.select(
                range(min(cfg.num_examples, len(train)))
            ).remove_columns("_q")
        # Restore original order so the sampler mapping is well-defined
        train = train.sort("_orig_idx")
    elif cfg.filter == "loss":
        # Filter based on the final item loss
        grad_loss = grad_ds.map(
            lambda x: {"loss_val": x["loss"].item()},
        )
        sorted_scores = torch.argsort(torch.tensor(grad_loss["loss_val"]))
        selected_indices = (
            sorted_scores[: cfg.num_examples]
            if cfg.lowest
            else sorted_scores[-cfg.num_examples :]
        )
        selected_indices, _ = selected_indices.sort()
        train = ds.select(selected_indices)
    elif cfg.filter == "random":
        generator = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(ds), generator=generator)
        selected_indices = perm[: min(cfg.num_examples, len(ds))]
        selected_indices, _ = selected_indices.sort()
        train = ds.select(selected_indices)
    else:
        raise ValueError(f"Invalid filter: {cfg.filter}")

    # Build the ordered sampler before removing _orig_idx.
    # The sampler replays randperm(full_dataset_size) each epoch and yields
    # only positions corresponding to items kept after filtering, so the
    # retrain sees items in the same order as the original full-dataset run.
    sampler = None
    if cfg.ordered:
        orig_to_pos = {int(idx): pos for pos, idx in enumerate(train["_orig_idx"])}
        sampler = OrderedFilterSampler(len(ds), orig_to_pos, cfg.seed)

    if "_orig_idx" in train.column_names:
        train = train.remove_columns("_orig_idx")

    # Tokenize and retrain from the base model on the filtered subset
    if local_rank == 0:
        print(f"Training on {len(train)} examples out of {len(ds)}.")

    train = train.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer, max_length=2048),
    )
    eval_ds = eval_ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer, max_length=2048),
    )

    metrics = run_sft(
        cfg,
        train,
        eval_ds,
        Path(f"examples/runs/{run_name}"),
        run_name=run_name,
        sampler=sampler,
    )

    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Filter: {cfg.filter}")
        print(f"Num training examples: {len(train)}")
        if metrics:
            print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
            print(f"Final eval loss: {metrics.get('eval_loss', 'N/A')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    cfg = parse(FilterConfig)

    main(cfg)
