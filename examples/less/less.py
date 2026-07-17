"""Launch this script with torchrun.
torchrun --nproc_per_node 8 -m examples.less.less --pdbs 4
"""

import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from peft import LoraConfig, get_peft_model
from simple_parsing import parse
from torch import Tensor
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bergson.config import DataConfig
from bergson.data import load_gradient_dataset, tokenize
from bergson.utils.utils import assert_type
from examples.less.download_less import download_less

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class LESSConfig:
    """Config for building the index and running the model/dataset pipeline."""

    model: str = "meta-llama/Llama-2-7b-hf"
    """Model to load."""

    split: str = "train"

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    num_examples: int = 5_000
    """Number of items to select from the training set after filtering."""

    eval_ds: Literal["mmlu"] = "mmlu"

    eval_split: str = "test"

    eval_prompt_column: str = "inputs"
    """Column in the dataset that contains the prompts."""

    eval_completion_column: str = "targets"
    """Optional column in the dataset that contains the completions."""

    eval_conversation_column: str = ""

    map_batch_size: int = 512
    """Batch size for processing the dataset."""

    seed: int = 42
    """Seed for reproducibility."""

    lowest: bool = False
    """Select the lowest scores."""

    warmup_epochs: int = 4
    """Number of epochs to train for."""

    warmup_fraction: float = 0.05

    lora_rank: int = 128
    """LoRA rank."""

    projection_dim: int = 8192
    """Projection dimension for the gradient index."""

    token_batch_size: int = 2048
    """Per-batch token budget for `bergson build`. Doubles as the per-example
    truncation cap (bergson sets ``max_length = min(model_max, token_batch_size)``
    when truncation is on)."""

    pdbs: int = 1
    "Per-device batch size"

    learning_rate: float = 2e-5

    precision: str = "fp32"

    subset: str = ""

    test: bool = False

    warmup_repo: str = "EleutherAI/less-replication-7b-warmup"
    """HF repo containing warmup checkpoints as ``epoch-{N}`` revisions. If
    set and warmup checkpoints aren't on disk, download them instead of
    running warmup. Set to empty string to disable the HF fallback."""

    recompute_warmup: bool = False
    """Force a fresh warmup SFT even if checkpoints are on disk or available
    on ``warmup_repo``. Off by default — pulls the published checkpoints when
    available so step 2+ can run without redoing step 1."""


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
    cfg: LESSConfig,
    ds: Dataset,
    output_path: Path,
    num_epochs: int,
    tokenizer,
    data_config: DataConfig,
    run_name: str | None = None,
    sampler: Sampler[int] | None = None,
):
    """SFT with HF Trainer, which handles DDP automatically.
    Returns the final eval metrics."""
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

    print(f"SFT on ({len(ds)} examples)...")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        dtype=torch.float32,
    )

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)  # type: ignore
    model.print_trainable_parameters()  # type: ignore

    effective_batch_size = 128
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_acc_steps = effective_batch_size // world_size // cfg.pdbs
    # num_train_steps = (len(train) // effective_batch_size) * num_epochs
    # eval_steps = max(1, num_train_steps // 5)

    trainer = SFTTrainer(
        model=model,  # type: ignore
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            max_length=8192,
            packing=True,
            output_dir=str(output_path),
            per_device_train_batch_size=cfg.pdbs,
            per_device_eval_batch_size=cfg.pdbs,
            gradient_accumulation_steps=grad_acc_steps,
            learning_rate=cfg.learning_rate,
            num_train_epochs=num_epochs,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            # eval_strategy="steps",
            # eval_steps=eval_steps,
            save_total_limit=num_epochs,
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

    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)

    print(f"SFT checkpoint saved to {output_path}")

    # Free GPU memory so subprocesses (bergson build) can use it.
    # empty_cache is needed here to release memory to a *separate process*.
    trainer.model.cpu()  # type: ignore
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        # Move optimizer state off GPU
        for state in trainer.optimizer.state.values():  # type: ignore
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Destroy NCCL process group before launching multi-GPU subprocesses
    # to free GPU NCCL resources. The subprocess manages its own dist.
    if dist.is_initialized():
        dist.destroy_process_group()


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

    return {k: v for k, v in os.environ.items() if k not in _TORCHRUN_ENV_VARS}


def build_subset_indices(
    cfg: LESSConfig,
    index_path: Path,
    model: str,
    data_path: Path,
    format_template: str = "",
    conversation_column: str = "",
    optimizer_state: str = "",
    nproc_per_node: int = 0,
) -> None:
    """Build a gradient index per subset found under *data_path*.
    Each subdirectory is treated as a subset.  The dataset is resolved as
    an HF ``save_to_disk`` directory (has ``dataset_info.json``) or the
    first ``*.jsonl`` file found.
    """
    subsets: dict[str, str] = {}
    for sub in sorted(data_path.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "dataset_info.json").exists():
            subsets[sub.name] = str(sub)
        else:
            jsonl_files = list(sub.glob("*.jsonl"))
            if jsonl_files:
                subsets[sub.name] = str(jsonl_files[0])

    assert subsets, f"No subsets found under {data_path}"
    print(f"Building {len(subsets)} subset indices under {index_path}")

    for name, dataset_str in subsets.items():
        sub_index = index_path / name
        if sub_index.exists():
            print(f"Skipping {name} (already built)")
            continue

        cmd = [
            sys.executable,
            "-m",
            "bergson",
            "build",
            str(sub_index),
            "--model",
            model,
            "--dataset",
            dataset_str,
            "--truncation",
            "--projection_dim",
            str(cfg.projection_dim),
            "--projection_target",
            "global",
            "--token_batch_size",
            str(cfg.token_batch_size),
            "--precision",
            cfg.precision,
            "--overwrite",
        ]
        if optimizer_state:
            cmd += ["--optimizer_state", optimizer_state]
        if format_template:
            cmd += ["--format_template", format_template]
        if conversation_column:
            cmd += ["--conversation_column", conversation_column]
        if nproc_per_node > 0:
            cmd += ["--nproc_per_node", str(nproc_per_node)]

        print(f"Building {name}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=_clean_dist_env())
        if result.returncode != 0:
            raise RuntimeError(
                f"bergson build failed for {name} (exit code {result.returncode})"
            )


def evaluate_mmlu(model, tokenizer, num_fewshot=5, batch_size=8, subjects=None):
    """Run MMLU evaluation and return accuracy.
    Args:
        model: HuggingFace model (already loaded).
        tokenizer: Corresponding tokenizer.
        num_fewshot: Number of few-shot examples (default 5).
        batch_size: Eval batch size.
        subjects: Optional list of MMLU subjects (e.g. ["abstract_algebra", "anatomy"]).
                  If None, runs all 57 subjects.
    Returns:
        dict with "overall_acc" and per-subject "subject_accs".
    """
    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    if subjects:
        tasks = [f"mmlu_{s}" for s in subjects]
    else:
        tasks = ["mmlu"]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    results = assert_type(dict, results)

    subject_accs = {
        task_name.removeprefix("mmlu_"): metrics["acc,none"]
        for task_name, metrics in results["results"].items()
        if task_name != "mmlu"
    }

    overall_acc = results["results"].get("mmlu", {}).get("acc,none")
    if overall_acc is None and subject_accs:
        overall_acc = sum(subject_accs.values()) / len(subject_accs)

    return {
        "overall_acc": overall_acc,
        "subject_accs": subject_accs,
    }


def _get_checkpoint_mean_lr(checkpoint_path: Path) -> float:
    """Mean learning rate over the epoch that ends at this checkpoint.

    Reads trainer_state.json's log_history (one entry per step because
    logging_steps=1) and averages learning_rate over the half-open epoch
    range (end_epoch - 1, end_epoch].
    """
    state_path = checkpoint_path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.json in {checkpoint_path}")
    with state_path.open() as f:
        state = json.load(f)

    end_epoch = state["epoch"]
    start_epoch = end_epoch - 1.0

    lrs = [
        entry["learning_rate"]
        for entry in state["log_history"]
        if "learning_rate" in entry and start_epoch < entry["epoch"] <= end_epoch
    ]
    assert lrs, f"No LR log entries in epoch range for {checkpoint_path.name}"
    return sum(lrs) / len(lrs)


def _compute_epoch_scores(
    cfg: LESSConfig, train_grad_ds: Dataset, eval_index_path: Path, device
) -> Tensor:
    """Compute cosine-similarity scores between train and eval gradients.
    Returns a 1-D tensor of scores (one per training example), without
    any selection applied.
    """
    eval_queries = []
    for subdir in sorted(eval_index_path.iterdir()):
        if subdir.is_dir():
            eval_grad_ds = load_gradient_dataset(subdir, structured=False)
            eval_grad_ds.set_format("torch")

            # Compute the mean of the gradients in the evaluation subset
            acc = {"sum": torch.zeros_like(eval_grad_ds[0]["gradients"], device=device)}

            def sum_(col):
                acc["sum"] += col.to(device).sum(0)

            # Do not use num_proc because we are accumulating in a single variable
            # https://colab.research.google.com/drive/1jCLv31Y4cDfqD0lhO0AnqEv3Or-LLvWe?usp=sharing
            eval_grad_ds.map(
                sum_,
                input_columns="gradients",
                batched=True,
                batch_size=cfg.map_batch_size,
            )
            query = acc["sum"] / len(eval_grad_ds)

            # Append the mean gradient to the query set
            eval_queries.append(query)

    # Stack per-subject mean gradients into a (num_subjects, grad_dim) tensor
    query = torch.stack(eval_queries)
    query_norms = query.norm(dim=-1, keepdim=True)
    zero_query = (query_norms == 0).sum().item()
    if zero_query:
        print(f"WARNING: {zero_query}/{len(query)} eval query gradients have zero norm")
    query = torch.nan_to_num(query / query_norms)

    # Score the training set
    acc = {"scores": [], "zero_norm_count": 0, "total_count": 0}

    def score_nearest(batch):
        gradients_batch = batch.to(device)

        norms = gradients_batch.norm(dim=1, keepdim=True)
        acc["zero_norm_count"] += (norms == 0).sum().item()
        acc["total_count"] += gradients_batch.shape[0]
        gradients_batch = torch.nan_to_num(gradients_batch / norms)

        batch_scores = gradients_batch @ query.T

        # Take the maximum batch score for each item in the batch
        # (query has multiple rows)
        batch_scores = batch_scores.max(dim=-1).values

        acc["scores"].append(batch_scores)

    train_grad_ds.map(
        score_nearest,
        input_columns="gradients",
        batched=True,
        batch_size=cfg.map_batch_size,
    )

    zero_pct = acc["zero_norm_count"] / max(acc["total_count"], 1) * 100
    print(
        f"Zero-norm train gradients: {acc['zero_norm_count']}/{acc['total_count']} "
        f"({zero_pct:.2f}%)"
    )
    if zero_pct > 1:
        print("WARNING: >1% zero-norm gradients — this may indicate a bug")

    return torch.cat(acc["scores"], dim=0).to(device)


def _select_from_scores(
    cfg: LESSConfig,
    importance_scores: Tensor,
    scores_path: Path,
) -> Tensor:
    """Select top-k (or bottom-k) indices from pre-computed scores."""
    print(
        f"Score stats: min={importance_scores.min():.6f}, "
        f"max={importance_scores.max():.6f}, "
        f"mean={importance_scores.mean():.6f}, "
        f"std={importance_scores.std():.6f}"
    )

    print("Saving importance scores to disk.")
    torch.save(importance_scores, scores_path)

    # Select the indices of the top-k (or bottom-k) scored items
    sorted_scores = torch.argsort(importance_scores)
    selected_indices = (
        sorted_scores[: cfg.num_examples]
        if cfg.lowest
        else sorted_scores[-cfg.num_examples :]
    )

    # Sort so the filtered dataset is in original order (required for the
    # OrderedFilterSampler mapping)
    selected_indices, _ = selected_indices.sort()
    return selected_indices


def load_ds(cfg: LESSConfig, rank: int = 0):
    dataset_paths = download_less(rank)

    if dist.is_initialized():
        dist.barrier()

    base = dataset_paths / "data"

    # Load and merge train datasets, tagging each with its source name.
    train_ds_path = base / "train" / "processed"
    train_datasets = []
    for ds_name in ["cot", "dolly", "flan_v2", "oasst1"]:
        ds = load_dataset(
            "json",
            data_files=str(train_ds_path / ds_name / "*.jsonl"),
            split="train",
        )
        ds = ds.add_column("ds_name", [ds_name] * len(ds))
        train_datasets.append(ds)

    train_ds = concatenate_datasets(train_datasets)

    # Add an index column so any shuffles can be undone.
    train_ds = train_ds.add_column("_orig_idx", list(range(len(train_ds))))

    # Process MMLU eval CSVs into per-subject HF datasets matching the MCQA
    # template format: question (str), choices (list[str]), answer (int index).
    eval_path = base / "eval" / "mmlu_processed"
    eval_path.mkdir(parents=True, exist_ok=True)
    answer_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for csv_file in sorted((base / "eval" / "mmlu" / cfg.eval_split).glob("*.csv")):
        subject = csv_file.stem.removesuffix(f"_{cfg.eval_split}")
        subject_path = eval_path / subject
        if (subject_path / "dataset_info.json").exists():
            continue
        ds = load_dataset(
            "csv",
            data_files=str(csv_file),
            column_names=["question", "A", "B", "C", "D", "answer"],
            split="train",
        )
        ds = ds.map(
            lambda row: {
                "choices": [row["A"], row["B"], row["C"], row["D"]],
                "answer": answer_to_idx[row["answer"]],
            }
        )
        ds = ds.remove_columns(["A", "B", "C", "D"])
        ds.save_to_disk(str(subject_path))

    train_data_config = DataConfig(
        prompt_column="",
        completion_column="",
        conversation_column="messages",
        truncation=True,
    )

    return train_ds, train_ds_path, eval_path, train_data_config


def load_test_ds(cfg: LESSConfig, local_rank: int, warmup_path: Path):
    """Build the small dataset slice + 2-subject MMLU eval used by --test runs.

    Mutates ``cfg`` to reduce defaults (num_examples, warmup_epochs,
    warmup_fraction) so the rest of the pipeline finishes quickly. Mirrors
    the return signature of ``load_ds``.
    """
    cfg.num_examples = 200
    cfg.warmup_epochs = 1
    cfg.warmup_fraction = 0.5

    # Use a slice of the production LESS data so the chat template +
    # bergson tokenize path matches what the real pipeline uses, and
    # the assistant content is recoverable by `tokenize()`'s substring
    # search. Magpie's content gets altered by Jinja rendering and
    # fails that check.
    full_ds, _, full_eval_path, _ = load_ds(cfg)
    ds = full_ds.select(range(min(1000, len(full_ds))))
    ds = ds.remove_columns(
        [c for c in ds.column_names if c not in ("messages", "_orig_idx")]
    )

    train_data_path = warmup_path.parent / "test_train_data" / "less_slice"
    eval_data_path = warmup_path.parent / "test_eval_data"
    test_subjects = sorted(p.name for p in full_eval_path.iterdir() if p.is_dir())[:2]

    # Rank-0 only: save_to_disk and symlink_to race across ranks.
    if local_rank == 0:
        train_data_path.mkdir(parents=True, exist_ok=True)
        if not (train_data_path / "dataset_info.json").exists():
            ds.save_to_disk(str(train_data_path))
        eval_data_path.mkdir(parents=True, exist_ok=True)
        for subj in test_subjects:
            link = eval_data_path / subj
            if not link.exists():
                link.symlink_to((full_eval_path / subj).resolve())
    if dist.is_initialized():
        dist.barrier()
    train_data_path = train_data_path.parent

    train_data_config = DataConfig(
        prompt_column="",
        completion_column="",
        conversation_column="messages",
        truncation=True,
    )
    return ds, train_data_path, eval_data_path, train_data_config


def build_paths(cfg: LESSConfig):
    # Set up data paths
    model_name = cfg.model.split("/")[-1]

    if cfg.test:
        run_path = Path(
            f"runs/less-test/{model_name}"
            f"p{cfg.projection_dim}_s{cfg.seed}_lr{cfg.learning_rate}"
        )
    else:
        run_path = Path(
            f"runs/less/{model_name}"
            f"p{cfg.projection_dim}_s{cfg.seed}_lr{cfg.learning_rate}"
        )
    os.makedirs(run_path, exist_ok=True)

    run_name = run_path.stem
    warmup_run_name = f"warmup_{run_path.stem}_{cfg.pdbs}"

    warmup_path = run_path / "warmup"
    eval_index_path = run_path / "eval_index"
    train_index_path = run_path / "train_index"
    final_path = run_path / "filtered_model"
    scores_path = run_path / "importance_scores.pt"

    return (
        warmup_path,
        warmup_run_name,
        train_index_path,
        eval_index_path,
        final_path,
        run_name,
        scores_path,
    )


def download_warmup_from_hub(warmup_path: Path, repo_id: str, num_epochs: int) -> None:
    """Pull warmup checkpoints from ``repo_id``'s ``epoch-{N}`` revisions.

    Each revision's contents are placed in ``warmup_path/checkpoint-{N}``.
    Logs (and skips) revisions that are already present locally.
    """
    from huggingface_hub import snapshot_download

    warmup_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        target = warmup_path / f"checkpoint-{epoch}"
        if (target / "adapter_config.json").exists():
            print(f"Warmup checkpoint-{epoch} already on disk, skipping download")
            continue
        revision = f"epoch-{epoch}"
        print(f"Downloading {repo_id}@{revision} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
        )


def main(
    cfg: LESSConfig,
):
    set_seeds(cfg.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    run_id = os.environ.get("MASTER_PORT", "0")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # Set up data paths
    (
        warmup_path,
        warmup_run_name,
        train_index_path,
        eval_index_path,
        final_path,
        run_name,
        scores_path,
    ) = build_paths(cfg)

    # Load the dataset for training.
    if cfg.test:
        ds, train_data_path, eval_data_path, train_data_config = load_test_ds(
            cfg, local_rank, warmup_path
        )
    else:
        ds, train_data_path, eval_data_path, train_data_config = load_ds(cfg)

    # Define data config
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, max_length=8192)

    # Define the chat template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}{{ eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    warmup_ds = ds.select(range(math.ceil(len(ds) * cfg.warmup_fraction)))

    have_local_warmup = bool(list(warmup_path.glob("checkpoint-*")))
    use_hub = (
        not have_local_warmup
        and not cfg.recompute_warmup
        and cfg.warmup_repo
        and not cfg.test
    )
    if use_hub:
        if local_rank == 0:
            download_warmup_from_hub(warmup_path, cfg.warmup_repo, cfg.warmup_epochs)
        _file_barrier(warmup_path, "hub_warmup", local_rank, world_size)
        have_local_warmup = bool(list(warmup_path.glob("checkpoint-*")))

    if cfg.recompute_warmup or not have_local_warmup:
        run_sft(
            cfg,
            warmup_ds,
            warmup_path,
            cfg.warmup_epochs,
            tokenizer,
            train_data_config,
            warmup_run_name,
        )
    elif local_rank == 0:
        print(f"Using existing warmup checkpoints in {warmup_path}")

    # Build gradient indices and score at each warmup epoch checkpoint,
    # weighting by the checkpoint's learning rate, then sum for final scores.
    checkpoint_dirs = sorted(warmup_path.glob("checkpoint-*"))
    assert checkpoint_dirs, f"No checkpoints found in {warmup_path}"
    if local_rank == 0:
        print(
            f"Using {len(checkpoint_dirs)} checkpoints: "
            f"{[d.name for d in checkpoint_dirs]}"
        )

    accumulated_scores: Tensor | None = None

    for ckpt_dir in checkpoint_dirs:
        # e.g. "checkpoint-106"
        epoch_eval_index = eval_index_path / ckpt_dir.name
        epoch_train_index = train_index_path / ckpt_dir.name

        lr: float = _get_checkpoint_mean_lr(ckpt_dir)

        if local_rank == 0:
            print(f"Epoch checkpoint: {ckpt_dir.name}, mean_lr={lr:.6e}")
            # Per the LESS paper's InfAdam formula: the train side is the
            # Adam-preconditioned per-example direction Γ̃(z, θ); the eval
            # side is the raw averaged validation gradient ∇̄ℓ. Only train
            # gets the optimizer state.
            # Use 1 GPU for eval (subjects have ~100 examples each).
            build_subset_indices(
                cfg,
                epoch_eval_index,
                str(ckpt_dir),
                eval_data_path,
                format_template="bergson/templates/mcqa.yaml",
                nproc_per_node=1,
            )
            build_subset_indices(
                cfg,
                epoch_train_index,
                str(ckpt_dir),
                train_data_path,
                conversation_column="messages",
                optimizer_state=str(ckpt_dir),
                nproc_per_node=world_size,
            )

        _file_barrier(
            eval_index_path, f"{run_id}_{ckpt_dir.name}", local_rank, world_size
        )

        # Score on rank 0
        if local_rank == 0:
            train_grad_parts = []
            for subdir in sorted(epoch_train_index.iterdir()):
                if subdir.is_dir():
                    grad_ds = load_gradient_dataset(subdir, structured=False)
                    grad_ds = grad_ds.add_column(
                        "ds_name", [subdir.stem] * len(grad_ds)
                    )
                    train_grad_parts.append(grad_ds)

            train_grad_ds = concatenate_datasets(train_grad_parts)
            train_grad_ds.set_format("torch")

            epoch_scores = _compute_epoch_scores(
                cfg, train_grad_ds, epoch_eval_index, device
            )
            weighted_scores = epoch_scores * lr

            if accumulated_scores is None:
                accumulated_scores = weighted_scores
            else:
                accumulated_scores += weighted_scores

            print(
                f"Epoch {ckpt_dir.name} scores: "
                f"mean={epoch_scores.mean():.4f}, mean_lr={lr:.6e}, "
                f"weighted_mean={weighted_scores.mean():.6e}"
            )
            del train_grad_ds, train_grad_parts

    # Scoring runs on rank 0 only; broadcast selected_indices to all ranks
    if local_rank == 0:
        assert accumulated_scores is not None
        print("\nSelecting from accumulated lr-weighted scores...")
        selected_indices = _select_from_scores(cfg, accumulated_scores, scores_path)
        torch.save(selected_indices, scores_path.parent / "selected_indices.pt")

    _file_barrier(scores_path.parent, "scoring", local_rank, world_size)

    selected_indices = torch.load(
        scores_path.parent / "selected_indices.pt", map_location="cpu"
    )
    filtered_ds = ds.select(selected_indices)

    # Build the ordered sampler before removing _orig_idx.
    # The sampler replays randperm(full_dataset_size) each epoch and yields
    # only positions corresponding to items kept after filtering, so the
    # retrain sees items in the same order as the original full-dataset run.
    orig_to_pos = {int(idx): pos for pos, idx in enumerate(filtered_ds["_orig_idx"])}
    sampler = OrderedFilterSampler(len(ds), orig_to_pos, cfg.seed)

    if "_orig_idx" in filtered_ds.column_names:
        filtered_ds = filtered_ds.remove_columns("_orig_idx")

    # Tokenize and retrain from the base model on the filtered subset
    if local_rank == 0:
        print(f"Training on {len(filtered_ds)} examples out of {len(ds)}.")

    # Re-init so Accelerate's DDP wrapper has a fresh clean process group.
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group("nccl")

    run_sft(
        cfg,
        filtered_ds,
        final_path,
        num_epochs=1,
        run_name=run_name,
        tokenizer=tokenizer,
        data_config=train_data_config,
        sampler=sampler,
    )
    print(f"SFT checkpoint saved to {final_path}")

    # Re-init dist for MMLU eval (run_sft destroys the process group)
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group("nccl")

    final_model = AutoModelForCausalLM.from_pretrained(
        final_path, device_map={"": device}, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(final_path)

    eval_subjects = None
    if cfg.test:
        eval_subjects = sorted(p.name for p in eval_data_path.iterdir() if p.is_dir())
    mmlu_results = evaluate_mmlu(
        final_model, tokenizer, subjects=eval_subjects, batch_size=32
    )

    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"MMLU 5-shot accuracy: {mmlu_results['overall_acc']:.4f}")
        for subject, acc in sorted(mmlu_results["subject_accs"].items()):
            print(f"  {subject}: {acc:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    cfg = parse(LESSConfig)

    main(cfg)
