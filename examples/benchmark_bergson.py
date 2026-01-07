"""Utilities for benchmarking Bergson influence analysis scaling (in-memory reduce + score)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.collection import pad_and_tensor
from bergson.gradients import GradientCollector, GradientProcessor
from bergson.utils import assert_type, get_layer_list

# Import from same directory
try:
    from benchmark_common import MODEL_SPECS, ModelSpec, DEFAULT_DATASET
except ImportError:
    from examples.benchmark_common import MODEL_SPECS, ModelSpec, DEFAULT_DATASET

SCHEMA_VERSION = 1
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


@dataclass
class RunRecord:
    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    eval_tokens: int
    dataset: str
    train_split: str
    eval_split: str
    batch_size: int
    max_length: int
    reduce_seconds: float | None  # Time to collect training gradients
    score_seconds: float | None  # Time to compute inner products
    total_runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace(",", "")
    if text.endswith("tokens"):
        text = text[:-6]
    if not text:
        raise ValueError("empty token spec")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    unit = 1
    if text[-1] in suffixes:
        unit = suffixes[text[-1]]
        text = text[:-1]
    number = float(text)
    return int(number * unit)


def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        value = tokens / 1_000_000_000
        suffix = "B"
    elif tokens >= 1_000_000:
        value = tokens / 1_000_000
        suffix = "M"
    elif tokens >= 1_000:
        value = tokens / 1_000
        suffix = "K"
    else:
        return str(tokens)
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.2f}{suffix}"


def timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_run_path(
    base: Path,
    spec: ModelSpec,
    train_tokens: int,
    eval_tokens: int,
    tag: str | None,
) -> Path:
    train_label = format_tokens(train_tokens)
    eval_label = format_tokens(eval_tokens)
    run_tag = tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = base / spec.key / f"{train_label}-{eval_label}-{run_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_record(path: Path, record: RunRecord) -> None:
    with open(path / "benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)


def cmd_run(args: argparse.Namespace) -> None:
    if args.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model}'")
    spec = MODEL_SPECS[args.model]
    train_tokens = parse_tokens(args.train_tokens)
    eval_tokens = parse_tokens(args.eval_tokens)
    
    # Enable FSDP for larger models (>= 1B parameters) or if explicitly requested
    use_fsdp = args.fsdp or (spec.params >= 1_000_000_000)
    
    if use_fsdp:
        print(
            f"Running Bergson benchmark for {args.model} with {train_tokens} train "
            f"and {eval_tokens} eval tokens (using FSDP)"
        )
    else:
        print(
            f"Running Bergson benchmark for {args.model} with {train_tokens} train "
            f"and {eval_tokens} eval tokens"
        )

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_path = (
        Path(args.run_path).resolve()
        if args.run_path
        else ensure_run_path(run_root, spec, train_tokens, eval_tokens, args.tag)
    )

    start_wall = timestamp()
    start = time.perf_counter()
    status = "success"
    error_message: str | None = None
    reduce_time: float | None = None
    score_time: float | None = None

    try:
        # Set up distributed training if FSDP is enabled
        rank = 0
        world_size = 1
        if use_fsdp:
            # Check if we're in a distributed environment
            if "LOCAL_RANK" in os.environ:
                rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
            else:
                # Single GPU - FSDP will still work but won't provide benefits
                world_size = torch.cuda.device_count()
                if world_size <= 1:
                    print("Warning: FSDP requested but only 1 GPU available. FSDP will have minimal benefit.")
            
            if world_size > 1:
                torch.cuda.set_device(rank)
                addr = os.environ.get("MASTER_ADDR", "localhost")
                port = os.environ.get("MASTER_PORT", "29500")
                
                dist.init_process_group(
                    "nccl",
                    init_method=f"tcp://{addr}:{port}",
                    device_id=torch.device(f"cuda:{rank}"),
                    rank=rank,
                    timeout=timedelta(hours=1),
                    world_size=world_size,
                )
                print(f"Initialized distributed training: rank={rank}, world_size={world_size}")
            else:
                rank = 0
                world_size = 1

        # Load model and tokenizer
        # For FSDP, load to CPU first, then wrap with FSDP
        device_map = "cpu" if use_fsdp else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map=device_map
        )
        
        if not use_fsdp:
            model.cuda()
        else:
            # Move to appropriate device and wrap with FSDP
            if world_size > 1:
                torch.cuda.set_device(rank)
            else:
                model = model.cuda()
            
            # Wrap model with FSDP
            embed = model.get_input_embeddings()
            model.requires_grad_(False)  # Freeze the model
            embed.requires_grad_(True)  # Make sure backward hooks are called though
            
            # Shard each individual transformer layer
            for layer in get_layer_list(model):
                fully_shard(layer)
            
            # Shard the entire model
            fully_shard(model)
            
            if rank == 0:
                print("Model wrapped with FSDP")

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize(batch):
            encoded = tokenizer.batch_encode_plus(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            # Add labels for loss computation
            encoded["labels"] = encoded["input_ids"].clone()
            return encoded

        # Load datasets
        train_dataset = assert_type(
            Dataset, load_dataset(args.dataset, split=args.train_split)
        )

        # Estimate examples needed based on token count
        max_length = args.max_length or 512
        train_examples_needed = max(1, train_tokens // max_length)
        eval_examples_needed = max(1, eval_tokens // max_length)

        # Select enough examples
        total_needed = train_examples_needed + eval_examples_needed
        train_dataset = train_dataset.select(range(min(total_needed, len(train_dataset))))

        eval_dataset = train_dataset.select(
            range(train_examples_needed, train_examples_needed + eval_examples_needed)
        )
        train_dataset = train_dataset.select(range(train_examples_needed))

        train_dataset = train_dataset.map(tokenize, batched=True)
        eval_dataset = eval_dataset.map(tokenize, batched=True)

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        eval_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # Create processor (no normalization, no preconditioners, no projection)
        processor = GradientProcessor(
            normalizers={},  # No normalization
            projection_dim=None,  # No projection
            reshape_to_square=False,
            projection_type="rademacher",
        )

        # REDUCE PHASE: Collect training gradients in-memory
        # Only run on rank 0 when using distributed training
        if use_fsdp and world_size > 1 and rank != 0:
            # Other ranks wait
            dist.barrier()
        else:
            if rank == 0 or world_size == 1:
                print("Collecting training gradients (reduce phase)...")
            reduce_start = time.perf_counter()
            
            train_grads = defaultdict(list)
            
            def train_callback(name: str, g: torch.Tensor):
                # Flatten and store gradients in-memory
                # No normalization, no preconditioning as per requirements
                train_grads[name].append(g.flatten(1).cpu())
            
            train_collector = GradientCollector(
                model.base_model,
                train_callback,
                processor,
            )
            
            # Process training data in batches
            for i in range(0, len(train_dataset), args.batch_size):
                batch_indices = list(range(i, min(i + args.batch_size, len(train_dataset))))
                batch_items = [train_dataset[j] for j in batch_indices]
                
                # Extract and convert to lists for pad_and_tensor
                input_ids_list = [item["input_ids"].cpu().tolist() if isinstance(item["input_ids"], torch.Tensor) else item["input_ids"] for item in batch_items]
                labels_list = [item["labels"].cpu().tolist() if isinstance(item.get("labels"), torch.Tensor) else item.get("labels", item["input_ids"]) for item in batch_items]
                
                # Get the device - for FSDP, use the current rank's device
                if use_fsdp and world_size > 1:
                    device = torch.device(f"cuda:{rank}")
                else:
                    device = next(model.parameters()).device
                
                x, y = pad_and_tensor(
                    input_ids_list,
                    labels=labels_list,
                    device=device,
                )
                
                with train_collector:
                    logits = model(x).logits[:, :-1]
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y[:, 1:].flatten(),
                        reduction="none",
                    ).reshape_as(y[:, 1:])
                    # Mean reduction per example
                    masks = y[:, 1:] != -100
                    denoms = masks.sum(dim=1, dtype=logits.dtype)
                    losses = losses.sum(1) / denoms
                    losses.mean().backward()
                
                model.zero_grad()
                torch.cuda.synchronize()
            
            # Concatenate all training gradients
            train_grads_flat = {
                name: torch.cat(grads, dim=0) for name, grads in train_grads.items()
            }
            del train_grads
            
            reduce_time = time.perf_counter() - reduce_start
            if rank == 0 or world_size == 1:
                print(f"Reduce phase completed in {reduce_time:.2f} seconds")
                print(f"Training gradients shape: {[(k, v.shape) for k, v in train_grads_flat.items()]}")

            # SCORE PHASE: Compute inner products with test gradients
            if rank == 0 or world_size == 1:
                print("Computing influence scores (score phase)...")
            score_start = time.perf_counter()
            
            all_scores = []
            
            for i, example in enumerate(eval_dataset):
                # Get the device - for FSDP, use the current rank's device
                if use_fsdp and world_size > 1:
                    device = torch.device(f"cuda:{rank}")
                else:
                    device = next(model.parameters()).device
                
                input_ids = example["input_ids"].unsqueeze(0).to(device)
                labels = example["labels"].unsqueeze(0).to(device)
                
                # Collect test gradient
                test_grads = {}
                
                def test_callback(name: str, g: torch.Tensor):
                    test_grads[name] = g.flatten(1).cpu()
                
                test_collector = GradientCollector(
                    model.base_model,
                    test_callback,
                    processor,
                )
                
                with test_collector:
                    logits = model(input_ids).logits[:, :-1]
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, 1:].flatten(),
                        reduction="mean",
                    )
                    loss.backward()
                
                model.zero_grad()
                torch.cuda.synchronize()
                
                # Compute inner products (no normalization, no preconditioning)
                # Sum across all modules
                scores = torch.zeros(len(train_dataset), device="cpu")
                for name in test_grads:
                    if name in train_grads_flat:
                        # Inner product: test_grad @ train_grads^T
                        scores += (test_grads[name] @ train_grads_flat[name].T).squeeze(0)
                
                all_scores.append(scores)
                
                if i >= args.max_eval_examples - 1:
                    break
            
            score_time = time.perf_counter() - score_start
            if rank == 0 or world_size == 1:
                print(f"Score phase completed in {score_time:.2f} seconds")
                print(f"Computed scores for {len(all_scores)} test examples")
        
        # Synchronize all ranks before saving
        if use_fsdp and world_size > 1:
            dist.barrier()

    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = repr(exc)
        import traceback
        traceback.print_exc()

    runtime = time.perf_counter() - start
    end_wall = timestamp()

    record = RunRecord(
        schema_version=SCHEMA_VERSION,
        status=status,
        model_key=spec.key,
        model_name=spec.hf_id,
        params=spec.params,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        dataset=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        max_length=args.max_length or 512,
        reduce_seconds=reduce_time,
        score_seconds=score_time,
        total_runtime_seconds=runtime,
        start_time=start_wall,
        end_time=end_wall,
        run_path=str(run_path),
        notes=args.notes,
        error=error_message,
    )
    save_record(run_path, record)

    print(json.dumps(asdict(record), indent=2))

    if status != "success":
        sys.exit(1)


def load_records(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for meta in root.rglob("benchmark.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(RunRecord(**payload))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Bergson influence analysis scaling (in-memory)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Examples:
  python -m examples.benchmark_bergson run pythia-14m 1M 100K
  python -m examples.benchmark_bergson run pythia-70m 5M 500K"""
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Execute a single Bergson benchmark run"
    )
    run_parser.add_argument("model", help="Key for the model to benchmark")
    run_parser.add_argument(
        "train_tokens", help="Target training tokens (e.g. 1M, 10M)"
    )
    run_parser.add_argument(
        "eval_tokens", help="Target evaluation tokens (e.g. 100K, 1M)"
    )
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--max-length", type=int, default=512)
    run_parser.add_argument("--max-eval-examples", type=int, default=10)
    run_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    run_parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    run_parser.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    run_parser.add_argument("--run-root", default="runs/bergson-scaling")
    run_parser.add_argument("--run-path")
    run_parser.add_argument("--tag")
    run_parser.add_argument("--notes")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
