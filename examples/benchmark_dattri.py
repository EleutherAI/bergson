"""Utilities for benchmarking Dattri influence analysis scaling."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from dattri.algorithm.base import BaseInnerProductAttributor
from dattri.task import AttributionTask
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.utils import assert_type

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
    runtime_seconds: float | None
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
    print(
        f"Running Dattri benchmark for {args.model} with {train_tokens} train "
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

    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize(batch):
            return tokenizer.batch_encode_plus(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )

        # Load datasets
        train_dataset = assert_type(
            Dataset, load_dataset(args.dataset, split=args.train_split)
        )

        # Estimate examples needed based on token count
        # We'll sample until we have enough tokens
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

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        def collate_fn(batch):
            # Dattri expects tuples of (input_ids, labels) where labels = input_ids for language modeling
            # Keep on CPU - dattri will handle device placement
            input_ids = torch.stack([item["input_ids"] for item in batch])
            labels = input_ids.clone()  # For language modeling, labels are the same as input_ids
            return (input_ids, labels)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )

        # Get model device
        model_device = next(model.parameters()).device
        
        def loss_func(params, data_target_pair):
            x, y = data_target_pair
            # Ensure data is on the same device as model
            if isinstance(x, torch.Tensor) and x.device != model_device:
                x = x.to(model_device)
            if isinstance(y, torch.Tensor) and y.device != model_device:
                y = y.to(model_device)
            # functional_call returns a tuple for transformers models, extract logits
            output = torch.func.functional_call(model, params, (x,))
            if isinstance(output, tuple):
                logits = output[0]  # First element is logits
            else:
                logits = output.logits if hasattr(output, 'logits') else output
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss

        # Create task
        task = AttributionTask(
            loss_func=loss_func,
                       model=model,
                       checkpoints=model.state_dict(),
        )

        # Create attributor and cache
        # Try to set device if BaseInnerProductAttributor supports it
        try:
            attributor = BaseInnerProductAttributor(task=task, device="cuda")
        except TypeError:
            # Device parameter not supported, use default
            attributor = BaseInnerProductAttributor(task=task)
        print("Caching training data...")
        attributor.cache(train_loader)

        # Compute attributions
        print("Computing attributions...")
        with torch.no_grad():
            scores = attributor.attribute(train_loader, test_loader)

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
        runtime_seconds=runtime,
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
        description="Benchmark Dattri influence analysis scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Examples:
  python examples/benchmark_dattri.py run pythia-14m 1M 100K
  python examples/benchmark_dattri.py run pythia-70m 5M 500K"""
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Execute a single Dattri benchmark run"
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
    run_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    run_parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    run_parser.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    run_parser.add_argument("--run-root", default="runs/dattri-scaling")
    run_parser.add_argument("--run-path")
    run_parser.add_argument("--tag")
    run_parser.add_argument("--notes")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
