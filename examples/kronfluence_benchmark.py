"""Utilities for benchmarking Kronfluence influence analysis scaling."""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.utils import assert_type
from benchmark_common import MODEL_SPECS, ModelSpec, DEFAULT_DATASET

SCHEMA_VERSION = 1
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


class LossTask(Task):
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        input_ids = batch["input_ids"].cuda()
        output = model(input_ids, batch["attention_mask"].cuda())
        loss = F.cross_entropy(
            output.logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten(0, 1)
        )
        return loss

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)


@dataclass
class RunRecord:
    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_examples: int
    eval_examples: int
    dataset: str
    train_split: str
    eval_split: str
    factors_name: str
    scores_name: str
    strategy: str
    use_empirical_fisher: bool
    covariance_max_examples: int
    per_device_batch_size: int
    per_device_query_batch_size: int
    per_device_train_batch_size: int
    amp_dtype: str
    activation_covariance_dtype: str
    gradient_covariance_dtype: str
    per_sample_gradient_dtype: str
    score_dtype: str
    offload_activations_to_cpu: bool
    runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None


def parse_examples(value: str) -> int:
    text = value.strip().lower().replace(",", "")
    if text.endswith("examples"):
        text = text[:-8]
    if not text:
        raise ValueError("empty example spec")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    unit = 1
    if text[-1] in suffixes:
        unit = suffixes[text[-1]]
        text = text[:-1]
    number = float(text)
    return int(number * unit)


def format_examples(examples: int) -> str:
    if examples >= 1_000_000_000:
        value = examples / 1_000_000_000
        suffix = "B"
    elif examples >= 1_000_000:
        value = examples / 1_000_000
        suffix = "M"
    elif examples >= 1_000:
        value = examples / 1_000
        suffix = "K"
    else:
        return str(examples)
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.2f}{suffix}"


def timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_run_path(
    base: Path,
    spec: ModelSpec,
    train_examples: int,
    eval_examples: int,
    tag: str | None,
) -> Path:
    train_label = format_examples(train_examples)
    eval_label = format_examples(eval_examples)
    run_tag = tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = base / spec.key / f"{train_label}-{eval_label}-{run_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_record(path: Path, record: RunRecord) -> None:
    with open(path / "benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)


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


def summarize_records(records: Iterable[RunRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in records])
    if "params" in df.columns:
        df["params_b"] = df["params"] / 1_000_000_000
    return df


def estimate_scaling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    subset = df.query("status == 'success' and runtime_seconds.notnull()")
    if subset.empty:
        raise ValueError("No successful runs with recorded runtime found.")

    X = np.column_stack(
        [
            np.ones(len(subset)),
            np.log(subset["train_examples"].astype(float)),
            np.log(subset["eval_examples"].astype(float)),
            np.log(subset["params"].astype(float)),
        ]
    )
    y = np.log(subset["runtime_seconds"].astype(float))
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    log_pred = X @ coeffs
    subset = subset.copy()
    subset["runtime_pred"] = np.exp(log_pred)

    resid = y - log_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    params = {
        "log_scale": float(coeffs[0]),
        "beta_train_examples": float(coeffs[1]),
        "beta_eval_examples": float(coeffs[2]),
        "beta_params": float(coeffs[3]),
        "scale": float(math.exp(coeffs[0])),
        "r2": float(r2),
        "num_samples": int(len(subset)),
    }
    return subset, params


def plot_scaling(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key, group in df.groupby("model_key"):
        grp = group.sort_values("train_examples")
        ax.plot(
            grp["train_examples"],
            grp["runtime_seconds"],
            marker="o",
            linewidth=1.5,
            label=model_key,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Wall clock time (s)")
    ax.set_title("Kronfluence influence analysis scaling")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Model", fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cmd_run(args: argparse.Namespace) -> None:
    if args.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model}'")
    spec = MODEL_SPECS[args.model]
    train_examples = parse_examples(args.train_examples)
    eval_examples = parse_examples(args.eval_examples)
    print(
        f"Running Kronfluence benchmark for {args.model} with {train_examples} train "
        f"and {eval_examples} eval examples"
    )

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_path = (
        Path(args.run_path).resolve()
        if args.run_path
        else ensure_run_path(run_root, spec, train_examples, eval_examples, args.tag)
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
        train_dataset = train_dataset.select(range(train_examples + eval_examples))

        eval_dataset = train_dataset.select(
            range(train_examples, train_examples + eval_examples)
        )
        train_dataset = train_dataset.select(range(train_examples))

        train_dataset = train_dataset.map(tokenize, batched=True)
        eval_dataset = eval_dataset.map(tokenize, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Set up Kronfluence
        task = LossTask()
        model = prepare_model(model=model, task=task)
        analyzer = Analyzer(analysis_name=args.analysis_name, model=model, task=task)

        # Fit factors
        analyzer.fit_all_factors(
            factors_name=args.factors_name,
            dataset=train_dataset,
            per_device_batch_size=args.per_device_batch_size,
            overwrite_output_dir=True,
            factor_args=FactorArguments(
                strategy=args.strategy,
                use_empirical_fisher=args.use_empirical_fisher,
                covariance_max_examples=args.covariance_max_examples,
                amp_dtype=getattr(torch, args.amp_dtype),
                activation_covariance_dtype=getattr(
                    torch, args.activation_covariance_dtype
                ),
                gradient_covariance_dtype=getattr(
                    torch, args.gradient_covariance_dtype
                ),
            ),
        )

        if args.do_query:
            # Compute pairwise scores
            analyzer.compute_pairwise_scores(
                scores_name=args.scores_name,
                factors_name=args.factors_name,
                query_dataset=eval_dataset,
                train_dataset=train_dataset,
                per_device_query_batch_size=args.per_device_query_batch_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                score_args=ScoreArguments(
                    amp_dtype=getattr(torch, args.amp_dtype),
                    per_sample_gradient_dtype=getattr(
                        torch, args.per_sample_gradient_dtype
                    ),
                    score_dtype=getattr(torch, args.score_dtype),
                    offload_activations_to_cpu=args.offload_activations_to_cpu,
                ),
            )

            # Load scores to verify completion
            # scores = analyzer.load_pairwise_scores(scores_name=args.scores_name)

    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_message = repr(exc)

    runtime = time.perf_counter() - start
    end_wall = timestamp()

    record = RunRecord(
        schema_version=SCHEMA_VERSION,
        status=status,
        model_key=spec.key,
        model_name=spec.hf_id,
        params=spec.params,
        train_examples=train_examples,
        eval_examples=eval_examples,
        dataset=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        factors_name=args.factors_name,
        scores_name=args.scores_name,
        strategy=args.strategy,
        use_empirical_fisher=args.use_empirical_fisher,
        covariance_max_examples=args.covariance_max_examples,
        per_device_batch_size=args.per_device_batch_size,
        per_device_query_batch_size=args.per_device_query_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        amp_dtype=args.amp_dtype,
        activation_covariance_dtype=args.activation_covariance_dtype,
        gradient_covariance_dtype=args.gradient_covariance_dtype,
        per_sample_gradient_dtype=args.per_sample_gradient_dtype,
        score_dtype=args.score_dtype,
        offload_activations_to_cpu=args.offload_activations_to_cpu,
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


def default_train_examples() -> list[str]:
    return ["1K", "10K", "100K", "1M"]


def default_eval_examples() -> list[str]:
    return ["100", "1K", "10K"]


def existing_success_lookup(
    records: Iterable[RunRecord],
) -> set[tuple[str, int, int, str]]:
    return {
        (r.model_key, r.train_examples, r.eval_examples, r.strategy)
        for r in records
        if r.status == "success"
    }


def cmd_commands(args: argparse.Namespace) -> None:
    train_examples = [parse_examples(tok) for tok in args.train_examples]
    eval_examples = [parse_examples(tok) for tok in args.eval_examples]
    models = args.models or list(MODEL_SPECS.keys())

    run_root = Path(args.run_root).resolve()
    records = load_records(run_root)
    seen = existing_success_lookup(records)

    for model_key in models:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{model_key}'")
        for train_ex in train_examples:
            for eval_ex in eval_examples:
                key = (model_key, train_ex, eval_ex, args.strategy)
                if key in seen and not args.include_completed:
                    continue
                pieces = [
                    "python",
                    "examples/kronfluence_benchmark.py",
                    "run",
                    model_key,
                    format_examples(train_ex),
                    format_examples(eval_ex),
                ]
                if args.tag_prefix:
                    pieces.extend(
                        [
                            "--tag",
                            f"{args.tag_prefix}{format_examples(train_ex)}-{format_examples(eval_ex)}",
                        ]
                    )
                if args.do_query:
                    pieces.append("--do-query")
                if args.use_empirical_fisher:
                    pieces.append("--use-empirical-fisher")
                if args.offload_activations_to_cpu:
                    pieces.append("--offload-activations-to-cpu")
                if args.max_length is not None:
                    pieces.extend(["--max-length", str(args.max_length)])
                if args.covariance_max_examples is not None:
                    pieces.extend(
                        ["--covariance-max-examples", str(args.covariance_max_examples)]
                    )
                print(" ".join(pieces))


def cmd_fit(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    records = load_records(run_root)
    df = summarize_records(records)

    if df.empty:
        print("No benchmark records found.")
        return

    # Select dfs where error is None
    df = df.query("error.isna()")

    df_path = Path(args.output_table).resolve()
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(df_path, index=False)
    print(f"Wrote combined table to {df_path}")

    try:
        subset, params = estimate_scaling(df)
    except ValueError as exc:
        print(f"Skipping fit: {exc}")
        return

    fit_path = Path(args.fit_output).resolve()
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    print(f"Saved scaling fit parameters to {fit_path}")

    plot_path = Path(args.plot_output).resolve()
    plot_scaling(subset, plot_path)
    print(f"Saved scaling plot to {plot_path}")


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Key for the model to benchmark")
    parser.add_argument(
        "train_examples", help="Target training examples (e.g. 10K, 1M)"
    )
    parser.add_argument(
        "eval_examples", help="Target evaluation examples (e.g. 100, 1K)"
    )
    parser.add_argument(
        "--strategy", default="diagonal", choices=["diagonal", "kfac", "ekfac"]
    )
    parser.add_argument("--use-empirical-fisher", action="store_true")
    parser.add_argument("--covariance-max-examples", type=int, default=100)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--per-device-query-batch-size", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument(
        "--amp-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--activation-covariance-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--gradient-covariance-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--per-sample-gradient-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--score-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--offload-activations-to-cpu", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--analysis-name", default="kronfluence_benchmark")
    parser.add_argument("--factors-name", default="my_factors")
    parser.add_argument("--scores-name", default="my_scores")
    parser.add_argument("--run-root", default="runs/kronfluence-scaling")
    parser.add_argument("--run-path")
    parser.add_argument("--tag")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--notes")
    parser.add_argument("--do-query", action="store_true")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Kronfluence influence analysis scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Examples:\n"
            "  python examples/kronfluence_benchmark.py run pythia-160m 10K 1K\n"
            "  python examples/kronfluence_benchmark.py commands --tag-prefix exp-\n"
            "  python examples/kronfluence_benchmark.py fit"""
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Execute a single Kronfluence benchmark run"
    )
    add_common_run_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    cmd_parser = subparsers.add_parser("commands", help="List run commands")
    cmd_parser.add_argument(
        "--train-examples", nargs="*", default=default_train_examples()
    )
    cmd_parser.add_argument(
        "--eval-examples", nargs="*", default=default_eval_examples()
    )
    cmd_parser.add_argument("--models", nargs="*")
    cmd_parser.add_argument(
        "--strategy", default="diagonal", choices=["diagonal", "kfac", "ekfac"]
    )
    cmd_parser.add_argument("--use-empirical-fisher", action="store_true")
    cmd_parser.add_argument("--covariance-max-examples", type=int)
    cmd_parser.add_argument("--per-device-batch-size", type=int, default=1)
    cmd_parser.add_argument("--per-device-query-batch-size", type=int, default=1)
    cmd_parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    cmd_parser.add_argument(
        "--amp-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    cmd_parser.add_argument(
        "--activation-covariance-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    cmd_parser.add_argument(
        "--gradient-covariance-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    cmd_parser.add_argument(
        "--per-sample-gradient-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    cmd_parser.add_argument(
        "--score-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    cmd_parser.add_argument("--offload-activations-to-cpu", action="store_true")
    cmd_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    cmd_parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    cmd_parser.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    cmd_parser.add_argument("--analysis-name", default="kronfluence_benchmark")
    cmd_parser.add_argument("--factors-name", default="my_factors")
    cmd_parser.add_argument("--scores-name", default="my_scores")
    cmd_parser.add_argument("--run-root", default="runs/kronfluence-scaling")
    cmd_parser.add_argument("--tag-prefix")
    cmd_parser.add_argument("--include-completed", action="store_true")
    cmd_parser.add_argument("--max-length", type=int)
    cmd_parser.add_argument("--do-query", action="store_true")
    cmd_parser.set_defaults(func=cmd_commands)

    fit_parser = subparsers.add_parser("fit", help="Aggregate results and fit scaling")
    fit_parser.add_argument("--run-root", default="runs/kronfluence-scaling")
    fit_parser.add_argument("--output-table", default="data/kronfluence_scaling.csv")
    fit_parser.add_argument("--fit-output", default="data/kronfluence_scaling_fit.json")
    fit_parser.add_argument("--plot-output", default="figures/kronfluence_scaling.png")
    fit_parser.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
