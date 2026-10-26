"""Utilities for benchmarking Kronfluence influence analysis scaling."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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
from simple_parsing import ArgumentParser, field
from transformers import AutoModelForCausalLM

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_info,
    get_run_path,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)
from bergson.utils.utils import assert_type

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
class RunConfig:
    """Configuration for a Kronfluence benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark."""

    train_examples: str = field(positional=True)
    """Target training examples (e.g. 10K, 1M)."""

    eval_examples: str = field(positional=True)
    """Target evaluation examples (e.g. 100, 1K)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    strategy: str = "diagonal"
    """Strategy to use: diagonal, kfac, or ekfac."""

    use_empirical_fisher: bool = False
    """Use empirical Fisher information matrix."""

    covariance_max_examples: int = 100
    """Maximum examples for covariance computation."""

    per_device_batch_size: int = 1
    """Batch size per device."""

    per_device_query_batch_size: int = 1
    """Query batch size per device."""

    per_device_train_batch_size: int = 1
    """Training batch size per device."""

    amp_dtype: str = "bfloat16"
    """AMP dtype: float16, bfloat16, or float32."""

    activation_covariance_dtype: str = "bfloat16"
    """Activation covariance dtype: float16, bfloat16, or float32."""

    gradient_covariance_dtype: str = "bfloat16"
    """Gradient covariance dtype: float16, bfloat16, or float32."""

    per_sample_gradient_dtype: str = "bfloat16"
    """Per-sample gradient dtype: float16, bfloat16, or float32."""

    score_dtype: str = "bfloat16"
    """Score dtype: float16, bfloat16, or float32."""

    offload_activations_to_cpu: bool = False
    """Offload activations to CPU."""

    dataset: str = ""
    """Dataset to use."""

    train_split: str = DEFAULT_TRAIN_SPLIT
    """Dataset split for training."""

    eval_split: str = DEFAULT_EVAL_SPLIT
    """Dataset split for evaluation."""

    analysis_name: str = "kronfluence_benchmark"
    """Analysis name."""

    factors_name: str = "my_factors"
    """Factors name."""

    scores_name: str = "my_scores"
    """Scores name."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    max_length: int | None = None
    """Maximum sequence length."""

    notes: str | None = None
    """Optional notes for the run."""

    do_query: bool = False
    """Compute pairwise scores."""


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
    hardware: str


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


@dataclass
class Run:
    """Execute a single Kronfluence benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark."""
        if not self.run_cfg.dataset:
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")
        spec = MODEL_SPECS[self.run_cfg.model]
        train_examples = parse_examples(self.run_cfg.train_examples)
        eval_examples = parse_examples(self.run_cfg.eval_examples)
        print(
            f"Running Kronfluence benchmark for {self.run_cfg.model} with "
            f"{train_examples} train and {eval_examples} eval examples"
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        run_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_examples,
                eval_examples,
                eval_examples,
                self.run_cfg.tag,
            )
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
            model.cuda()  # type: ignore

            # Load datasets
            train_dataset = assert_type(
                Dataset,
                load_dataset(self.run_cfg.dataset, split=self.run_cfg.train_split),
            )
            train_dataset = train_dataset.select(range(train_examples + eval_examples))

            eval_dataset = train_dataset.select(
                range(train_examples, train_examples + eval_examples)
            )
            train_dataset = train_dataset.select(range(train_examples))

            train_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask"]
            )
            eval_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask"]
            )

            # Set up Kronfluence
            task = LossTask()
            model = prepare_model(model=model, task=task)
            analyzer = Analyzer(
                analysis_name=self.run_cfg.analysis_name, model=model, task=task
            )

            # Fit factors
            analyzer.fit_all_factors(
                factors_name=self.run_cfg.factors_name,
                dataset=train_dataset,
                per_device_batch_size=self.run_cfg.per_device_batch_size,
                overwrite_output_dir=True,
                factor_args=FactorArguments(
                    strategy=self.run_cfg.strategy,
                    use_empirical_fisher=self.run_cfg.use_empirical_fisher,
                    covariance_max_examples=self.run_cfg.covariance_max_examples,
                    amp_dtype=getattr(torch, self.run_cfg.amp_dtype),
                    activation_covariance_dtype=getattr(
                        torch, self.run_cfg.activation_covariance_dtype
                    ),
                    gradient_covariance_dtype=getattr(
                        torch, self.run_cfg.gradient_covariance_dtype
                    ),
                ),
            )

            if self.run_cfg.do_query:
                # Compute pairwise scores
                analyzer.compute_pairwise_scores(
                    scores_name=self.run_cfg.scores_name,
                    factors_name=self.run_cfg.factors_name,
                    query_dataset=eval_dataset,
                    train_dataset=train_dataset,
                    per_device_query_batch_size=self.run_cfg.per_device_query_batch_size,
                    per_device_train_batch_size=self.run_cfg.per_device_train_batch_size,
                    score_args=ScoreArguments(
                        amp_dtype=getattr(torch, self.run_cfg.amp_dtype),
                        per_sample_gradient_dtype=getattr(
                            torch, self.run_cfg.per_sample_gradient_dtype
                        ),
                        score_dtype=getattr(torch, self.run_cfg.score_dtype),
                        offload_activations_to_cpu=self.run_cfg.offload_activations_to_cpu,
                    ),
                )

                # Load scores to verify completion
                # scores = analyzer.load_pairwise_scores(
                #     scores_name=self.run_cfg.scores_name
                # )

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
            dataset=self.run_cfg.dataset,
            train_split=self.run_cfg.train_split,
            eval_split=self.run_cfg.eval_split,
            factors_name=self.run_cfg.factors_name,
            scores_name=self.run_cfg.scores_name,
            strategy=self.run_cfg.strategy,
            use_empirical_fisher=self.run_cfg.use_empirical_fisher,
            covariance_max_examples=self.run_cfg.covariance_max_examples,
            per_device_batch_size=self.run_cfg.per_device_batch_size,
            per_device_query_batch_size=self.run_cfg.per_device_query_batch_size,
            per_device_train_batch_size=self.run_cfg.per_device_train_batch_size,
            amp_dtype=self.run_cfg.amp_dtype,
            activation_covariance_dtype=self.run_cfg.activation_covariance_dtype,
            gradient_covariance_dtype=self.run_cfg.gradient_covariance_dtype,
            per_sample_gradient_dtype=self.run_cfg.per_sample_gradient_dtype,
            score_dtype=self.run_cfg.score_dtype,
            offload_activations_to_cpu=self.run_cfg.offload_activations_to_cpu,
            runtime_seconds=runtime,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(run_path),
            notes=self.run_cfg.notes,
            error=error_message,
            hardware=get_hardware_info(),
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


@dataclass
class CommandsConfig:
    """Configuration for generating run commands."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    train_examples: list[str] = field(
        default_factory=lambda: ["1K", "10K", "100K", "1M"]
    )
    """Training examples to test (e.g. 10K, 1M)."""

    eval_examples: list[str] = field(default_factory=lambda: ["100", "1K", "10K"])
    """Evaluation examples to test (e.g. 100, 1K)."""

    models: list[str] | None = None
    """Models to benchmark (default: all models)."""

    strategy: str = "diagonal"
    """Strategy to use: diagonal, kfac, or ekfac."""

    use_empirical_fisher: bool = False
    """Use empirical Fisher information matrix."""

    covariance_max_examples: int | None = None
    """Maximum examples for covariance computation."""

    per_device_batch_size: int = 1
    """Batch size per device."""

    per_device_query_batch_size: int = 1
    """Query batch size per device."""

    per_device_train_batch_size: int = 1
    """Training batch size per device."""

    amp_dtype: str = "bfloat16"
    """AMP dtype: float16, bfloat16, or float32."""

    activation_covariance_dtype: str = "bfloat16"
    """Activation covariance dtype: float16, bfloat16, or float32."""

    gradient_covariance_dtype: str = "bfloat16"
    """Gradient covariance dtype: float16, bfloat16, or float32."""

    per_sample_gradient_dtype: str = "bfloat16"
    """Per-sample gradient dtype: float16, bfloat16, or float32."""

    score_dtype: str = "bfloat16"
    """Score dtype: float16, bfloat16, or float32."""

    offload_activations_to_cpu: bool = False
    """Offload activations to CPU."""

    dataset: str = ""
    """Dataset to use."""

    train_split: str = DEFAULT_TRAIN_SPLIT
    """Dataset split for training."""

    eval_split: str = DEFAULT_EVAL_SPLIT
    """Dataset split for evaluation."""

    analysis_name: str = "kronfluence_benchmark"
    """Analysis name."""

    factors_name: str = "my_factors"
    """Factors name."""

    scores_name: str = "my_scores"
    """Scores name."""

    tag_prefix: str | None = None
    """Prefix for run tags."""

    include_completed: bool = False
    """Include already completed runs."""

    max_length: int | None = None
    """Maximum sequence length."""

    do_query: bool = False
    """Compute pairwise scores."""


@dataclass
class Commands:
    """Generate run commands for benchmarking."""

    commands_cfg: CommandsConfig

    def execute(self) -> None:
        """Generate commands."""
        train_examples = [
            parse_examples(tok) for tok in self.commands_cfg.train_examples
        ]
        eval_examples = [parse_examples(tok) for tok in self.commands_cfg.eval_examples]
        models = self.commands_cfg.models or list(MODEL_SPECS.keys())

        run_root = Path(self.commands_cfg.run_root).resolve()
        records = load_records(run_root)
        seen = existing_success_lookup(records)

        for model_key in models:
            if model_key not in MODEL_SPECS:
                raise ValueError(f"Unknown model '{model_key}'")
            for train_ex in train_examples:
                for eval_ex in eval_examples:
                    key = (model_key, train_ex, eval_ex, self.commands_cfg.strategy)
                    if key in seen and not self.commands_cfg.include_completed:
                        continue
                    pieces = [
                        "python",
                        "benchmarks/kronfluence_benchmark.py",
                        "run",
                        model_key,
                        format_examples(train_ex),
                        format_examples(eval_ex),
                        self.commands_cfg.run_root,
                    ]
                    if self.commands_cfg.tag_prefix:
                        pieces.extend(
                            [
                                "--tag",
                                f"{self.commands_cfg.tag_prefix}{format_examples(train_ex)}-{format_examples(eval_ex)}",
                            ]
                        )
                    if self.commands_cfg.do_query:
                        pieces.append("--do_query")
                    if self.commands_cfg.use_empirical_fisher:
                        pieces.append("--use_empirical_fisher")
                    if self.commands_cfg.offload_activations_to_cpu:
                        pieces.append("--offload_activations_to_cpu")
                    if self.commands_cfg.max_length is not None:
                        pieces.extend(
                            ["--max_length", str(self.commands_cfg.max_length)]
                        )
                    if self.commands_cfg.covariance_max_examples is not None:
                        pieces.extend(
                            [
                                "--covariance_max_examples",
                                str(self.commands_cfg.covariance_max_examples),
                            ]
                        )
                    print(" ".join(pieces))


@dataclass
class FitConfig:
    """Configuration for fitting scaling results."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    output_table: str = field(positional=True)
    """Path to save combined results table."""

    fit_output: str = field(positional=True)
    """Path to save scaling fit parameters."""

    plot_output: str = field(positional=True)
    """Path to save scaling plot."""


@dataclass
class Fit:
    """Aggregate results and fit scaling."""

    fit_cfg: FitConfig

    def execute(self) -> None:
        """Fit scaling results."""
        run_root = Path(self.fit_cfg.run_root).resolve()
        records = load_records(run_root)
        df = summarize_records(records)

        if df.empty:
            print("No benchmark records found.")
            return

        # Select dfs where error is None
        df = df.query("error.isna()")

        df_path = Path(self.fit_cfg.output_table).resolve()
        df_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(df_path, index=False)
        print(f"Wrote combined table to {df_path}")

        try:
            subset, params = estimate_scaling(df)
        except ValueError as exc:
            print(f"Skipping fit: {exc}")
            return

        fit_path = Path(self.fit_cfg.fit_output).resolve()
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fit_path, "w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2)
        print(f"Saved scaling fit parameters to {fit_path}")

        plot_path = Path(self.fit_cfg.plot_output).resolve()
        plot_scaling(subset, plot_path)
        print(f"Saved scaling plot to {plot_path}")


@dataclass
class Main:
    """Benchmark Kronfluence influence analysis scaling."""

    command: Run | Commands | Fit

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(
        description="Benchmark Kronfluence influence analysis scaling"
    )
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
