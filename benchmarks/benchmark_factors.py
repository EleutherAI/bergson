"""Benchmark factor/preconditioning computation overhead for bergson and kronfluence."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from simple_parsing import ArgumentParser, ConflictResolution, field
from transformers import AutoModelForCausalLM

from benchmarks.benchmark_utils import (
    MAX_BENCHMARK_LENGTH,
    MODEL_SPECS,
    format_tokens,
    get_hardware_details,
    load_benchmark_dataset,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)

SCHEMA_VERSION = 1


def _model_device(model: torch.nn.Module) -> torch.device:
    """Return the CUDA device of a model's first parameter."""
    return next(model.parameters()).device


BERGSON_FACTOR_TYPES = {"normalizer", "preconditioner", "kfac", "ekfac"}
KRONFLUENCE_FACTOR_TYPES = {"diagonal", "kfac", "ekfac"}
DATTRI_FACTOR_TYPES = {"ekfac", "datainf", "arnoldi"}
ALL_FACTOR_TYPES = BERGSON_FACTOR_TYPES | KRONFLUENCE_FACTOR_TYPES | DATTRI_FACTOR_TYPES


@dataclass
class RunConfig:
    """Configuration for a factor computation benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 100K, 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    method: str = field(positional=True)
    """Method to benchmark: bergson, kronfluence, or dattri."""

    factor_type: str = field(positional=True)
    """Factor type. bergson: normalizer/preconditioner/kfac/ekfac.
    kronfluence: diagonal/kfac/ekfac.
    dattri: ekfac/datainf/arnoldi."""

    normalizer: str = "adafactor"
    """Normalizer type for bergson normalizer factor: adafactor or adam."""

    token_batch_size: int = 1024
    """Token batch size for bergson methods."""

    auto_batch_size: bool = False
    """Automatically determine optimal token_batch_size for hardware."""

    per_device_batch_size: int = 1
    """Batch size per device for kronfluence factor computation."""

    projection_dim: int = 16
    """Projection dimension for bergson preconditioner. 0 = no projection."""

    amp_dtype: str = "bfloat16"
    """AMP dtype: float16, bfloat16, or float32."""

    dataset: str = ""
    """Path to pre-tokenized dataset (auto-populated if empty)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    notes: str | None = None
    """Optional notes for the run."""


@dataclass
class RunRecord:
    """Record of a factor computation benchmark run."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    method: str
    factor_type: str
    factor_seconds: float | None
    run_path: str
    notes: str | None
    error: str | None
    peak_memory_mb: float | None = None
    hardware: str | None = None
    gpu_name: str | None = None
    num_gpus_available: int | None = None
    gpu_vram_gb: float | None = None


def load_records(root: Path) -> list[RunRecord]:
    """Load all factor benchmark records from a directory tree."""
    records: list[RunRecord] = []
    for meta in root.rglob("benchmark.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(RunRecord(**payload))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


def _get_factor_run_path(
    base: Path,
    model_key: str,
    train_tokens: int,
    method: str,
    factor_type: str,
    tag: str | None,
) -> Path:
    """Create a run directory path for factor benchmarks."""
    train_label = format_tokens(train_tokens)
    run_tag = tag or timestamp()
    return base / model_key / f"{method}-{factor_type}-{train_label}-{run_tag}"


# ---------------------------------------------------------------------------
# Bergson factor runners
# ---------------------------------------------------------------------------


def _run_bergson_preconditioner(
    run_cfg: RunConfig,
    spec,
    train_tokens: int,
    run_path: Path,
    ds,
) -> tuple[float, float]:
    """Run bergson preconditioner (P^T@P + eigendecomp).

    Returns (seconds, peak_mb).
    """
    from bergson.collector.collector import CollectorComputer
    from bergson.collector.gradient_collectors import GradientCollector
    from bergson.config import DataConfig, IndexConfig
    from bergson.data import allocate_batches
    from bergson.gradients import GradientProcessor
    from bergson.utils.worker_utils import setup_model_and_peft

    proj_dim = run_cfg.projection_dim if run_cfg.projection_dim > 0 else None
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=spec.hf_id,
        data=DataConfig(
            dataset=run_cfg.dataset,
            split="train",
            prompt_column="text",
        ),
        token_batch_size=run_cfg.token_batch_size,
        max_tokens=train_tokens,
        precision="bf16",
        projection_dim=run_cfg.projection_dim,
        skip_preconditioners=False,
        skip_index=True,
    )
    Path(index_cfg.run_path).mkdir(parents=True, exist_ok=True)
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    model, _ = setup_model_and_peft(index_cfg, device_map_auto=True)
    batches = allocate_batches(ds["length"][:], run_cfg.token_batch_size)

    collector = GradientCollector(
        model=model.base_model,  # type: ignore
        processor=GradientProcessor(projection_dim=proj_dim),
        data=ds,
        cfg=index_cfg,
    )
    computer = CollectorComputer(
        model=model,
        data=ds,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )

    device = _model_device(model)
    print(f"Running preconditioner over {len(batches)} batches...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    computer.run_with_collector_hooks(desc="preconditioner")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak_mb


def _run_bergson_kfac(
    run_cfg: RunConfig,
    spec,
    train_tokens: int,
    run_path: Path,
    ds,
    *,
    ev_correction: bool = False,
) -> tuple[float, float]:
    """Run bergson kfac or ekfac. Set ev_correction=True for ekfac."""
    from bergson.collector.collector import CollectorComputer, fwd_bwd_hessian_factory
    from bergson.config import DataConfig, HessianConfig, IndexConfig
    from bergson.data import allocate_batches
    from bergson.hessians.eigenvectors import (
        LambdaCollector,
        compute_eigendecomposition,
    )
    from bergson.hessians.kfac import CovarianceCollector
    from bergson.utils.worker_utils import setup_model_and_peft

    hessian_cfg = HessianConfig(method="kfac", ev_correction=ev_correction)
    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=spec.hf_id,
        data=DataConfig(
            dataset=run_cfg.dataset,
            split="train",
            prompt_column="text",
        ),
        token_batch_size=run_cfg.token_batch_size,
        max_tokens=train_tokens,
        precision="bf16",
        skip_preconditioners=True,
        skip_index=True,
    )
    # Append method subdir like approximate_hessians does
    index_cfg.run_path = index_cfg.run_path + "/kfac"
    Path(index_cfg.run_path).mkdir(parents=True, exist_ok=True)
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    # ekfac requires single-device (matching hessian_worker pattern)
    model, target_modules = setup_model_and_peft(index_cfg, device_map_auto=False)
    batches = allocate_batches(ds["length"][:], run_cfg.token_batch_size)

    collector = CovarianceCollector(
        model=model.base_model,  # type: ignore
        target_modules=target_modules,
        path=str(index_cfg.partial_run_path),
        filter_modules=index_cfg.filter_modules,
        dtype=model.dtype,  # type: ignore[attr-defined]
    )
    computer = CollectorComputer(
        model=model,
        data=ds,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )
    computer.forward_backward = fwd_bwd_hessian_factory(index_cfg, hessian_cfg)

    device = _model_device(model)
    label = "ekfac" if ev_correction else "kfac"
    print(f"Running {label} covariances over {len(batches)} batches...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    computer.run_with_collector_hooks(desc=f"{label} covariances")

    # Load total_processed saved by CollectorComputer
    total_processed = torch.load(
        f"{index_cfg.partial_run_path}/total_processed.pt",
        map_location="cpu",
        weights_only=False,
    )

    print("Computing eigendecomposition...")
    compute_eigendecomposition(
        str(index_cfg.partial_run_path / "activation_sharded"),
        total_processed=total_processed,
    )
    compute_eigendecomposition(
        str(index_cfg.partial_run_path / "gradient_sharded"),
        total_processed=total_processed,
    )

    if ev_correction:
        # Eigenvalue correction pass (EKFAC Eq. 20)
        print("Computing eigenvalue corrections...")
        lambda_collector = LambdaCollector(
            model=model.base_model,  # type: ignore
            target_modules=target_modules,
            path=str(index_cfg.partial_run_path),
            filter_modules=index_cfg.filter_modules,
        )
        lambda_computer = CollectorComputer(
            model=model,
            data=ds,
            collector=lambda_collector,
            batches=batches,
            cfg=index_cfg,
        )
        lambda_computer.forward_backward = fwd_bwd_hessian_factory(
            index_cfg, hessian_cfg
        )
        lambda_computer.run_with_collector_hooks(desc="ekfac eigenvalue correction")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak_mb


# ---------------------------------------------------------------------------
# Kronfluence factor runner
# ---------------------------------------------------------------------------


def _run_kronfluence(
    run_cfg: RunConfig,
    spec,
    train_tokens: int,
    run_path: Path,
    ds,
) -> tuple[float, float]:
    """Run kronfluence factor computation and return (seconds, peak_mb)."""
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments
    from kronfluence.task import Task

    # ds is already subsetted to train_tokens by the caller
    train_dataset = ds

    # Add attention_mask if not present
    if "attention_mask" not in train_dataset.column_names:
        train_dataset = train_dataset.map(
            lambda ex: {"attention_mask": [1] * len(ex["input_ids"])},
        )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.cuda()  # type: ignore

    # Enumerate trackable modules, excluding embed_out (50304-dim causes
    # cusolver failures during eigendecomposition for kfac/ekfac).
    import torch.nn as nn

    tracked_modules = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and "embed_out" not in name
    ]

    class KronTask(Task):
        def get_influence_tracked_modules(self):
            return tracked_modules

        def compute_train_loss(self, batch, model, sample=False):
            input_ids = batch["input_ids"].cuda()
            output = model(input_ids, batch["attention_mask"].cuda())
            return F.cross_entropy(
                output.logits[:, :-1].flatten(0, 1),
                input_ids[:, 1:].flatten(0, 1),
            )

        def compute_measurement(self, batch, model):
            return self.compute_train_loss(batch, model)

    task = KronTask()
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name="factor_benchmark",
        model=model,
        task=task,
    )

    device = _model_device(model)
    print(f"Running kronfluence {run_cfg.factor_type} factors...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    analyzer.fit_all_factors(
        factors_name="factors",
        dataset=train_dataset,
        per_device_batch_size=run_cfg.per_device_batch_size,
        overwrite_output_dir=True,
        factor_args=FactorArguments(
            strategy=run_cfg.factor_type,
            use_empirical_fisher=False,
            amp_dtype=getattr(torch, run_cfg.amp_dtype),
        ),
    )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak_mb


# ---------------------------------------------------------------------------
# Dattri factor runner
# ---------------------------------------------------------------------------


def _run_dattri(
    run_cfg: RunConfig,
    spec,
    train_tokens: int,
    run_path: Path,
    ds,
) -> tuple[float, float]:
    """Run dattri factor computation (cache step) and return (seconds, peak_mb)."""
    import torch.nn as nn
    from dattri.algorithm.influence_function import (
        IFAttributorArnoldi,
        IFAttributorDataInf,
        IFAttributorEKFAC,
    )
    from dattri.task import AttributionTask

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.cuda()  # type: ignore

    # Add attention_mask if missing
    if "attention_mask" not in ds.column_names:
        ds = ds.map(lambda ex: {"attention_mask": [1] * len(ex["input_ids"])})
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model_device = next(model.parameters()).device

    def loss_func(params, data_target_pair):
        x, y = data_target_pair
        if isinstance(x, torch.Tensor) and x.device != model_device:
            x = x.to(model_device)
        if isinstance(y, torch.Tensor) and y.device != model_device:
            y = y.to(model_device)
        output = torch.func.functional_call(model, params, (x,))
        logits = output.logits if hasattr(output, "logits") else output
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = y[:, 1:].contiguous()
        return nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    task = AttributionTask(
        loss_func=loss_func,
        model=model,
        checkpoints=model.state_dict(),
    )

    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        padded_input_ids = []
        for item in batch:
            ids = item["input_ids"]
            if len(ids) < max_len:
                padding = torch.zeros(max_len - len(ids), dtype=ids.dtype)
                ids = torch.cat([ids, padding])
            padded_input_ids.append(ids)
        input_ids = torch.stack(padded_input_ids)
        labels = input_ids.clone()
        # Return list (not tuple) — dattri arnoldi mutates this in-place
        return [input_ids, labels]

    train_loader = torch.utils.data.DataLoader(
        ds,  # type: ignore
        batch_size=run_cfg.per_device_batch_size,
        collate_fn=collate_fn,
    )

    if run_cfg.factor_type == "ekfac":
        attributor = IFAttributorEKFAC(task=task, device="cuda", damping=1e-3)
    elif run_cfg.factor_type == "datainf":
        attributor = IFAttributorDataInf(task=task, device="cuda", regularization=0.0)
    elif run_cfg.factor_type == "arnoldi":
        attributor = IFAttributorArnoldi(task=task, device="cuda", proj_dim=100)
    else:
        raise ValueError(f"Unknown dattri factor type: {run_cfg.factor_type}")

    device = _model_device(model)
    print(
        f"Running dattri {run_cfg.factor_type} cache "
        f"over {len(train_loader)} batches..."
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    attributor.cache(train_loader)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak_mb


# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Execute a single factor computation benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark with fair timing (model/data loading excluded)."""
        run_cfg = self.run_cfg

        # Validate method and factor_type
        method_factor_types = {
            "bergson": BERGSON_FACTOR_TYPES,
            "kronfluence": KRONFLUENCE_FACTOR_TYPES,
            "dattri": DATTRI_FACTOR_TYPES,
        }
        if run_cfg.method not in method_factor_types:
            raise ValueError(
                f"Unknown method '{run_cfg.method}'. "
                f"Use one of: {sorted(method_factor_types)}"
            )
        valid_types = method_factor_types[run_cfg.method]
        if run_cfg.factor_type not in valid_types:
            raise ValueError(
                f"Unknown {run_cfg.method} factor type '{run_cfg.factor_type}'. "
                f"Choose from: {sorted(valid_types)}"
            )

        if not run_cfg.dataset:
            run_cfg.dataset = str(prepare_benchmark_ds_path())

        if run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{run_cfg.model}'")

        spec = MODEL_SPECS[run_cfg.model]
        train_tokens = parse_tokens(run_cfg.train_tokens)

        print(
            f"Factor benchmark: {run_cfg.method}/{run_cfg.factor_type} "
            f"for {run_cfg.model} with {format_tokens(train_tokens)} tokens"
        )

        run_root = Path(run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        run_path = _get_factor_run_path(
            run_root,
            spec.key,
            train_tokens,
            run_cfg.method,
            run_cfg.factor_type,
            run_cfg.tag,
        )

        # --- UNTIMED: load dataset and subset to train_tokens ---
        ds = load_benchmark_dataset(run_cfg.dataset)

        seq_len = MAX_BENCHMARK_LENGTH
        train_examples = max(1, train_tokens // seq_len)
        if len(ds) < train_examples:
            raise ValueError(
                f"Dataset has {len(ds)} examples but need {train_examples} "
                f"for {format_tokens(train_tokens)} tokens"
            )
        ds = ds.select(range(train_examples))
        print(
            f"Using {train_examples} examples "
            f"(~{train_examples * seq_len:,} tokens)"
        )

        # --- TIMED: factor computation ---
        status = "success"
        error_message: str | None = None
        factor_time: float | None = None
        peak_memory_mb: float | None = None

        try:
            result: tuple[float, float] | None = None
            if run_cfg.method == "bergson":
                if run_cfg.factor_type == "preconditioner":
                    result = _run_bergson_preconditioner(
                        run_cfg, spec, train_tokens, run_path, ds
                    )
                elif run_cfg.factor_type == "kfac":
                    result = _run_bergson_kfac(
                        run_cfg, spec, train_tokens, run_path, ds
                    )
                elif run_cfg.factor_type == "ekfac":
                    result = _run_bergson_kfac(
                        run_cfg,
                        spec,
                        train_tokens,
                        run_path,
                        ds,
                        ev_correction=True,
                    )
            elif run_cfg.method == "kronfluence":
                result = _run_kronfluence(run_cfg, spec, train_tokens, run_path, ds)
            elif run_cfg.method == "dattri":
                result = _run_dattri(run_cfg, spec, train_tokens, run_path, ds)

            if result is not None:
                factor_time, peak_memory_mb = result
                print(
                    f"Factor computation completed in "
                    f"{factor_time:.2f} seconds"
                    f" (peak VRAM: {peak_memory_mb:.0f} MB)"
                )

        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = repr(exc)
            print(f"Error: {error_message}", file=sys.stderr)

        record = RunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            method=run_cfg.method,
            factor_type=run_cfg.factor_type,
            factor_seconds=factor_time,
            run_path=str(run_path),
            notes=run_cfg.notes,
            error=error_message,
            peak_memory_mb=peak_memory_mb,
            **vars(get_hardware_details()),
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Main:
    """Benchmark factor/preconditioning computation overhead."""

    command: Run

    def execute(self) -> None:
        self.command.execute()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark factor/preconditioning computation overhead",
    )
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
