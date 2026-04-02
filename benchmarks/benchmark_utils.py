import json
import platform
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_from_disk

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline

MAX_BENCHMARK_LENGTH = 1024


@dataclass
class HardwareInfo:
    """Structured hardware information for benchmark records."""

    hardware: str
    gpu_name: str | None = None
    num_gpus_available: int | None = None
    gpu_vram_gb: float | None = None


def get_hardware_info() -> str:
    """Get hardware information string."""
    info = get_hardware_details()
    return info.hardware


def get_hardware_details() -> HardwareInfo:
    """Get structured hardware information.

    Uses nvidia-smi for total GPU count (unaffected by
    CUDA_VISIBLE_DEVICES) and torch for GPU name and VRAM.
    """
    gpu_name: str | None = None
    num_gpus_available: int | None = None
    gpu_vram_gb: float | None = None

    # nvidia-smi for total GPU count on machine
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=count,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        if lines:
            parts = lines[0].split(", ")
            if len(parts) >= 3:
                num_gpus_available = int(parts[0])
                gpu_name = parts[1].strip()
                gpu_vram_gb = round(float(parts[2]) / 1024, 1)

    hw_str = platform.node()
    if num_gpus_available and gpu_name:
        hw_str += f" ({num_gpus_available}x {gpu_name})"
    else:
        hw_str += " (unknown)"

    return HardwareInfo(
        hardware=hw_str,
        gpu_name=gpu_name,
        num_gpus_available=num_gpus_available,
        gpu_vram_gb=gpu_vram_gb,
    )


def prepare_benchmark_ds_path():
    benchmark_ds_path = Path("data/EleutherAI/SmolLM2-135M-10B-tokenized")
    if not benchmark_ds_path.exists():
        benchmark_ds_path.mkdir(parents=True, exist_ok=True)

        index_cfg = IndexConfig(
            run_path="data/EleutherAI/SmolLM2-135M-10B",
            token_batch_size=1024,
            data=DataConfig(
                dataset="EleutherAI/SmolLM2-135M-10B",
                split="train",
                truncation=True,
            ),
            auto_batch_size=True,
        )
        ds, _ = setup_data_pipeline(index_cfg)
        ds.save_to_disk(benchmark_ds_path)

        # Count number of tokens in the dataset
        total_tokens = sum(len(tokens) for tokens in ds["input_ids"])
        print(f"Total tokens: {total_tokens}")

    return benchmark_ds_path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "pythia-14m": ModelSpec("pythia-14m", "EleutherAI/pythia-14m", 14_000_000),
    "pythia-70m": ModelSpec("pythia-70m", "EleutherAI/pythia-70m", 70_000_000),
    "pythia-160m": ModelSpec("pythia-160m", "EleutherAI/pythia-160m", 160_000_000),
    # "pythia-410m": ModelSpec("pythia-410m", "EleutherAI/pythia-410m", 410_000_000),
    "pythia-1b": ModelSpec("pythia-1b", "EleutherAI/pythia-1b", 1_000_000_000),
    "pythia-6.9b": ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", 6_900_000_000),
    "pythia-12b": ModelSpec("pythia-12b", "EleutherAI/pythia-12b", 12_000_000_000),
}


def save_record(path: Path, record, filename: str = "benchmark.json") -> None:
    assert is_dataclass(record) and not isinstance(record, type)

    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, "w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)


def get_run_path(
    base: Path,
    spec: ModelSpec,
    train_tokens: int,
    eval_tokens: int,
    eval_sequences: int,
    tag: str | None,
    num_gpus: int = 1,
) -> Path:
    """Create a run directory with a standardized naming convention."""
    train_label = format_tokens(train_tokens)
    eval_label = format_tokens(eval_tokens)
    run_tag = tag or timestamp()
    gpu_label = f"{num_gpus}gpu"
    path = (
        base
        / spec.key
        / f"{train_label}-{eval_label}-{eval_sequences}-{gpu_label}-{run_tag}"
    )
    return path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


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


def extract_gpu_info(hardware_string: str | None) -> str | None:
    """
    Extract just the GPU information from a hardware string.

    Hardware strings are typically in the format: "hostname (NxGPU_NAME)"
    This function returns just the part in parentheses: "NxGPU_NAME"

    Parameters
    ----------
    hardware_string : str | None
        The full hardware string, e.g. "gpu-server-01 (8x NVIDIA A100-SXM4-80GB)"

    Returns
    -------
    str | None
        Just the GPU info part, e.g. "8x NVIDIA A100-SXM4-80GB",
        or None if no valid format
    """
    if not hardware_string:
        return None

    # Look for content within parentheses
    start = hardware_string.find("(")
    end = hardware_string.find(")")

    if start != -1 and end != -1 and end > start:
        return hardware_string[start + 1 : end]

    # If no parentheses found, return the original string
    return hardware_string


def load_benchmark_dataset(
    path: str | Path | None = None,
    min_length: int = MAX_BENCHMARK_LENGTH,
) -> Dataset:
    """
    Load the on-disk tokenized benchmark dataset and filter to sequences >= min_length.

    This ensures all sequences are the same length (max_length) for consistent batching
    and benchmarking.

    Parameters
    ----------
    path : str | Path
        Path to the tokenized dataset on disk.
    min_length : int
        Minimum sequence length to keep. Sequences shorter than this are filtered out.

    Returns
    -------
    Dataset
        Filtered dataset with only sequences >= min_length.
    """
    if path is None:
        path = prepare_benchmark_ds_path()
    path = Path(path)

    print(f"Loading tokenized dataset from {path}...")
    ds = load_from_disk(str(path))

    # Count tokens before filtering
    total_tokens_before = sum(len(tokens) for tokens in ds["input_ids"])
    num_examples_before = len(ds)

    print(
        f"Dataset loaded: {num_examples_before:,} examples, {total_tokens_before:,} "
        "tokens"
    )

    # Filter to only sequences >= min_length
    print(f"Filtering sequences to length >= {min_length}...")
    ds = ds.filter(lambda ex: len(ex["input_ids"]) >= min_length)

    # Count tokens after filtering
    total_tokens_after = sum(len(tokens) for tokens in ds["input_ids"])
    num_examples_after = len(ds)

    num_examples_removed = num_examples_before - num_examples_after
    tokens_removed = total_tokens_before - total_tokens_after

    print("\nFiltered dataset:")
    print(f"  Examples: {num_examples_after:,} (removed {num_examples_removed:,})")
    print(f"  Tokens: {total_tokens_after:,} (removed {tokens_removed:,})")
    print(
        f"  Average length: {total_tokens_after / num_examples_after:.1f}"
        " tokens/example"
    )

    return ds
