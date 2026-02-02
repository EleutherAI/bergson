import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from transformers import PreTrainedModel

from bergson.config import IndexConfig

if TYPE_CHECKING:
    from bergson.collector.gradient_collectors import HookCollectorBase


def _clear_cache() -> None:
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_system_metadata(cfg: IndexConfig) -> Dict[str, Any]:
    """Identify the current hardware and model configuration."""
    gpu_name = "cpu"
    gpu_mem = 0.0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "model": cfg.model,
        "fsdp": cfg.fsdp,
        "precision": getattr(cfg, "precision", "fp32"),
        "projection_dim": getattr(cfg, "projection_dim", 0),
        "reshape_to_square": getattr(cfg, "reshape_to_square", False),
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
    }


def _check_cache(cache_file: Path, current_meta: Dict[str, Any]) -> Optional[int]:
    """Read JSONL file and look for a matching configuration."""
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if all(row.get(k) == v for k, v in current_meta.items()):
                        return row.get("token_batch_size")
                except json.JSONDecodeError:
                    continue
    except Exception:
        return None

    return None


def _append_to_cache(
    cache_file: Path, current_meta: Dict[str, Any], batch_size: int
) -> None:
    """Append a new row to the JSONL cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    entry = current_meta.copy()
    entry["token_batch_size"] = batch_size

    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Cached batch size {batch_size} to {cache_file}")


def _try_validate(
    model: PreTrainedModel, size: int, collector: "HookCollectorBase"
) -> bool:
    """
    Returns True if the batch size fits, False otherwise.
    Wraps the user-provided logic to handle cleanup and error catching.
    """
    _clear_cache()

    # Check model max length constraint immediately
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is not None and size > max_seq_len:
        return False

    try:
        # Create random tokens
        random_tokens = torch.randint(
            0, 10, (1, size), device=model.device, dtype=torch.long
        )

        # Run the collector pass
        with collector:
            loss = model(random_tokens).logits[0, 0, 0].float()
            loss.backward()
            model.zero_grad()

        return True

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError):
        return False
    finally:
        # Ensure gradients/graphs are wiped before next attempt
        model.zero_grad(set_to_none=True)
        _clear_cache()


def determine_batch_size(
    root: Path,
    cfg: IndexConfig,
    model: PreTrainedModel,
    collector: "HookCollectorBase",
    starting_batch_size: int = 8192,
) -> int:
    """
    Finds the largest viable token batch size that fits in memory.

    1. Checks cache.
    2. Performs binary search for max size.
    3. Saves to cache.
    4. Returns optimal size.
    """
    cache_path = root / "batch_size_cache.jsonl"
    metadata = _get_system_metadata(cfg)

    # Check Cache
    cached_size = _check_cache(cache_path, metadata)
    if cached_size is not None:
        print(f"Loaded optimal token_batch_size from cache: {cached_size}")
        return cached_size

    print("Determining optimal batch size via binary search...")

    # Setup Bounds
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos and starting_batch_size > max_pos:
        starting_batch_size = max_pos

    current_size = starting_batch_size
    last_working_size = None

    while True:
        # Safety break
        if current_size < 16:
            if last_working_size is not None:
                current_size = last_working_size
                break
            raise RuntimeError("Could not fit even token_batch_size=16 in memory.")

        print(f"Testing batch size: {current_size}...", end=" ", flush=True)
        is_success = _try_validate(model, current_size, collector)

        if is_success:
            print("✓ Fits")
            last_working_size = current_size

            # Growth Phase: Try next power of 2
            next_size = current_size * 2

            # Stop if we exceed model max length
            if max_pos and next_size > max_pos:
                break

            current_size = next_size
        else:
            print("✗ OOM / Too Large")

            if last_working_size is not None:
                # We were growing and just hit the ceiling,
                # so the last one was the winner.
                current_size = last_working_size
                break
            else:
                # We haven't found a working size yet (Shrink Phase)
                current_size = current_size // 2

    print(f"Largest viable batch size found: {current_size}")

    # Save to Cache
    _append_to_cache(cache_path, metadata, current_size)

    return current_size
