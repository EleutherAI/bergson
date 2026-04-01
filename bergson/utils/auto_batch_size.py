import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel

from bergson.config import IndexConfig
from bergson.gradients import GradientProcessor

if TYPE_CHECKING:
    from bergson.collector.gradient_collectors import (
        HookCollectorBase,
    )


def maybe_auto_batch_size(
    cfg: IndexConfig,
    model: PreTrainedModel,
    ds: Dataset | IterableDataset,
    processor: GradientProcessor,
    target_modules: set[str] | None,
    rank: int = 0,
) -> None:
    """Run auto batch size determination if enabled.

    Mutates ``cfg.token_batch_size`` in place. Only rank 0
    runs the search; other ranks wait at a gloo barrier then
    read the cached result.
    """
    if not cfg.auto_batch_size:
        return

    from bergson.collector.gradient_collectors import (
        GradientCollector,
    )

    # Create gloo group before the search so all ranks
    # participate in this collective call together.
    if dist.is_initialized():
        gloo_group = dist.new_group(backend="gloo")
    else:
        gloo_group = None

    if rank == 0:
        # skip_index=True avoids creating a Builder, whose
        # create_index() calls dist.barrier() on the default
        # NCCL group — which would deadlock since only rank 0
        # creates this collector.
        probe_cfg = replace(cfg, skip_index=True)
        cfg.token_batch_size = determine_batch_size(
            root=Path(".cache"),
            cfg=cfg,
            model=model,
            collector=GradientCollector(
                model=model.base_model,
                cfg=probe_cfg,
                processor=processor,
                target_modules=target_modules,
                data=ds,  # type: ignore
            ),
            starting_batch_size=cfg.token_batch_size,
        )

    if gloo_group is not None:
        dist.barrier(group=gloo_group)  # type: ignore
        dist.destroy_process_group(gloo_group)  # type: ignore

        if rank != 0:
            cache_path = Path(".cache") / "batch_size_cache.jsonl"
            metadata = _get_system_metadata(cfg)
            cached = _check_cache(cache_path, metadata)
            assert cached is not None, "Cache missing after barrier"
            cfg.token_batch_size = cached
            print(f"[rank {rank}] Loaded token_batch_size" f" from cache: {cached}")


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
        "precision": cfg.precision,
        "projection_dim": cfg.projection_dim,
        "reshape_to_square": cfg.reshape_to_square,
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
    model: PreTrainedModel,
    token_budget: int,
    collector: "HookCollectorBase",
) -> bool:
    """
    Returns True if the token budget fits, False otherwise.

    The token budget is split into multiple sequences of
    at most max_position_embeddings tokens, matching how
    batches are actually packed at runtime.
    """
    _clear_cache()

    # Worst case VRAM usage is with maximally long sequences due to
    # O(N^2) attention
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is not None and max_seq_len > 0:
        seq_len = min(token_budget, max_seq_len)
    else:
        seq_len = token_budget
    num_seqs = max(1, token_budget // seq_len)

    try:
        input_ids = torch.randint(
            0,
            10,
            (num_seqs, seq_len),
            device=model.device,
            dtype=torch.long,
        )
        labels = torch.randint(
            0,
            10,
            (num_seqs, seq_len),
            device=model.device,
            dtype=torch.long,
        )

        with collector:
            logits = model(input_ids).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss.backward()
            model.zero_grad()

        return True

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError):
        return False
    finally:
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

    Uses an exponential search to find bounds, then binary search
    to refine. Results are cached to a JSONL file.
    """
    cache_path = root / "batch_size_cache.jsonl"
    metadata = _get_system_metadata(cfg)

    cached_size = _check_cache(cache_path, metadata)
    if cached_size is not None:
        print(f"Loaded token_batch_size from cache: {cached_size}")
        return cached_size

    print("Determining optimal batch size...")

    # Phase 1: exponential search to find bounds [lo, hi]
    lo, hi = None, None
    current_size = starting_batch_size

    while current_size >= 16:
        print(f"  Testing {current_size}...", end=" ", flush=True)
        if _try_validate(model, current_size, collector):
            print("fits")
            lo = current_size
            current_size *= 2
        else:
            print("OOM")
            hi = current_size
            if lo is not None:
                break
            current_size //= 2

    if lo is None:
        raise RuntimeError("Could not fit even token_batch_size=16 in memory.")

    # Phase 2: binary search between lo and hi
    if hi is not None:
        while hi - lo > max(256, lo // 16):
            mid = (lo + hi) // 2
            print(f"  Testing {mid}...", end=" ", flush=True)
            if _try_validate(model, mid, collector):
                print("fits")
                lo = mid
            else:
                print("OOM")
                hi = mid

    print(f"Optimal batch size: {lo}")
    _append_to_cache(cache_path, metadata, lo)

    return lo
