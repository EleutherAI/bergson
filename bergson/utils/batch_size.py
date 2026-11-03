import dataclasses
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel

from bergson.collector.collector import fwd_bwd_factory
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config.config import AttentionConfig, IndexConfig, PreprocessConfig
from bergson.data import _allocate_batches_world, pad_and_tensor
from bergson.gradients import GradientProcessor

CACHE_PATH = Path(".cache") / "batch_size_cache.jsonl"

PROBE_MEMORY_FRACTION = 0.9
"""Share of device memory the probe batches may use, leaving the rest for the
fragmentation a full run builds up."""

UNKEYED_FIELDS = {
    "run_path",
    "overwrite",
    "token_batch_size",
    "auto_batch_size",
    "distributed",
    "debug",
    "profile",
}
"""IndexConfig fields that leave a run's memory unchanged."""


def maybe_auto_batch_size(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    processor: GradientProcessor,
    target_modules: set[str] | None,
    attention_cfgs: dict[str, AttentionConfig],
    preprocess_cfg: PreprocessConfig,
) -> None:
    """Set ``cfg.token_batch_size`` to the largest budget whose batches fit, if
    ``cfg.auto_batch_size``, and ``cfg.max_batch_size`` to the most documents
    a batch fits if it isn't set.

    Every rank probes the same batches and the ranks agree on each result, so
    sharded models and multi-node runs search together.
    """
    if not cfg.auto_batch_size:
        return

    lengths = [int(n) for n in ds["length"][:]]
    key = _cache_key(cfg, preprocess_cfg, lengths)
    rank = dist.get_rank() if dist.is_initialized() else 0

    found = [_check_cache(CACHE_PATH, key) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(found, src=0)
    if found[0] is not None:
        cfg.token_batch_size, cfg.max_batch_size = found[0]
        print(
            f"Loaded token_batch_size from cache: {cfg.token_batch_size}, "
            f"max_batch_size: {cfg.max_batch_size}"
        )
        return

    collector = GradientCollector(
        model=model.base_model,  # type: ignore
        cfg=cfg,
        processor=processor,
        target_modules=target_modules,
        data=ds,
        preprocess_cfg=preprocess_cfg,
        attention_cfgs=attention_cfgs,
        filter_modules=cfg.filter_modules,
    )
    try:
        if cfg.max_batch_size is None:
            cfg.max_batch_size = determine_max_batch_size(
                cfg, model, ds, collector, lengths
            )
        cfg.token_batch_size = determine_batch_size(cfg, model, ds, collector, lengths)
    finally:
        del collector
        _clear_cache()

    if rank == 0:
        _append_to_cache(CACHE_PATH, key, cfg.token_batch_size, cfg.max_batch_size)


def determine_batch_size(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    collector: GradientCollector,
    lengths: list[int],
) -> int:
    """Find the largest token budget whose batches fit, doubling from the
    longest document and then bisecting."""
    # Documents longer than the budget can't be batched. Past the context length
    # a batch holds several documents, up to ``cfg.max_batch_size`` of them.
    lo = max(lengths)
    limit = sum(lengths)
    if cfg.max_batch_size is not None:
        limit = min(limit, cfg.max_batch_size * lo)
    limit = max(lo, limit)
    print(f"Determining token_batch_size in [{lo}, {limit}]...")

    def fits(n: int) -> bool:
        print(f"  Testing {n}...", end=" ", flush=True)
        probes = probe_batches(
            lengths, n, _world_size(), max_batch_size=cfg.max_batch_size
        )
        return _fits(cfg, model, ds, collector, probes)

    if not fits(lo):
        raise RuntimeError(
            f"token_batch_size={lo}, the longest document, does not fit in memory."
        )
    lo = _search(fits, lo, limit, tolerance=lambda lo: max(256, lo // 16))

    print(f"Optimal token_batch_size: {lo}")
    return lo


def determine_max_batch_size(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    collector: GradientCollector,
    lengths: list[int],
) -> int | None:
    """Find the most documents a batch fits, from batches of the shortest
    documents, or None if all of them fit together.

    Each document of a batch keeps its own gradients, so a large token budget
    can hold more short documents than fit in memory. Capping the count first
    lets the budget be set by the long documents.
    """
    # Batching skips documents shorter than 2 tokens
    order = [i for i in np.argsort(lengths, kind="stable").tolist() if lengths[i] >= 2]
    limit = len(order)
    print(f"Determining max_batch_size in [1, {limit}]...")

    def fits(n: int) -> bool:
        print(f"  Testing {n} documents...", end=" ", flush=True)
        return _fits(cfg, model, ds, collector, [order[:n]])

    found = _search(fits, 1, limit, tolerance=lambda lo: max(1, lo // 16))
    if found == limit:
        print("All documents fit in one batch, max_batch_size stays unset")
        return None

    print(f"Optimal max_batch_size: {found}")
    return found


def _search(
    fits: Callable[[int], bool], lo: int, limit: int, tolerance: Callable[[int], int]
) -> int:
    """The largest value in ``[lo, limit]`` that fits to within ``tolerance``,
    doubling from ``lo``, which fits, and then bisecting."""
    hi = None
    while lo < limit:
        candidate = min(2 * lo, limit)
        if not fits(candidate):
            hi = candidate
            break
        lo = candidate

    # lo fits and hi, when set, does not
    while hi is not None and hi - lo > tolerance(lo):
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def probe_batches(
    lengths: list[int], N: int, world_size: int, max_batch_size: int | None
) -> list[list[int]]:
    """The batches of budget ``N`` with the most padded tokens, the most
    documents and the most attention entries, over every rank's batches."""
    batches = [
        b
        for rank in _allocate_batches_world(
            lengths, N, world_size, max_batch_size=max_batch_size
        )
        for b in rank
    ]

    def shape(b):
        return len(b), max(lengths[i] for i in b)

    keys = (
        lambda b: shape(b)[0] * shape(b)[1],
        lambda b: shape(b)[0],
        lambda b: shape(b)[0] * shape(b)[1] ** 2,
    )
    return [max(batches, key=key) for key in keys]


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _fits(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    collector: GradientCollector,
    probes: list[list[int]],
) -> bool:
    """Whether every rank runs the ``probes`` batches in memory."""
    device = torch.device(model.device)  # type: ignore[attr-defined]
    fwd_bwd = fwd_bwd_factory(cfg)

    fits = True
    _clear_cache()
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(PROBE_MEMORY_FRACTION, device)
    try:
        for indices in probes:
            batch = ds[indices]
            x, y, _, collection_mask = pad_and_tensor(
                batch["input_ids"],
                labels=batch.get("labels"),
                device=device,
                sync_max_len=False,
            )
            with collector.with_batch(collection_mask):
                losses = fwd_bwd(model, x, y, batch)
            collector.process_batch(indices, losses=losses)
    except torch.cuda.OutOfMemoryError:
        fits = False
    finally:
        collector.mod_grads.clear()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(1.0, device)
        _clear_cache()

    if dist.is_initialized():
        flag = torch.tensor(int(fits), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        fits = bool(flag.item())
    print("fits" if fits else "OOM")
    return fits


def _clear_cache() -> None:
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_key(
    cfg: IndexConfig, preprocess_cfg: PreprocessConfig, lengths: list[int]
) -> dict[str, Any]:
    """Everything a run's memory depends on: its configs, the document lengths,
    the device and the number of ranks."""
    index = {
        k: v for k, v in dataclasses.asdict(cfg).items() if k not in UNKEYED_FIELDS
    }
    gpu_name, gpu_mem = "cpu", 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
    configs = json.dumps(
        [index, dataclasses.asdict(preprocess_cfg)], sort_keys=True, default=str
    )
    return {
        "configs": hashlib.sha256(configs.encode()).hexdigest(),
        "lengths": hashlib.sha256(np.asarray(lengths, np.int64).tobytes()).hexdigest(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_mem,
        "torch": torch.__version__,
    }


def _check_cache(
    cache_file: Path, key: dict[str, Any]
) -> tuple[int, int | None] | None:
    """The token budget and document cap cached under ``key``, the latest if
    there are several."""
    if not cache_file.exists():
        return None

    found = None
    with open(cache_file) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key") == key:
                found = (row["token_batch_size"], row.get("max_batch_size"))
    return found


def _append_to_cache(
    cache_file: Path, key: dict[str, Any], batch_size: int, max_batch_size: int | None
) -> None:
    """Append a new row to the JSONL cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "token_batch_size": batch_size, "max_batch_size": max_batch_size}
    with open(cache_file, "a") as f:
        f.write(json.dumps(row) + "\n")

    print(
        f"Cached token_batch_size {batch_size}, max_batch_size {max_batch_size} "
        f"to {cache_file}"
    )
