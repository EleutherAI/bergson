import dataclasses
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedModel

from bergson.collector.collector import fwd_bwd_factory
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config.config import AttentionConfig, IndexConfig, PreprocessConfig
from bergson.data import _allocate_batches_world, pad_and_tensor
from bergson.gradients import GradientProcessor
from bergson.utils.worker_utils import max_tokens_for_model

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
    ``cfg.auto_batch_size``.

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
        cfg.token_batch_size = found[0]
        print(f"Loaded token_batch_size from cache: {found[0]}")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)
    max_len = max_tokens_for_model(tokenizer, cfg.model, cfg.revision)
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
        cfg.token_batch_size = determine_batch_size(
            cfg, model, ds, collector, lengths, max_len
        )
    finally:
        del collector
        _clear_cache()

    if rank == 0:
        _append_to_cache(CACHE_PATH, key, cfg.token_batch_size)


def determine_batch_size(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    collector: GradientCollector,
    lengths: list[int],
    max_len: int,
) -> int:
    """Find the largest token budget whose batches fit, doubling from the
    longest document and then bisecting."""
    # Documents longer than the budget can't be batched, and a budget past
    # the context length is rejected when the data is tokenized
    lo = max(lengths)
    limit = max(lo, min(max_len, sum(lengths)))
    print(f"Determining token_batch_size in [{lo}, {limit}]...")
    if not _fits(cfg, model, ds, collector, lengths, lo):
        raise RuntimeError(
            f"token_batch_size={lo}, the longest document, does not fit in memory."
        )

    hi = None
    while lo < limit:
        candidate = min(2 * lo, limit)
        if not _fits(cfg, model, ds, collector, lengths, candidate):
            hi = candidate
            break
        lo = candidate

    # lo fits and hi, when set, does not
    while hi is not None and hi - lo > max(256, lo // 16):
        mid = (lo + hi) // 2
        if _fits(cfg, model, ds, collector, lengths, mid):
            lo = mid
        else:
            hi = mid

    print(f"Optimal token_batch_size: {lo}")
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


def _fits(
    cfg: IndexConfig,
    model: PreTrainedModel | PeftModel,
    ds: Dataset,
    collector: GradientCollector,
    lengths: list[int],
    N: int,
) -> bool:
    """Whether every rank runs the probe batches of budget ``N`` in memory."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(model.device)  # type: ignore[attr-defined]
    fwd_bwd = fwd_bwd_factory(cfg)

    print(f"  Testing {N}...", end=" ", flush=True)
    fits = True
    _clear_cache()
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(PROBE_MEMORY_FRACTION, device)
    try:
        for indices in probe_batches(lengths, N, world_size, cfg.max_batch_size):
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


def _check_cache(cache_file: Path, key: dict[str, Any]) -> int | None:
    """The batch size cached under ``key``, the latest if there are several."""
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
                found = row.get("token_batch_size")
    return found


def _append_to_cache(cache_file: Path, key: dict[str, Any], batch_size: int) -> None:
    """Append a new row to the JSONL cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "token_batch_size": batch_size}) + "\n")

    print(f"Cached batch size {batch_size} to {cache_file}")
