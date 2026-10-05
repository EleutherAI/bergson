"""Read each rank's own shard of a checkpoint, so no rank holds a whole model."""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_NAME = "model.safetensors.index.json"


def chunk_bounds(n: int, world: int, rank: int) -> tuple[int, int]:
    """Bounds of ``rank``'s piece of dim 0 under ``Shard(0)`` (``torch.chunk``)."""
    per = (n + world - 1) // world
    start = min(rank * per, n)
    return start, min(start + per, n)


def checkpoint_key(path: str) -> str:
    """Checkpoint name for a PEFT-wrapped parameter path."""
    path = path.removeprefix("base_model.model.")
    return path.replace(".base_layer.", ".")


def lora_init(path: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """PEFT's LoRA init, seeded by the path so every rank builds the same one."""
    out = torch.empty(shape, dtype=dtype)
    if ".lora_B" in path or "lora_embedding_B" in path:
        return out.zero_()

    # kaiming_uniform_(a=sqrt(5))
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    gen = torch.Generator().manual_seed(zlib.crc32(path.encode()))
    return out.uniform_(-bound, bound, generator=gen)


class ShardReader:
    """Serves one rank's slice of each parameter from a safetensors checkpoint."""

    def __init__(self, model_path: str | Path, world: int, rank: int):
        self.world = world
        self.rank = rank
        root = Path(model_path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    str(model_path), allow_patterns=["*.safetensors", "*.json"]
                )
            )
        self.root = root

        index = root / INDEX_NAME
        if index.exists():
            self.weight_map: dict[str, str] = json.loads(index.read_text())[
                "weight_map"
            ]
        else:
            files = sorted(root.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors checkpoint under {root}")
            self.weight_map = {}
            for f in files:
                with safe_open(f, framework="pt") as handle:
                    for key in handle.keys():
                        self.weight_map[key] = f.name
        self._open: dict[str, object] = {}

    def _handle(self, filename: str):
        if filename not in self._open:
            self._open[filename] = safe_open(
                self.root / filename, framework="pt"
            ).__enter__()
        return self._open[filename]

    def close(self) -> None:
        for handle in self._open.values():
            handle.__exit__(None, None, None)  # type: ignore[attr-defined]
        self._open.clear()

    def local(self, path: str, param: torch.Tensor) -> torch.Tensor:
        """This rank's dim-0 slice of ``path``, on ``param``'s dtype."""
        lo, hi = chunk_bounds(param.shape[0], self.world, self.rank)
        key = checkpoint_key(path)

        filename = self.weight_map.get(key)
        if filename is None:
            # Adapter weights are not in the checkpoint
            full = lora_init(path, param.shape, param.dtype)
            return full[lo:hi].clone()

        sliced = self._handle(filename).get_slice(key)[lo:hi]  # type: ignore[attr-defined]
        return sliced.to(param.dtype).contiguous()


def materialize_buffers(model, config, device) -> None:
    """Rebuild the buffers (e.g. rotary tables) of a model built on meta."""
    for module in model.modules():
        meta_names = [
            name
            for name, buf in module._buffers.items()
            if buf is not None and buf.is_meta
        ]
        if not meta_names:
            continue
        rebuilt = type(module)(config=config, device=device)
        for name in meta_names:
            source = rebuilt._buffers[name]
            if source is None or source.is_meta:
                raise RuntimeError(
                    f"could not rebuild buffer {name} on {type(module).__name__}"
                )
            module._buffers[name] = source.to(device)
