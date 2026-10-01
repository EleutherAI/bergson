"""Build a sharded model without any rank holding a whole parameter.

``distribute_tensor`` scatters from a source rank, so every rank needs the full
tensor in memory before it is split: a node therefore needs ``nproc_per_node``
full replicas just to start, which rules out models much larger than host
memory divided by the local rank count. Reading each rank's own slice out of the
checkpoint instead keeps peak memory at one slice, and reads ``world_size``
times fewer bytes off disk.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_NAME = "model.safetensors.index.json"


def chunk_bounds(n: int, world: int, rank: int) -> tuple[int, int]:
    """Half-open bounds of ``rank``'s piece of dim 0 under ``Shard(0)``.

    Matches ``torch.chunk`` semantics, which is what DTensor uses, including
    the empty shards a rank gets when ``n < world``.
    """
    per = (n + world - 1) // world
    start = min(rank * per, n)
    return start, min(start + per, n)


def checkpoint_key(path: str) -> str:
    """Checkpoint name for a PEFT-wrapped parameter path."""
    path = path.removeprefix("base_model.model.")
    return path.replace(".base_layer.", ".")


def lora_init(path: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """PEFT's adapter init, reproduced identically on every rank.

    Adapter weights are not in the checkpoint, so each rank builds the whole
    (small) tensor and slices it. The seed comes from the parameter's path via
    crc32 rather than ``hash``, which is salted per process and would hand
    different ranks different values.
    """
    out = torch.empty(shape, dtype=dtype)
    if ".lora_B" in path or "lora_embedding_B" in path:
        return out.zero_()

    # kaiming_uniform_(a=sqrt(5)) on a (out, in) weight reduces to this bound.
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
            # Single-file checkpoints have no index.
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
            # Adapter weights are created, not loaded.
            full = lora_init(path, param.shape, param.dtype)
            return full[lo:hi].clone()

        # get_slice reads only the requested rows off disk.
        sliced = self._handle(filename).get_slice(key)[lo:hi]  # type: ignore[attr-defined]
        return sliced.to(param.dtype).contiguous()


def materialize_buffers(model, config, device) -> None:
    """Give every meta buffer real values.

    A model built on the meta device has no buffer contents, and the rotary
    tables live there, so the first forward would read empty storage. Each
    owning module is rebuilt on ``device`` from the same config, which is how
    those buffers were computed in the first place.
    """
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
