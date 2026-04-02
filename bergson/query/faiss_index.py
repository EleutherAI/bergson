import json
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import numpy as np
import psutil
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray
from tqdm import tqdm

from bergson.config import FaissConfig
from bergson.process_grads import precondition_flat_grads

if TYPE_CHECKING:
    import faiss  # noqa: F401  # pyright: ignore[reportMissingImports]


class Index(Protocol):
    """Protocol for searchable FAISS index."""

    def search(self, x: NDArray, k: int) -> tuple[NDArray, NDArray]: ...
    @property
    def ntotal(self) -> int: ...
    def train(self, x: NDArray) -> None: ...
    def add(self, x: NDArray) -> None: ...


def validate_ram(batch: NDArray, max_shard_size: int):
    """
    Estimate the peak RAM footprint for the largest shard and assert it fits.

    Parameters
    ----------
    batch : NDArray
        Sample of gradients used to approximate bytes per gradient vector.
    max_shard_size : int
        Maximum number of gradients that will be grouped into any shard.
    """
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    bytes_per_grad = batch.nbytes / batch.shape[0]
    estimated_shard_ram_gb = (max_shard_size * bytes_per_grad) / (1024**3)

    print(f"Estimated RAM required for largest shard: {estimated_shard_ram_gb} GB")
    print(f"Available RAM: {available_ram_gb} GB")

    assert estimated_shard_ram_gb <= available_ram_gb, (
        "Not enough RAM to build index."
        "Increase the number of shards to reduce peak RAM usage."
    )


def normalize_grads(
    grads: NDArray,
    device: str,
    batch_size: int,
) -> NDArray:
    """
    Normalize gradients to unit norm in batches to keep GPU memory bounded.

    Parameters
    ----------
    grads : NDArray
        Array containing all gradient vectors to normalize.
    device : str
        Device identifier understood by PyTorch (e.g., ``\"cuda:0\"`` or ``\"cpu\"``).
    batch_size : int
        Number of gradients processed per batch before writing back to host memory.
    """
    normalized_grads = np.zeros_like(grads).astype(grads.dtype)

    for i in range(0, grads.shape[0], batch_size):
        batch = torch.from_numpy(grads[i : i + batch_size]).to(device)
        normalized_grads[i : i + batch_size] = (
            (batch / batch.norm(dim=1, keepdim=True)).cpu().numpy()
        )

    return normalized_grads


def gradients_loader(root_dir: Path):
    """
    Yield memory-mapped gradient shards stored under ``root_dir``.

    Handles both single-shard exports (``info.json`` directly under ``root_dir``)
    and multi-shard layouts (directories named ``*shard*``). Each yielded object
    stays memory-mapped so downstream code can stream slices without excessive RAM.
    """

    def load_shard(shard_dir: Path) -> np.memmap:
        with (shard_dir / "info.json").open("r") as f:
            info = json.load(f)

        return np.memmap(
            shard_dir / "gradients.bin",
            dtype=info["dtype"],
            mode="r",
            shape=(info["num_grads"],),
        )

    if (root_dir / "info.json").exists():
        yield load_shard(root_dir)
    else:
        for path in sorted(root_dir.iterdir()):
            if "shard" in path.name:
                yield load_shard(path)


def _require_faiss() -> ModuleType:
    """Import faiss at runtime and raise an error if missing."""

    try:
        import faiss as faiss_module  # type: ignore[import]
    except ImportError as e:
        raise ImportError("Faiss not found, run `pip install faiss-gpu-cu12`") from e

    return faiss_module


def index_to_device(index: Index, device: str) -> Index:
    """
    Move a FAISS index between CPU and GPU devices, optionally sharding.

    Parameters
    ----------
    index : Index
        Existing FAISS index instance.
    device : str
        Destination device string, e.g. ``\"cpu\"``, ``\"cuda\"``, or ``\"cuda:1\"``.
    """
    faiss = _require_faiss()

    if device != "cpu":
        gpus = (
            list(range(torch.cuda.device_count()))
            if device == "cuda"
            else [int(device.split(":")[1])]
        )

        try:
            options = faiss.GpuMultipleClonerOptions()
        except AttributeError as e:
            raise ImportError(
                "Faiss not found, you may have faiss-cpu installed instead "
                "of faiss-gpu with `pip install faiss-gpu-cu12`..."
            ) from e

        options.shard = True
        return faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

    return faiss.index_gpu_to_cpu(index)


class FaissIndex:
    """Sharded FAISS index for efficient nearest neighbor search."""

    shards: list[Index]

    faiss_cfg: FaissConfig

    ordered_modules: list[str]

    def __init__(self, path: Path, device: str, mmap_index: bool):
        faiss = _require_faiss()

        config_path = Path(path) / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"FAISS index configuration not found at {config_path}."
                "Run `FaissIndex.create_index` to create the index."
            )

        with open(config_path) as f:
            config = json.load(f)

        self.unit_norm = config["unit_norm"]
        self.ordered_modules = config["ordered_modules"]
        self.faiss_cfg = FaissConfig(**config["faiss_cfg"])

        shard_paths = sorted(
            (c for c in path.glob("*.faiss") if c.stem.isdigit()),
            key=lambda p: int(p.stem),
        )

        shards = []
        for shard_path in shard_paths:
            shard = faiss.read_index(
                str(shard_path),
                faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
            )
            if not mmap_index:
                shard = index_to_device(shard, device)

            shards.append(shard)

        self.shards = shards

    @staticmethod
    def create_index(
        gradients_path: Path,
        faiss_path: Path,
        faiss_cfg: FaissConfig,
        device: str,
        unit_norm: bool,
        preconditioners: dict[str, torch.Tensor],
    ):
        faiss = _require_faiss()

        print("Building FAISS index...")
        start = perf_counter()

        faiss_path.mkdir(exist_ok=True, parents=True)

        # Write the gradients into an on-disk FAISS index
        if (gradients_path / "info.json").exists():
            info_paths = [gradients_path / "info.json"]
        else:
            info_paths = [
                shard_path / "info.json"
                for shard_path in gradients_path.iterdir()
                if (shard_path / "info.json").exists()
            ]

        assert info_paths, f"No gradient metadata found under {gradients_path}"

        total_grads = sum(
            [json.load(open(info_path))["num_grads"] for info_path in info_paths]
        )

        assert faiss_cfg.num_shards <= total_grads and faiss_cfg.num_shards > 0

        # Set the number of grads for each faiss index shard
        base_shard_size = total_grads // faiss_cfg.num_shards
        remainder = total_grads % faiss_cfg.num_shards
        shard_sizes = [base_shard_size] * (faiss_cfg.num_shards)
        shard_sizes[-1] += remainder

        # Verify all gradients will be consumed
        assert (
            sum(shard_sizes) == total_grads
        ), f"Shard sizes {shard_sizes} don't sum to total_grads {total_grads}"

        dl = gradients_loader(gradients_path)
        buffer: list[NDArray] = []
        buffer_size = 0
        shard_idx = 0

        def build_shard_from_buffer(
            buffer_parts: list[NDArray], shard_idx: int
        ) -> None:
            shard_path = faiss_path / f"{shard_idx}.faiss"
            if shard_path.exists():
                print(f"Shard {shard_idx} already exists, skipping...")
                return
            else:
                print(f"Building shard {shard_idx}...")

            grads_chunk = np.concatenate(buffer_parts, axis=0)
            grads_chunk = precondition_flat_grads(
                torch.from_numpy(grads_chunk), preconditioners, ordered_modules
            ).numpy()
            buffer_parts.clear()

            index = faiss.index_factory(
                grads_chunk.shape[1],
                faiss_cfg.index_factory,
                faiss.METRIC_INNER_PRODUCT,
            )
            index = index_to_device(index, device)
            if faiss_cfg.max_train_examples is not None:
                train_examples = min(faiss_cfg.max_train_examples, grads_chunk.shape[0])
            else:
                train_examples = grads_chunk.shape[0]
            index.train(grads_chunk[:train_examples])
            index.add(grads_chunk)

            del grads_chunk

            index = index_to_device(index, "cpu")
            faiss.write_index(index, str(shard_path))

        ordered_modules = []
        for i, grads in enumerate(tqdm(dl, desc="Loading gradients")):
            if i == 0:
                ordered_modules = list(grads.dtype.names or [])

            grads = structured_to_unstructured(grads)

            if i == 0:
                validate_ram(grads, shard_sizes[-1])

            if unit_norm:
                grads = normalize_grads(grads, device, faiss_cfg.batch_size)

            batch_idx = 0
            batch_size = grads.shape[0]
            while batch_idx < batch_size and shard_idx < faiss_cfg.num_shards:
                remaining_in_shard = shard_sizes[shard_idx] - buffer_size
                take = min(remaining_in_shard, batch_size - batch_idx)

                if take > 0:
                    buffer.append(grads[batch_idx : batch_idx + take])
                    buffer_size += take
                    batch_idx += take

                if buffer_size == shard_sizes[shard_idx]:
                    build_shard_from_buffer(buffer, shard_idx)
                    buffer = []
                    buffer_size = 0
                    shard_idx += 1

            del grads

        # Write the configuration to disk
        with open(faiss_path / "config.yaml", "w") as f:
            json.dump(
                {
                    "faiss_cfg": faiss_cfg.__dict__,
                    "gradients_path": str(gradients_path),
                    "device": device,
                    "unit_norm": unit_norm,
                    "ordered_modules": ordered_modules,
                },
                f,
                indent=2,
            )

        print(f"Built index in {(perf_counter() - start) / 60:.2f} minutes.")

    def search(
        self, q: NDArray, k: int | None, reverse: bool = False
    ) -> tuple[NDArray, NDArray]:
        """
        Perform a nearest neighbor search on the index.

        If fewer than `k` items are found, invalid items will be returned
        with index -1 and a maximum-valued negative distance. If `k` is
        `None`, all available items are returned.

        Args:
            q: Query vectors of shape [num_queries, dim].
            k: Number of results to return per query.
            reverse: If True, return lowest influence examples instead of highest.
        """
        shard_distances = []
        shard_indices = []
        offset = 0

        # For reverse mode with FAISS, we need to fetch all results and then
        # select the lowest scores
        fetch_k = self.ntotal if reverse else k

        for shard in self.shards:
            if hasattr(shard, "nprobe"):
                shard.nprobe = self.faiss_cfg.nprobe  # type: ignore

            distances, indices = shard.search(q, fetch_k or shard.ntotal)

            indices += offset
            offset += shard.ntotal

            shard_distances.append(distances)
            shard_indices.append(indices)

        distances = np.concatenate(shard_distances, axis=1)
        indices = np.concatenate(shard_indices, axis=1)

        # Rerank results overfetched from multiple shards or for reverse mode
        if len(self.shards) > 1 or reverse:
            if reverse:
                # For reverse mode, sort ascending (lowest first) and take first k
                topk_indices = np.argsort(distances, axis=1)[:, : k or self.ntotal]
            else:
                # For normal mode, sort descending (highest first) and take first k
                topk_indices = np.argsort(-distances, axis=1)[:, : k or self.ntotal]
            indices = indices[np.arange(indices.shape[0])[:, None], topk_indices]
            distances = distances[np.arange(distances.shape[0])[:, None], topk_indices]

        return distances, indices

    @property
    def ntotal(self) -> int:
        return sum(shard.ntotal for shard in self.shards)

    @property
    def nprobe(self) -> int:
        return self.faiss_cfg.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        for shard in self.shards:
            if hasattr(shard, "nprobe"):
                shard.nprobe = value  # type: ignore
