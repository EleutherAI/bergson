import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Generator, Protocol

import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray
from torch import Tensor, nn
from tqdm import tqdm

from .data import load_gradients, load_unstructured_gradients
from .gradients import GradientCollector, GradientProcessor


class TraceResult:
    """Result of a .trace() call."""

    def __init__(self):
        # Should be set by the Attributor after a search
        self._indices: Tensor | None = None
        self._scores: Tensor | None = None

    @property
    def indices(self) -> Tensor:
        """The indices of the top-k examples."""
        if self._indices is None:
            raise ValueError("No indices available. Exit the context manager first.")

        return self._indices

    @property
    def scores(self) -> Tensor:
        """The attribution scores of the top-k examples."""
        if self._scores is None:
            raise ValueError("No scores available. Exit the context manager first.")

        return self._scores


class Index(Protocol):
    """Protocol for any FAISS index that supports search operations."""

    def search(self, x: NDArray, k: int) -> tuple[NDArray, NDArray]: ...
    @property
    def ntotal(self) -> int: ...
    @property
    def nprobe(self) -> int: ...
    @nprobe.setter
    def nprobe(self, value: int) -> None: ...
    def train(self, x: NDArray) -> None: ...
    def add(self, x: NDArray) -> None: ...


@dataclass
class FaissConfig:
    """Configuration for FAISS index."""

    index_factory: str = "IVF1,SQfp16"
    """
    The [FAISS index factory string](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

    Common FAISS factory strings:
        - "IVF1,SQfp16": exact nearest neighbors with brute force search and fp16.
        - "IVF1024,SQfp16": approximate nearest neighbors with 1024 cluster centers
            and fp16. Fast approximate queries are produced at the cost of a slower
            initial index build.
        - "PQ6720": nearest neighbors with vector product quantization to 6720 elements.
            Reduces memory usage at the cost of accuracy.
    """
    mmap_index: bool = False
    """Whether to query the gradients on-disk."""
    max_train_examples: int | None = None
    """The maximum number of examples to train the index on.
        If `None`, all examples will be used."""
    batch_size: int = 1024
    """The batch size for pre-processing gradients."""
    num_shards: int = 1
    """The number of shards to build for an index.
        Using more shards reduces peak RAM usage."""


def normalize_grads(
    grads: NDArray,
    device: str,
    batch_size: int,
) -> NDArray:
    normalized_grads = np.zeros_like(grads).astype(grads.dtype)

    for i in range(0, grads.shape[0], batch_size):
        batch = torch.from_numpy(grads[i : i + batch_size]).to(device)
        normalized_grads[i : i + batch_size] = (
            (batch / batch.norm(dim=1, keepdim=True)).cpu().numpy()
        )

    return normalized_grads


def gradients_loader(root_dir: str):
    def load_shard(shard_dir: str) -> np.memmap:
        with open(os.path.join(shard_dir, "info.json")) as f:
            info = json.load(f)

        if "grad_size" in info:
            return load_unstructured_gradients(shard_dir)

        dtype = info["dtype"]
        num_grads = info["num_grads"]

        return np.memmap(
            os.path.join(shard_dir, "gradients.bin"),
            dtype=dtype,
            mode="r",
            shape=(num_grads,),
        )

    root_path = Path(root_dir)
    if (root_path / "info.json").exists():
        yield load_shard(root_dir)
    else:
        for shard_path in sorted(root_path.iterdir()):
            if shard_path.is_dir():
                yield load_shard(str(shard_path))


def index_to_device(index: Index, device: str) -> Index:
    if device != "cpu":
        gpus = (
            list(range(torch.cuda.device_count()))
            if device == "cuda"
            else [int(device.split(":")[1])]
        )

        options = faiss.GpuMultipleClonerOptions()
        options.shard = True
        return faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

    return faiss.index_gpu_to_cpu(index)


def load_faiss_index(
    index_path: str,
    device: str,
    unit_norm: bool,
    faiss_cfg: FaissConfig,
) -> list[Index]:
    try:
        import faiss
    except ImportError:
        print("Faiss not found, run `pip install faiss-gpu-cu12`...")

    faiss_path = (
        Path("runs/faiss")
        / Path(index_path).stem
        / (
            f"{faiss_cfg.index_factory.replace(',', '_')}"
            f"{'_unit_norm' if unit_norm else ''}"
        )
    )

    if not faiss_path.exists():
        print("Building FAISS index...")
        start = time()

        faiss_path.mkdir(exist_ok=True, parents=True)

        num_dataset_shards = len(list(Path(index_path).iterdir()))
        shards_per_index = math.ceil(num_dataset_shards / faiss_cfg.num_shards)

        dl = gradients_loader(index_path)
        buffer = []
        index_idx = 0

        for grads in tqdm(dl, desc="Loading gradients"):
            if grads.dtype.names is not None:
                grads = structured_to_unstructured(grads)

            if unit_norm:
                grads = normalize_grads(grads, device, faiss_cfg.batch_size)

            buffer.append(grads)

            if len(buffer) == shards_per_index:
                # Build index shard
                print(f"Building shard {index_idx}...")

                grads = np.concatenate(buffer, axis=0)
                buffer = []

                index = faiss.index_factory(
                    grads.shape[1], faiss_cfg.index_factory, faiss.METRIC_INNER_PRODUCT
                )
                index = index_to_device(index, device)
                train_examples = faiss_cfg.max_train_examples or grads.shape[0]
                index.train(grads[:train_examples])
                index.add(grads)

                # Write index to disk
                del grads
                index = index_to_device(index, "cpu")
                faiss.write_index(index, str(faiss_path / f"{index_idx}.faiss"))

                index_idx += 1

        if buffer:
            grads = np.concatenate(buffer, axis=0)
            buffer = []
            index = faiss.index_factory(
                grads.shape[1], faiss_cfg.index_factory, faiss.METRIC_INNER_PRODUCT
            )
            index = index_to_device(index, device)
            index.train(grads)
            index.add(grads)

            # Write index to disk
            del grads
            index = index_to_device(index, "cpu")
            faiss.write_index(index, str(faiss_path / f"{index_idx}.faiss"))

        print(f"Built index in {(time() - start) / 60:.2f} minutes.")
        del buffer, index

    shards = []
    for i in range(faiss_cfg.num_shards):
        shard = faiss.read_index(
            str(faiss_path / f"{i}.faiss"), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        if not faiss_cfg.mmap_index:
            shard = index_to_device(shard, device)

        shards.append(shard)

    return shards


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = False,
        faiss_cfg: FaissConfig | None = None,
    ):
        if faiss_cfg:
            self.faiss_shards = load_faiss_index(
                index_path, device, unit_norm, faiss_cfg
            )
        else:
            mmap = load_gradients(index_path)
            if mmap.dtype.names is not None:
                mmap = structured_to_unstructured(mmap)

            self.grads = torch.tensor(mmap, device=device, dtype=dtype)

            # In-place normalize for numerical stability
            if unit_norm:
                self.grads /= self.grads.norm(dim=1, keepdim=True)

        self.device = device
        self.dtype = dtype
        self.unit_norm = unit_norm
        self.use_faiss = faiss_cfg is not None

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(
        self, queries: Tensor, k: int, nprobe: int = 10
    ) -> tuple[Tensor, Tensor]:
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        If fewer than `k` examples are found FAISS will return items with the index -1
        and the maximum negative distance.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            nprobe: The number of FAISS vector clusters to search if using ANN.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        q = queries

        if self.unit_norm:
            q /= q.norm(dim=1, keepdim=True)

        if not self.use_faiss:
            return torch.topk(q.to(self.device) @ self.grads.mT, k)

        q = q.cpu().numpy()

        shard_distances = []
        shard_indices = []
        offset = 0

        for index in self.faiss_shards:
            index.nprobe = nprobe
            distances, indices = index.search(q, k)

            indices += offset
            offset += index.ntotal

            shard_distances.append(distances)
            shard_indices.append(indices)

        distances = np.concatenate(shard_distances, axis=1)
        indices = np.concatenate(shard_indices, axis=1)

        # Rerank results overfetched from multiple shards
        if len(self.faiss_shards) > 1:
            indices = np.argsort(distances, axis=1)[:, :k]
            distances = distances[np.arange(distances.shape[0])[:, None], indices]

        return torch.from_numpy(distances.squeeze()), torch.from_numpy(
            indices.squeeze()
        )

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int,
        *,
        precondition: bool = False,
        unit_norm: bool = True,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads: list[Tensor] = []
        result = TraceResult()

        def callback(name: str, g: Tensor):
            # Precondition the gradient using Cholesky solve
            if precondition:
                P = self.processor.preconditioners[name]
                g = g.flatten(1).type_as(P)
                g = torch.cholesky_solve(g.mT, P).mT
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads.append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = torch.cat(mod_grads, dim=1)

        if queries.isnan().any():
            raise ValueError("NaN found in queries.")

        if unit_norm:
            queries /= queries.norm(dim=1, keepdim=True)

        result._scores, result._indices = self.search(queries, k)
