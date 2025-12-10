# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""Compute EKFAC ground truth for testing.

This script computes ground truth covariance matrices, eigenvectors, and eigenvalue
corrections for EKFAC on a single GPU without sharding. By specifying the number of
workers we can simulate distributed computation.
"""

import argparse
import builtins
import gc
import json
import os
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from ground_truth.collector import (
    GroundTruthAmortizedLambdaCollector,
    GroundTruthCovarianceCollector,
)
from safetensors.torch import load_file, save_file
from test_utils import set_all_seeds
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from bergson.data import DataConfig, IndexConfig, Precision, pad_and_tensor, tokenize
from bergson.hessians.utils import TensorDict
from bergson.utils import assert_type, get_device

Batches = list[list[list[int]]]

# %% [markdown]
# ## -1. Helper functions


# %%
def allocate_batches_test(
    doc_lengths: list[int], N: int, workers: Optional[int] = None
) -> Batches:
    """
    Modification of allocate_batches to return a flat list of batches for testing.

    Allocate documents into batches that are then distributed evenly across
    a fixed number of workers.

    Parameters
    ----------
    doc_lengths : Sequence[int]
        Length (in tokens) of each document.
    workers : int
        Number of parallel workers ( 1 ≤ workers ≤ 8).
    N : int
        Hard memory budget per *batch*, expressed as
        ``max(length in batch) * (# docs in batch) ≤ N``.

    Returns
    -------
    list[list[list[int]]]
        ``allocation[w][b]`` is the list of document indices that belong to the
        *b-th* batch assigned to worker ``w``.
    """
    if workers is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        world_size = workers

    if not doc_lengths:
        raise RuntimeError("Empty document list.")
    if max(doc_lengths) > N:
        raise RuntimeError("At least one document is too long for the budget N.")

    # First-fit decreasing (FFD) bin packing
    docs_sorted = sorted(enumerate(doc_lengths), key=lambda x: x[1], reverse=True)
    batches: list[list[int]] = []
    batch_meta = []

    for idx, length in docs_sorted:
        placed = False
        for j, (mx, sz) in enumerate(batch_meta):
            new_mx = max(mx, length)
            new_sz = sz + 1
            if new_mx * new_sz <= N:
                batches[j].append(idx)
                batch_meta[j] = (new_mx, new_sz)
                placed = True
                break

        if not placed:
            batches.append([idx])
            batch_meta.append((length, 1))

    # Ensure every worker gets ≥ 1 batch
    if len(batches) < world_size:
        batches.sort(key=len, reverse=True)
        while len(batches) < world_size:
            big = batches.pop(0)
            if len(big) == 1:
                raise RuntimeError(
                    "Not enough documents to give each worker at least one batch."
                )
            batches.append([big.pop()])
            batches.append(big)

    # Pad the number of batches to a multiple of `workers`
    k = -(-len(batches) // world_size)
    target_batches = world_size * k

    i = 0
    while len(batches) < target_batches:
        batch = batches[i % len(batches)]
        if len(batch) == 1:
            i += 1
            continue
        batches.append([batch.pop()])
        i += 1

    assert len(batches) == target_batches
    assert all(
        max(doc_lengths[i] for i in batch) * len(batch) <= N for batch in batches
    )

    # Round-robin assignment to workers
    allocation: Batches = [[] for _ in range(world_size)]
    for b_idx, batch in enumerate(batches):
        allocation[b_idx % world_size].append(batch)

    assert len({len(b) for b in allocation}) == 1
    return allocation


# %% [markdown]
# ## 0. Hyperparameters


# %%
def parse_config() -> tuple[Precision, str, str, int, bool]:
    """Parse command-line arguments or return defaults."""
    parser = argparse.ArgumentParser(
        description="Compute EKFAC ground truth for testing"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "int4", "int8"],
        help="Model precision (default: fp32)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=os.path.join(
            os.getcwd(), "test_files", "pile_100_examples", "ground_truth"
        ),
        help="Output directory for ground truth results (default: test_files/pile_100_examples/ground_truth)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="EleutherAI/Pythia-14m",
        help="Model name to use (default: EleutherAI/Pythia-14m)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of workers for simulated distributed computation (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing ground truth data and config",
    )

    # For interactive mode (Jupyter/IPython) or no args, use defaults
    if len(sys.argv) > 1 and not hasattr(builtins, "__IPYTHON__"):
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    # Set random seeds for reproducibility
    set_all_seeds(42)

    return (
        args.precision,
        args.output_dir,
        args.model_name,
        args.world_size,
        args.overwrite,
    )


if __name__ == "__main__" or TYPE_CHECKING:
    precision, test_path, model_name, world_size_arg, overwrite_arg = parse_config()


# %%
def setup_paths_and_config(
    precision: Precision,
    test_path: str,
    model_name: str,
    world_size: int,
    overwrite: bool = False,
) -> tuple[IndexConfig, int, torch.device, Any, torch.dtype]:
    """Setup paths and configuration object."""
    os.makedirs(test_path, exist_ok=True)

    current_path = os.getcwd()
    parent_path = os.path.join(current_path, "test_files", "pile_100_examples")

    # Configuration
    cfg = IndexConfig(run_path="")
    cfg.model = model_name
    cfg.precision = precision
    cfg.fsdp = False
    cfg.data = DataConfig(dataset=os.path.join(parent_path, "data"))

    # model_max_length is limited in some models like `roneneldan/TinyStories-1M`
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if (
        hasattr(tokenizer, "model_max_length")
        and tokenizer.model_max_length < cfg.token_batch_size
    ):
        print(
            f"Warning: Got --token-batch-size {cfg.token_batch_size} but {model_name} only supports up to {tokenizer.model_max_length}"
        )
        cfg.token_batch_size = tokenizer.model_max_length

    data_str = cfg.data.dataset

    # Create pile-100 dataset if it doesn't exist
    if not os.path.exists(data_str):
        full_dataset = load_dataset("NeelNanda/pile-10k", split="train")
        assert isinstance(full_dataset, Dataset), "Expected Dataset, got something else"
        subset = full_dataset.select(range(100))
        os.makedirs(os.path.dirname(data_str), exist_ok=True)
        subset.save_to_disk(data_str)
        print(f"Generated pile-100 in {data_str}")

    config_path = os.path.join(test_path, "index_config.json")
    if os.path.exists(config_path):
        if not overwrite:
            # Load existing config and compare
            with open(config_path, "r") as f:
                existing_cfg_dict = json.load(f)

            new_cfg_dict = asdict(cfg)

            if existing_cfg_dict != new_cfg_dict:
                # Show differences for debugging
                diffs = [
                    f"  {k}: {existing_cfg_dict[k]} != {new_cfg_dict[k]}"
                    for k in new_cfg_dict
                    if k in existing_cfg_dict
                    and existing_cfg_dict[k] != new_cfg_dict[k]
                ]
                raise RuntimeError(
                    f"Existing config at {config_path} differs from requested config:\n"
                    + "\n".join(diffs)
                    + "\n\nUse --overwrite to replace the existing config."
                )

            print(f"Using existing config from {config_path}")
        else:
            print(f"Overwriting existing config at {config_path}")
            with open(config_path, "w") as f:
                json.dump(asdict(cfg), f, indent=4)
    else:
        # Save new config
        with open(config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=4)

    # Setup
    workers = world_size
    device = torch.device(get_device(0))
    target_modules = None

    # Determine dtype
    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float16
            )
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    return cfg, workers, device, target_modules, dtype


if __name__ == "__main__" or TYPE_CHECKING:
    cfg, workers, device, target_modules, dtype = setup_paths_and_config(
        precision, test_path, model_name, world_size_arg, overwrite_arg
    )


# %% [markdown]
# ## 1. Loading model and data


# %%
def load_model_step(cfg: IndexConfig, dtype: torch.dtype) -> PreTrainedModel:
    """Load the model."""
    print(f"Loading model {cfg.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=cfg.precision == "int4",
                load_in_8bit=cfg.precision == "int8",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_storage=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if cfg.precision in ("int4", "int8")
            else None
        ),
        dtype=dtype,
    )
    return model


if __name__ == "__main__" or TYPE_CHECKING:
    model = load_model_step(cfg, dtype)


# %%
def load_dataset_step(cfg: IndexConfig) -> Dataset:
    """Load and return the dataset."""
    data_str = cfg.data.dataset
    print(f"Loading dataset from {data_str}...")

    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(data_str, split="train")
            if isinstance(ds, (DatasetDict, IterableDatasetDict)):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    assert isinstance(ds, Dataset)
    return ds


if __name__ == "__main__" or TYPE_CHECKING:
    ds = load_dataset_step(cfg)


# %%
def tokenize_and_allocate_step(
    ds: Dataset, cfg: IndexConfig, workers: int
) -> tuple[Dataset, Batches, Any]:
    """Tokenize dataset and allocate batches."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, model_max_length=cfg.token_batch_size
    )
    ds = ds.map(
        tokenize, batched=True, fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer)
    )
    data = ds

    # Allocate batches
    batches_world = allocate_batches_test(
        doc_lengths=ds["length"], N=cfg.token_batch_size, workers=workers
    )
    assert len(batches_world) == workers

    return data, batches_world, tokenizer


if __name__ == "__main__" or TYPE_CHECKING:
    data, batches_world, tokenizer = tokenize_and_allocate_step(ds, cfg, workers)


# %% [markdown]
# ## 2. Compute activation and gradient covariance


# %%
def compute_covariance(
    rank: int,
    model: PreTrainedModel,
    data: Dataset,
    batches_world: Batches,
    device: torch.device,
    target_modules: Any,
    activation_covariances: dict[str, Tensor],
    gradient_covariances: dict[str, Tensor],
) -> dict[str, Any]:
    """Compute activation and gradient covariances for a single worker."""
    total_processed = 0
    batches = batches_world[rank]
    loss_list = []

    collector = GroundTruthCovarianceCollector(
        model=model.base_model,
        activation_covariances=activation_covariances,
        gradient_covariances=gradient_covariances,
        target_modules=target_modules,
    )

    for sl in tqdm(batches, desc=f"Rank {rank} covariances"):
        batch = data[sl]
        x, y, valid_masks = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=device,
        )

        total_processed += valid_masks.sum()
        collector.set_valid_masks(valid_masks)

        with collector:
            logits = model(x).logits
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])

            losses.sum().backward()
            loss_list.append(losses.detach().cpu())
            model.zero_grad()

    return {"losses": loss_list, "total_processed_rank": total_processed.item()}


# %%
def compute_covariances_step(
    model: PreTrainedModel,
    data: Dataset,
    batches_world: Batches,
    device: torch.device,
    target_modules: Any,
    workers: int,
    test_path: str,
) -> str:
    """Compute covariances for all ranks and save to disk."""
    covariance_test_path = os.path.join(test_path, "covariances")

    for rank in range(workers):
        covariance_test_path_rank = os.path.join(covariance_test_path, f"rank_{rank}")
        os.makedirs(covariance_test_path_rank, exist_ok=True)

        activation_covariances = {}
        gradient_covariances = {}
        d = compute_covariance(
            rank=rank,
            model=model,
            data=data,
            batches_world=batches_world,
            device=device,
            target_modules=target_modules,
            activation_covariances=activation_covariances,
            gradient_covariances=gradient_covariances,
        )

        save_file(
            activation_covariances,
            os.path.join(
                covariance_test_path_rank, "activation_covariance.safetensors"
            ),
        )
        save_file(
            gradient_covariances,
            os.path.join(covariance_test_path_rank, "gradient_covariance.safetensors"),
        )
        with open(os.path.join(covariance_test_path_rank, "stats.json"), "w") as f:
            json.dump({"total_processed_rank": d["total_processed_rank"]}, f, indent=4)
            print(f"Rank {rank} processed {d['total_processed_rank']} tokens.")

    return covariance_test_path


if __name__ == "__main__" or TYPE_CHECKING:
    print("\n=== Computing Covariances ===")
    covariance_test_path = compute_covariances_step(
        model, data, batches_world, device, target_modules, workers, test_path
    )


# %%
def combine_covariances_step(
    covariance_test_path: str, workers: int, device: torch.device
) -> int:
    """Combine covariance results from all ranks."""
    activation_covariances = TensorDict({})
    gradient_covariances = TensorDict({})
    total_processed_global = 0

    for rank in range(workers):
        covariance_test_path_rank = os.path.join(covariance_test_path, f"rank_{rank}")

        with open(os.path.join(covariance_test_path_rank, "stats.json"), "r") as f:
            d = json.load(f)
            total_processed_global += d["total_processed_rank"]

        activation_covariances_rank = TensorDict(
            load_file(
                os.path.join(
                    covariance_test_path_rank, "activation_covariance.safetensors"
                )
            )
        ).to(device)

        gradient_covariances_rank = TensorDict(
            load_file(
                os.path.join(
                    covariance_test_path_rank, "gradient_covariance.safetensors"
                )
            )
        ).to(device)

        if not activation_covariances:
            activation_covariances = activation_covariances_rank
        else:
            activation_covariances = (
                activation_covariances + activation_covariances_rank
            )

        if not gradient_covariances:
            gradient_covariances = gradient_covariances_rank
        else:
            gradient_covariances = gradient_covariances + gradient_covariances_rank

    save_file(
        activation_covariances.to_dict(),
        os.path.join(covariance_test_path, "activation_covariance.safetensors"),
    )
    save_file(
        gradient_covariances.to_dict(),
        os.path.join(covariance_test_path, "gradient_covariance.safetensors"),
    )
    with open(os.path.join(covariance_test_path, "stats.json"), "w") as f:
        json.dump({"total_processed_global": total_processed_global}, f, indent=4)
        print(f"Global processed {total_processed_global} tokens.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_processed_global


if __name__ == "__main__" or TYPE_CHECKING:
    print("\n=== Combining Covariances ===")
    total_processed_global = combine_covariances_step(
        covariance_test_path, workers, device
    )


# %% [markdown]
# ## 3. Compute eigenvalues and eigenvectors


# %%
def compute_eigenvectors_step(
    test_path: str, device: torch.device, dtype: torch.dtype
) -> str:
    """Compute eigenvectors from covariances."""
    covariance_test_path = os.path.join(test_path, "covariances")
    eigenvectors_test_path = os.path.join(test_path, "eigenvectors")
    os.makedirs(eigenvectors_test_path, exist_ok=True)

    # Load covariances
    with open(os.path.join(covariance_test_path, "stats.json"), "r") as f:
        d = json.load(f)
        total_processed_global = d["total_processed_global"]

    activation_covariances = load_file(
        os.path.join(covariance_test_path, "activation_covariance.safetensors")
    )
    gradient_covariances = load_file(
        os.path.join(covariance_test_path, "gradient_covariance.safetensors")
    )

    eigenvectors_activations = {}
    eigenvectors_gradients = {}

    for name in activation_covariances.keys():
        a = activation_covariances[name].to(dtype=torch.float64, device=device)
        g = gradient_covariances[name].to(dtype=torch.float64, device=device)
        a = (a + a.T).div(2)
        g = (g + g.T).div(2)
        a.div_(total_processed_global)
        g.div_(total_processed_global)

        eigenvalues_a, eigenvectors_a = torch.linalg.eigh(a)
        eigenvalues_g, eigenvectors_g = torch.linalg.eigh(g)
        print(
            f"{name}: eigenvectors_a.sum()={eigenvectors_a.sum()}, eigenvectors_g.sum()={eigenvectors_g.sum()}"
        )
        eigenvectors_activations[name] = eigenvectors_a.to(dtype=dtype).contiguous()
        eigenvectors_gradients[name] = eigenvectors_g.to(dtype=dtype).contiguous()

    save_file(
        eigenvectors_activations,
        os.path.join(eigenvectors_test_path, "eigenvectors_activations.safetensors"),
    )
    save_file(
        eigenvectors_gradients,
        os.path.join(eigenvectors_test_path, "eigenvectors_gradients.safetensors"),
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eigenvectors_test_path


if __name__ == "__main__" or TYPE_CHECKING:
    print("\n=== Computing Eigenvectors ===")
    eigenvectors_test_path = compute_eigenvectors_step(test_path, device, dtype)


# %% [markdown]
# ## 4. Compute eigenvalue correction


# %%
def compute_eigenvalue_correction_amortized(
    rank: int,
    model: PreTrainedModel,
    data: Dataset,
    batches_world: Batches,
    device: torch.device,
    target_modules: Any,
    eigenvalue_corrections: dict[str, Tensor],
    eigenvectors_activations: dict[str, Tensor],
    eigenvectors_gradients: dict[str, Tensor],
) -> dict[str, int]:
    """Compute eigenvalue corrections using the amortized method."""
    total_processed = 0
    batches = batches_world[rank]

    collector = GroundTruthAmortizedLambdaCollector(
        model=model.base_model,
        eigenvalue_corrections=eigenvalue_corrections,
        eigenvectors_activations=eigenvectors_activations,
        eigenvectors_gradients=eigenvectors_gradients,
        device=device,
        target_modules=target_modules,
    )

    for sl in tqdm(batches, desc=f"Rank {rank} eigenvalue corrections"):
        batch = data[sl]
        x, y, valid_masks = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=device,
        )

        total_processed += valid_masks.sum()
        collector.set_valid_masks(valid_masks)

        with collector:
            logits = model(x).logits
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])

            losses.sum().backward()
            model.zero_grad()

    return {"total_processed_rank": total_processed.item()}


# %%
def compute_eigenvalue_corrections_step(
    model: PreTrainedModel,
    data: Dataset,
    batches_world: Batches,
    device: torch.device,
    target_modules: Any,
    workers: int,
    test_path: str,
) -> tuple[str, int]:
    """Compute eigenvalue corrections for all ranks."""
    eigenvectors_test_path = os.path.join(test_path, "eigenvectors")
    eigenvalue_correction_test_path = os.path.join(test_path, "eigenvalue_corrections")
    os.makedirs(eigenvalue_correction_test_path, exist_ok=True)

    # Load eigenvectors
    eigenvectors_activations = load_file(
        os.path.join(eigenvectors_test_path, "eigenvectors_activations.safetensors")
    )
    eigenvectors_gradients = load_file(
        os.path.join(eigenvectors_test_path, "eigenvectors_gradients.safetensors")
    )

    total_processed_global = 0
    for rank in range(workers):
        eigenvalue_correction_test_path_rank = os.path.join(
            eigenvalue_correction_test_path, f"rank_{rank}"
        )
        os.makedirs(eigenvalue_correction_test_path_rank, exist_ok=True)

        eigenvalue_corrections = {}
        d = compute_eigenvalue_correction_amortized(
            rank=rank,
            model=model,
            data=data,
            batches_world=batches_world,
            device=device,
            target_modules=target_modules,
            eigenvalue_corrections=eigenvalue_corrections,
            eigenvectors_activations=eigenvectors_activations,
            eigenvectors_gradients=eigenvectors_gradients,
        )

        save_file(
            eigenvalue_corrections,
            os.path.join(
                eigenvalue_correction_test_path_rank,
                "eigenvalue_corrections.safetensors",
            ),
        )
        with open(
            os.path.join(eigenvalue_correction_test_path_rank, "stats.json"), "w"
        ) as f:
            json.dump({"total_processed_rank": d["total_processed_rank"]}, f, indent=4)
            print(f"Rank {rank} processed {d['total_processed_rank']} tokens.")
        total_processed_global += d["total_processed_rank"]

    return eigenvalue_correction_test_path, total_processed_global


if __name__ == "__main__" or TYPE_CHECKING:
    print("\n=== Computing Eigenvalue Corrections ===")
    eigenvalue_correction_test_path, total_processed_global_lambda = (
        compute_eigenvalue_corrections_step(
            model, data, batches_world, device, target_modules, workers, test_path
        )
    )


# %%
def combine_eigenvalue_corrections_step(
    eigenvalue_correction_test_path: str,
    workers: int,
    device: torch.device,
    total_processed_global: int,
) -> TensorDict:
    """Combine eigenvalue correction results from all ranks."""
    eigenvalue_corrections = TensorDict({})

    for rank in range(workers):
        eigenvalue_correction_test_path_rank = os.path.join(
            eigenvalue_correction_test_path, f"rank_{rank}"
        )

        eigenvalue_corrections_rank = TensorDict(
            load_file(
                os.path.join(
                    eigenvalue_correction_test_path_rank,
                    "eigenvalue_corrections.safetensors",
                )
            )
        ).to(device)

        if not eigenvalue_corrections:
            eigenvalue_corrections = eigenvalue_corrections_rank
        else:
            eigenvalue_corrections = (
                eigenvalue_corrections + eigenvalue_corrections_rank
            )

    eigenvalue_corrections.div_(total_processed_global)
    save_file(
        eigenvalue_corrections.to_dict(),
        os.path.join(
            eigenvalue_correction_test_path, "eigenvalue_corrections.safetensors"
        ),
    )

    return eigenvalue_corrections


if __name__ == "__main__" or TYPE_CHECKING:
    eigenvalue_corrections = combine_eigenvalue_corrections_step(
        eigenvalue_correction_test_path, workers, device, total_processed_global_lambda
    )
    print("\n=== Ground Truth Computation Complete ===")
    print(f"Results saved to: {test_path}")
