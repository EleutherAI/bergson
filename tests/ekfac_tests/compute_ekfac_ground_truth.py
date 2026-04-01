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
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from ground_truth.collector import (
    GroundTruthAmortizedLambdaCollector,
    GroundTruthCovarianceCollector,
)
from safetensors.torch import load_file, save_file
from test_utils import add_tensor_dicts, tensor_dict_to_device
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from bergson.config import DataConfig, IndexConfig
from bergson.data import _allocate_batches_world, pad_and_tensor, tokenize
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import assert_type, get_device, setup_reproducibility

Precision = str  # Type alias for precision strings

Batches = list[list[list[int]]]

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
        help="Output directory for ground truth results "
        "(default: test_files/pile_100_examples/ground_truth)",
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

    setup_reproducibility()

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
    token_batch_size: int = 2048,
    n_samples: int = 100,
) -> tuple[IndexConfig, int, torch.device, Any, torch.dtype]:
    """Setup paths and configuration object."""
    os.makedirs(test_path, exist_ok=True)

    current_path = os.getcwd()
    parent_path = os.path.join(current_path, "test_files", f"pile_{n_samples}_examples")

    # Configuration
    cfg = IndexConfig(run_path="", loss_reduction="sum")
    cfg.model = model_name
    cfg.precision = precision  # type: ignore[assignment]
    cfg.fsdp = False
    cfg.data = DataConfig(dataset=os.path.join(parent_path, "data"), truncation=True)
    cfg.token_batch_size = token_batch_size

    # model_max_length is limited in some models like `roneneldan/TinyStories-1M`
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if (
        hasattr(tokenizer, "model_max_length")
        and tokenizer.model_max_length < cfg.token_batch_size
    ):
        print(
            f"Warning: Got --token-batch-size {cfg.token_batch_size} but "
            f"{model_name} only supports up to {tokenizer.model_max_length}"
        )
        cfg.token_batch_size = tokenizer.model_max_length

    data_str = cfg.data.dataset

    # Create dataset if it doesn't exist
    if not os.path.exists(data_str):
        full_dataset = load_dataset("NeelNanda/pile-10k", split="train")
        assert isinstance(full_dataset, Dataset), "Expected Dataset, got something else"
        subset = full_dataset.select(range(n_samples))
        os.makedirs(os.path.dirname(data_str), exist_ok=True)
        subset.save_to_disk(data_str)
        print(f"Generated pile-{n_samples} in {data_str}")

    config_path = os.path.join(test_path, "index_config.yaml")
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
            cfg.save_yaml(config_path)
    else:
        # Save new config
        cfg.save_yaml(config_path)

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
        torch_dtype=dtype,
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

    batches_world = _allocate_batches_world(
        doc_lengths=ds["length"][:], N=cfg.token_batch_size, world_size=workers
    )

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
    ekfac_collector: Optional[CovarianceCollector] = None,
) -> dict[str, Any]:
    """Compute activation and gradient covariances for a single worker.

    If ekfac_collector is provided, it will be run simultaneously with the ground
    truth collector during the same forward/backward passes. This ensures both
    collectors see exactly the same gradients.
    """
    total_processed = torch.tensor(0, device=device)
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

        # Run both collectors simultaneously during the same forward/backward pass
        # This ensures they see exactly the same gradients
        with collector.with_batch(valid_masks):
            if ekfac_collector is not None:
                ekfac_ctx = ekfac_collector.with_batch(valid_masks)
            else:
                ekfac_ctx = None

            if ekfac_ctx is not None:
                ekfac_ctx.__enter__()
            try:
                # Use same loss computation as EKFAC (fwd_bwd_hessian_factory)
                logits = model(x).logits[:, :-1]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:].flatten(),
                    reduction="none",
                ).reshape_as(y[:, 1:])
                # Sum over sequence first, then over batch
                # (like EKFAC with loss_reduction="sum")
                losses = losses.sum(1)
                losses.sum().backward()
                loss_list.append(losses.detach().cpu())
                model.zero_grad()
            finally:
                if ekfac_ctx is not None:
                    ekfac_ctx.__exit__(None, None, None)

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
    ekfac_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> str:
    """Compute covariances for all ranks and save to disk.

    If ekfac_path is provided, also runs the EKFAC CovarianceCollector simultaneously
    during the same forward/backward passes. This ensures both collectors see exactly
    the same gradients, enabling precise numerical comparison.
    """
    setup_reproducibility()

    covariance_test_path = os.path.join(test_path, "covariances")

    # Create EKFAC collector if path is provided
    ekfac_collector = None
    if ekfac_path is not None:
        os.makedirs(ekfac_path, exist_ok=True)
        ekfac_collector = CovarianceCollector(
            model=model.base_model,
            dtype=dtype,
            path=ekfac_path,
            target_modules=target_modules,
        )

    total_processed_global = 0
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
            ekfac_collector=ekfac_collector,
        )

        total_processed_global += d["total_processed_rank"]

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

    # Finalize EKFAC collector and save total processed
    if ekfac_collector is not None and ekfac_path is not None:
        ekfac_collector.teardown()
        torch.save(
            torch.tensor(total_processed_global, device=device),
            os.path.join(ekfac_path, "total_processed.pt"),
        )

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
    activation_covariances: dict[str, Tensor] = {}
    gradient_covariances: dict[str, Tensor] = {}
    total_processed_global = 0

    for rank in range(workers):
        covariance_test_path_rank = os.path.join(covariance_test_path, f"rank_{rank}")

        with open(os.path.join(covariance_test_path_rank, "stats.json"), "r") as f:
            d = json.load(f)
            total_processed_global += d["total_processed_rank"]

        activation_covariances_rank = tensor_dict_to_device(
            load_file(
                os.path.join(
                    covariance_test_path_rank, "activation_covariance.safetensors"
                )
            ),
            device,
        )

        gradient_covariances_rank = tensor_dict_to_device(
            load_file(
                os.path.join(
                    covariance_test_path_rank, "gradient_covariance.safetensors"
                )
            ),
            device,
        )

        if not activation_covariances:
            activation_covariances = activation_covariances_rank
        else:
            activation_covariances = add_tensor_dicts(
                activation_covariances, activation_covariances_rank
            )

        if not gradient_covariances:
            gradient_covariances = gradient_covariances_rank
        else:
            gradient_covariances = add_tensor_dicts(
                gradient_covariances, gradient_covariances_rank
            )

    save_file(
        activation_covariances,
        os.path.join(covariance_test_path, "activation_covariance.safetensors"),
    )
    save_file(
        gradient_covariances,
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
            f"{name}: eigenvectors_a.sum()={eigenvectors_a.sum()}, "
            f"eigenvectors_g.sum()={eigenvectors_g.sum()}"
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
) -> dict[str, Any]:
    """Compute eigenvalue corrections using the amortized method."""
    total_processed = torch.tensor(0, device=device)
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

        with collector.with_batch(valid_masks):
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
) -> dict[str, Tensor]:
    """Combine eigenvalue correction results from all ranks."""
    eigenvalue_corrections: dict[str, Tensor] = {}

    for rank in range(workers):
        eigenvalue_correction_test_path_rank = os.path.join(
            eigenvalue_correction_test_path, f"rank_{rank}"
        )

        eigenvalue_corrections_rank = tensor_dict_to_device(
            load_file(
                os.path.join(
                    eigenvalue_correction_test_path_rank,
                    "eigenvalue_corrections.safetensors",
                )
            ),
            device,
        )

        if not eigenvalue_corrections:
            eigenvalue_corrections = eigenvalue_corrections_rank
        else:
            eigenvalue_corrections = add_tensor_dicts(
                eigenvalue_corrections, eigenvalue_corrections_rank
            )

    # Divide by total_processed_global
    eigenvalue_corrections = {
        k: v / total_processed_global for k, v in eigenvalue_corrections.items()
    }
    save_file(
        eigenvalue_corrections,
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
