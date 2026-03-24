"""Score computation utilities for semantic experiments."""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bergson import IndexConfig
from bergson.data import load_gradients
from bergson.gradients import GradientProcessor
from bergson.process_grads import mix_preconditioners
from bergson.utils.math import damped_psd_power


def load_scores_matrix(scores_path: Path | str) -> np.ndarray:
    """Load the scores matrix from bergson score output as a dense array.

    Args:
        scores_path: Path to the scores directory containing info.json and scores.bin.

    Returns:
        Dense (num_items, num_scores) float32 array of scores.
    """
    scores_path = Path(scores_path)

    with open(scores_path / "info.json") as f:
        info = json.load(f)

    num_items = info["num_items"]
    num_scores = info["num_scores"]

    # Handle both tuple format (from bergson) and list format (from JSON serialization)
    dtype_spec = info["dtype"]
    if (
        isinstance(dtype_spec, list)
        and len(dtype_spec) > 0
        and isinstance(dtype_spec[0], list)
    ):
        # Convert list of lists back to list of tuples
        dtype_spec = [tuple(item) for item in dtype_spec]

    scores_mmap = np.memmap(
        scores_path / "scores.bin",
        dtype=np.dtype(dtype_spec),
        mode="r",
        shape=(num_items,),
    )

    # Extract score columns into a dense matrix
    scores = np.zeros((num_items, num_scores), dtype=np.float32)
    for i in range(num_scores):
        scores[:, i] = scores_mmap[f"score_{i}"]

    return scores


def compute_scores_fast(
    index_path: Path | str,
    output_path: Path | str,
    preconditioner_path: Path | str | None = None,
    unit_normalize: bool = True,
    batch_size: int = 256,
) -> None:
    """Compute pairwise similarities directly from precomputed gradients.

    Much faster than bergson score since it doesn't recompute gradients.
    Loads gradients from index, applies preconditioning, and computes G @ G.T.

    Args:
        index_path: Path to the gradient index.
        output_path: Path to save scores.
        preconditioner_path: Optional path to preconditioner for query gradients.
        unit_normalize: Whether to unit normalize gradients before scoring.
        batch_size: Batch size for score computation.
    """
    output_path = Path(output_path)
    index_path = Path(index_path)

    if output_path.exists():
        print(f"Scores already exist at {output_path}, skipping...")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Load gradients
    print("Loading gradients from index...")
    grads = load_gradients(index_path, structured=True)

    # Get module names
    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]
    n_samples = info["num_grads"]

    print(f"  {n_samples} samples, {len(module_names)} modules")

    # Load and apply preconditioner if specified
    if preconditioner_path:
        preconditioner_path = Path(preconditioner_path)
        print(f"Loading preconditioner from {preconditioner_path}...")
        proc = GradientProcessor.load(preconditioner_path)

        # Compute H^(-1) for each module using the shared utility
        h_inv = {}
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1)

        # Bergson's approach (from score.py):
        # 1. Query: precondition with H^(-1), then unit normalize
        # 2. Index: unit normalize (no preconditioning)
        # 3. Score: index @ query.T
        print("Loading gradients...")
        all_grads_raw = []
        for name in tqdm(module_names, desc="Loading gradients"):
            g = torch.from_numpy(grads[name].copy()).float()
            all_grads_raw.append(g)

        # Apply H^(-1) to query gradients first (before normalization)
        print("Applying H^(-1) to query gradients...")
        all_grads_query = []
        for name, g in zip(module_names, all_grads_raw):
            g_precond = (g.to(device) @ h_inv[name]).cpu()
            all_grads_query.append(g_precond)
        all_grads_query = torch.cat(all_grads_query, dim=1)
        all_grads_raw = torch.cat(all_grads_raw, dim=1)
        print(f"Gradient matrix shape: {all_grads_raw.shape}")

        # Unit normalize after preconditioning (for query) and raw (for index)
        if unit_normalize:
            print("Unit normalizing gradients...")
            # Normalize preconditioned query
            query_norms = all_grads_query.norm(dim=1, keepdim=True)
            all_grads_query = all_grads_query / (query_norms + 1e-8)
            # Normalize raw index
            index_norms = all_grads_raw.norm(dim=1, keepdim=True)
            all_grads_index = all_grads_raw / (index_norms + 1e-8)
        else:
            all_grads_index = all_grads_raw

        # Score: index (normalized) @ query (preconditioned then normalized).T
        print("Computing pairwise similarities...")
        all_grads_index = all_grads_index.cuda()
        all_grads_query = all_grads_query.cuda()

        scores = torch.zeros(n_samples, n_samples, dtype=torch.float32)
        for i in tqdm(range(0, n_samples, batch_size), desc="Scoring"):
            batch = all_grads_index[i : i + batch_size]
            scores[i : i + batch_size] = (batch @ all_grads_query.T).cpu()
    else:
        # No preconditioning - just concatenate modules
        print("Concatenating gradients (no preconditioning)...")
        all_grads = torch.from_numpy(
            load_gradients(index_path, structured=False).copy()
        ).float()

        print(f"Gradient matrix shape: {all_grads.shape}")

        # Unit normalize if requested
        if unit_normalize:
            print("Unit normalizing gradients...")
            norms = all_grads.norm(dim=1, keepdim=True)
            all_grads = all_grads / (norms + 1e-8)

        # Compute pairwise similarities in batches (G @ G.T)
        print("Computing pairwise similarities...")
        all_grads = all_grads.cuda()

        scores = torch.zeros(n_samples, n_samples, dtype=torch.float32)
        for i in tqdm(range(0, n_samples, batch_size), desc="Scoring"):
            batch = all_grads[i : i + batch_size]
            scores[i : i + batch_size] = (batch @ all_grads.T).cpu()

    # Save in bergson score format
    print(f"Saving scores to {output_path}...")

    # Create structured dtype for scores
    score_dtype_list = [(f"score_{i}", "<f4") for i in range(n_samples)]
    score_dtype = np.dtype(score_dtype_list)
    scores_np = np.zeros(n_samples, dtype=score_dtype)
    for i in range(n_samples):
        scores_np[f"score_{i}"] = scores[:, i].numpy()

    # Save as memmap
    scores_mmap = np.memmap(
        output_path / "scores.bin",
        dtype=score_dtype,
        mode="w+",
        shape=(n_samples,),
    )
    scores_mmap[:] = scores_np
    scores_mmap.flush()

    # Save info - dtype as list of tuples for JSON serialization
    with open(output_path / "info.json", "w") as f:
        json.dump(
            {
                "num_items": n_samples,
                "num_scores": n_samples,
                "dtype": score_dtype_list,
            },
            f,
            indent=2,
        )

    print("Done!")


def compute_scores_with_bergson(
    index_path: Path | str,
    output_path: Path | str,
    query_preconditioner_path: str | Path | None = None,
    index_preconditioner_path: str | Path | None = None,
    unit_normalize: bool = True,
) -> None:
    """Run bergson score to compute pairwise similarities.

    NOTE: This recomputes gradients, which is slow. For index-vs-index
    scoring, use compute_scores_fast() instead.

    If both query_preconditioner_path and index_preconditioner_path are given,
    they are mixed internally before scoring.

    Args:
        index_path: Path to the gradient index.
        output_path: Path to save scores.
        query_preconditioner_path: Optional path to query preconditioner.
        index_preconditioner_path: Optional path to index preconditioner.
        unit_normalize: Whether to unit normalize gradients.
    """
    output_path = Path(output_path)
    index_path = Path(index_path)

    if (output_path / "info.json").exists():
        print(f"Scores already exist at {output_path}, skipping...")
        return

    # Mix preconditioners if both paths are given, otherwise use whichever is provided
    preconditioner_path = None
    if query_preconditioner_path and index_preconditioner_path:
        mixed_path = output_path / "mixed_preconditioner"
        output_path.mkdir(parents=True, exist_ok=True)
        mix_preconditioners(
            query_preconditioner_path,
            index_preconditioner_path,
            mixed_path,
        )
        preconditioner_path = str(mixed_path)
    elif query_preconditioner_path:
        preconditioner_path = str(query_preconditioner_path)
    elif index_preconditioner_path:
        preconditioner_path = str(index_preconditioner_path)

    # Load index config to get model and dataset info
    index_cfg = IndexConfig.load_json(index_path / "index_config.json")

    # Get dataset and column info from config
    data_cfg = index_cfg.data
    dataset_path = data_cfg.dataset or str(index_path / "data.hf")
    prompt_column = data_cfg.prompt_column or "text"
    completion_column = data_cfg.completion_column or ""

    cmd = [
        "bergson",
        "score",
        str(output_path),
        "--model",
        index_cfg.model,
        "--dataset",
        dataset_path,
        "--query_path",
        str(index_path),
        "--score",
        "individual",
        "--projection_dim",
        str(index_cfg.projection_dim or 0),
        "--fsdp",
        "--prompt_column",
        prompt_column,
    ]

    if completion_column:
        cmd.extend(["--completion_column", completion_column])

    if unit_normalize:
        cmd.append("--unit_normalize")

    if preconditioner_path:
        cmd.extend(["--preconditioner_path", preconditioner_path])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"bergson score failed with return code {result.returncode}")
    print(result.stdout)
