"""Shared helpers for non-gradient attribution baselines on a re-train bank.

A "re-train bank" is any directory written with ``save_models=true``
(e.g. by ``examples/magic/gpt2_wikitext_bank.yaml``): it holds ``subsets.json``,
``retrained/base`` and ``retrained/subset_*`` checkpoints, and the ``config.yaml``
that built it. Each baseline reads the model + dataset straight from the bank,
produces a ``[num_train_docs, num_queries]`` score matrix, and evaluates it by
per-query LDS -- so ``--bank <dir>`` is all anyone needs to run one.

The LDS itself reuses ``bergson``'s ``evaluate_retrained``: it computes each
subset's query-loss diff from the banked models once, caches it under the bank,
and every later baseline on the same bank/query reuses that cache.
"""

import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

from bergson.config.config import DataConfig
from bergson.data import load_data_string
from bergson.magic.config import MagicConfig
from bergson.validate import evaluate_retrained

REPO = Path(__file__).resolve().parents[2]

# The default bank built on demand when ``--bank`` is omitted.
DEFAULT_BANK = REPO / "runs" / "gpt2_wikitext_bank"
DEFAULT_BANK_CONFIG = REPO / "examples" / "magic" / "gpt2_wikitext_bank.yaml"
# LDS query set (50 held-out docs); overridable per run.
DEFAULT_QUERY_SPLIT = "test[1:51]"


@dataclass
class BankSpec:
    """The model + data a bank was built on, read from its ``config.yaml``."""

    model: str
    base_model: str  # retrained/base if present, else the base model id
    dataset: str
    train_split: str
    prompt_column: str
    chunk_length: int
    batch_size: int


def ensure_bank(bank: str | None) -> Path:
    """Return a ready re-train bank, building the default one if none is given.

    A bank is ready when it has ``subsets.json`` and ``retrained/base``. When
    ``bank`` is ``None`` and the default bank is absent, it is built by running
    its config through ``bergson`` (a leave-k-out re-training job).
    """
    if bank is not None:
        path = Path(bank)
        if not (path / "subsets.json").exists():
            raise FileNotFoundError(
                f"{path} is not a re-train bank (no subsets.json); pass a dir "
                "written with save_models=true"
            )
        return path

    if not (DEFAULT_BANK / "subsets.json").exists():
        cmd = ["bergson", str(DEFAULT_BANK_CONFIG)]
        print(
            f"No bank passed and {DEFAULT_BANK} missing; building it:\n"
            f"  {' '.join(cmd)}"
        )
        subprocess.run(cmd, cwd=REPO, check=True)
    return DEFAULT_BANK


def read_bank_spec(bank: Path) -> BankSpec:
    """Read the model + dataset a bank was built on from its ``config.yaml``."""
    with open(bank / "config.yaml") as f:
        config = yaml.safe_load(f)
    # The bank-building step (magic/train) is the one carrying model + data.
    step = next(
        v
        for s in config["steps"]
        for v in s.values()
        if isinstance(v, dict) and "model" in v and "data" in v
    )
    data = step["data"]
    base = bank / "retrained" / "base"
    return BankSpec(
        model=step["model"],
        base_model=str(base) if base.exists() else step["model"],
        dataset=data["dataset"],
        train_split=data.get("split", "train"),
        prompt_column=data.get("prompt_column", "text"),
        chunk_length=data.get("chunk_length", 0),
        batch_size=step.get("batch_size", 64),
    )


@cache
def _tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)


def _texts(dataset, spec: BankSpec) -> list[str]:
    """Doc texts, decoding ``input_ids`` when the set is pre-tokenized."""
    if spec.prompt_column in dataset.column_names:
        return list(dataset[spec.prompt_column])
    if "input_ids" in dataset.column_names:
        tok = _tokenizer(spec.model)
        return [
            tok.decode(ids, skip_special_tokens=True) for ids in dataset["input_ids"]
        ]
    raise ValueError(
        f"dataset has neither {spec.prompt_column!r} nor input_ids: "
        f"{dataset.column_names}"
    )


def load_texts(
    spec: BankSpec, query_dataset: str, query_split: str
) -> tuple[list[str], list[str]]:
    """Return (train_texts, query_texts) in original doc order.

    Row i of ``train_texts`` is training doc id i, matching the ids in the
    bank's ``subsets.json`` and the rows of every score matrix. Pre-tokenized
    datasets (``input_ids`` only) are decoded with the model's tokenizer.
    """
    train = load_data_string(spec.dataset, spec.train_split)
    query = load_data_string(query_dataset, query_split)
    return _texts(train, spec), _texts(query, spec)


def evaluate_lds(
    bank: Path,
    score_path: Path,
    out_dir: Path,
    spec: BankSpec,
    query_dataset: str,
    query_split: str,
) -> np.ndarray:
    """Per-query LDS Spearman of a score matrix against the bank.

    Runs ``evaluate_retrained`` (no re-training; reuses the bank's cached
    per-subset query losses) and returns the per-query Spearman correlations it
    writes to ``summary.csv``.
    """
    run_cfg = MagicConfig(
        run_path=str(out_dir),
        model=spec.model,
        precision="fp32",
        batch_size=spec.batch_size,
        query=DataConfig(
            dataset=query_dataset,
            split=query_split,
            prompt_column=spec.prompt_column,
            chunk_length=0,
        ),
    )
    evaluate_retrained(run_cfg, str(bank), score_path=str(score_path))

    summary = np.genfromtxt(out_dir / "summary.csv", delimiter=",", names=True)
    return np.atleast_1d(summary["spearman_corr"]).astype(float)


def save_scores(scores: np.ndarray, out_dir: Path, name: str) -> Path:
    """Save a ``[num_train_docs, num_queries]`` score matrix as ``.npy``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npy"
    np.save(path, scores.astype(np.float32))
    print(f"Saved scores {scores.shape} -> {path}")
    return path


def report(name: str, rhos: np.ndarray) -> None:
    """Print a per-query LDS summary consistent with the repo's convention."""
    print(f"\n=== {name}: per-query LDS Spearman (n={len(rhos)} queries) ===")
    print(f"  mean   {np.mean(rhos):+.4f}")
    print(f"  median {np.median(rhos):+.4f}")
    print(f"  min    {np.min(rhos):+.4f}   max {np.max(rhos):+.4f}")
    print(f"  frac>0 {np.mean(rhos > 0):.2f}")
