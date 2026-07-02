"""Prepare train/query datasets for the factual-entailment recall evaluation.

Builds two on-disk HF datasets from the CVDB natural-statements corpus
(see data/generate_cvdb.py):

- ``<out_path>/train``: one natural-language statement per (entity, field)
  fact, plus optionally ``num_random`` distractor documents sampled from a
  pretraining corpus (default EleutherAI/SmolLM2-135M-10B). Distractor rows
  get ``identifier=-1`` and ``field="random"`` so they can never be a recall
  ground truth. The model is trained on this dataset and it is also the
  dataset that gets attributed.
- ``<out_path>/query``: ``num_queries`` QA pairs sampled from the *kept*
  facts, pre-tokenized as ``Q: {question}\\nA: {answer}`` with ``labels``
  masked (-100) everywhere except the answer tokens, so query gradients are
  exactly grad log p(answer | question). Bergson skips its own tokenization
  when ``input_ids`` is already present.

Each query's ground-truth training document is the unique train row with the
same ``(identifier, field)``; both datasets keep those columns so the recall
evaluation (examples/recall/recall_eval.py) can match them by value rather
than by row index.

Example:
    python -m examples.recall.prepare_recall_data runs/recall/data \
        --num_entities 2000 --num_queries 128 --num_random 8000
"""

import json
import random
import sys
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoTokenizer

from bergson.data import load_data_string


@dataclass
class RecallDataConfig:
    """Config for building the recall train/query datasets."""

    out_path: str = field(positional=True)
    """Directory to write the train/ and query/ datasets into."""

    dataset_path: str = "data/cvdb_natural.hf"
    """CVDB natural-variant dataset with text/question/answer/identifier/field
    columns (see data/generate_cvdb.py --variant natural)."""

    num_entities: int = 0
    """Number of entities to keep (each has 4 facts). 0 keeps all."""

    num_queries: int = 128
    """Number of QA queries to sample from the kept facts. Keep modest:
    per-query EK-FAC holds num_queries full model gradients in GPU memory
    during the apply and score steps."""

    num_random: int = 0
    """Number of distractor documents from ``random_dataset`` to mix into the
    train dataset (0 disables mixing)."""

    random_dataset: str = "EleutherAI/SmolLM2-135M-10B"
    """Pretraining corpus to sample distractor documents from (streamed)."""

    random_max_tokens: int = 256
    """Truncate each distractor document to this many tokens (re-decoded to
    text) so mixed batches stay comparable to the short fact statements."""

    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    """Tokenizer for query pre-tokenization and distractor truncation. Must
    match the model that will be trained/attributed."""

    seed: int = 0


def sample_entities(ds: Dataset, num_entities: int, seed: int) -> Dataset:
    """Keep all facts belonging to a random subset of entities."""
    identifiers = sorted(set(ds["identifier"]))
    if num_entities <= 0 or num_entities >= len(identifiers):
        return ds

    rng = random.Random(seed)
    keep = set(rng.sample(identifiers, num_entities))
    return ds.filter(lambda row: row["identifier"] in keep)


def sample_random_docs(run_cfg: RecallDataConfig, tokenizer) -> list[dict]:
    """Stream ``num_random`` distractor documents, truncated to a token budget.

    Documents are re-decoded to text so the train dataset stays uniform
    (bergson tokenizes it once, at training/attribution time).
    """
    stream = load_dataset(run_cfg.random_dataset, split="train", streaming=True)
    stream = stream.shuffle(seed=run_cfg.seed, buffer_size=10_000)

    rows = []
    for row in stream:
        ids = tokenizer(
            row["text"], truncation=True, max_length=run_cfg.random_max_tokens
        )
        text = tokenizer.decode(ids["input_ids"], skip_special_tokens=True)
        rows.append(
            {
                "text": text,
                "identifier": -1,
                "field": "random",
                "question": "",
                "answer": "",
                "entity": "",
            }
        )
        if len(rows) >= run_cfg.num_random:
            break

    if len(rows) < run_cfg.num_random:
        raise ValueError(
            f"Only found {len(rows)} distractor docs, wanted {run_cfg.num_random}"
        )
    return rows


def tokenize_query(question: str, answer: str, tokenizer) -> dict:
    """Tokenize ``Q: {question}\\nA: {answer}`` with labels on the answer only.

    The answer span is located by character offset (rightmost match, like
    bergson.data.tokenize) so BPE merges across the prompt/answer boundary
    can't misalign the mask.
    """
    text = f"Q: {question}\nA: {answer}"
    encoding = tokenizer(text, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]

    start_char = text.rfind(answer)
    labels = [
        tok if off[1] > start_char else -100
        for tok, off in zip(input_ids, encoding["offset_mapping"])
    ]
    assert any(label != -100 for label in labels), f"No answer tokens in {text!r}"

    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
        "text": text,
    }


def main():
    run_cfg = parse(RecallDataConfig)
    rng = random.Random(run_cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.tokenizer)

    facts = load_data_string(run_cfg.dataset_path)
    facts = sample_entities(facts, run_cfg.num_entities, run_cfg.seed)
    print(f"Kept {len(facts)} facts from {run_cfg.dataset_path}")

    keep_columns = ["text", "identifier", "field", "question", "answer", "entity"]
    train_rows = [{k: row[k] for k in keep_columns} for row in facts]

    if run_cfg.num_random > 0:
        print(
            f"Sampling {run_cfg.num_random} distractors from "
            f"{run_cfg.random_dataset}..."
        )
        train_rows += sample_random_docs(run_cfg, tokenizer)

    rng.shuffle(train_rows)
    train_ds = Dataset.from_list(train_rows)

    # Sample queries from the kept facts and pre-tokenize them. With an
    # asymmetric-answers dataset, only rows whose statement shares no surface
    # form with the answer are eligible (asym column, see data/cvdb.py).
    if "asym" in facts.column_names:
        eligible = [i for i, asym in enumerate(facts["asym"]) if asym]
        print(f"Sampling queries from {len(eligible)}/{len(facts)} asym facts")
    else:
        eligible = range(len(facts))
    query_indices = rng.sample(eligible, run_cfg.num_queries)
    query_rows = []
    for i in query_indices:
        row = facts[i]
        query_rows.append(
            {
                "identifier": row["identifier"],
                "field": row["field"],
                "question": row["question"],
                "answer": row["answer"],
                "entity": row["entity"],
                **tokenize_query(row["question"], row["answer"], tokenizer),
            }
        )
    query_ds = Dataset.from_list(query_rows)

    train_path = f"{run_cfg.out_path}/train"
    query_path = f"{run_cfg.out_path}/query"
    train_ds.save_to_disk(train_path)
    query_ds.save_to_disk(query_path)

    manifest = {
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "num_facts": len(facts),
        "num_train_docs": len(train_ds),
        "num_random": run_cfg.num_random,
        "num_queries": len(query_ds),
    }
    with open(f"{run_cfg.out_path}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(train_ds)} train docs to {train_path}")
    print(f"Saved {len(query_ds)} queries to {query_path}")
    print(
        "Reproduce with: python -m examples.recall.prepare_recall_data"
        f" {run_cfg.out_path} --dataset_path {run_cfg.dataset_path}"
        f" --num_entities {run_cfg.num_entities}"
        f" --num_queries {run_cfg.num_queries} --num_random {run_cfg.num_random}"
        f" --tokenizer {run_cfg.tokenizer} --seed {run_cfg.seed}"
    )


if __name__ == "__main__":
    main()
