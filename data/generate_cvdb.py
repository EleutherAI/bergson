"""Generate the CVDB-based QA dataset about famous people.

Two variants, following the paper:
- synthetic: aliases are three-token random strings like "sjdhf"; samples are
  templated QA pairs like "Q: When was <|sjdhf|> born?\nA: 1st century BC\n".
- natural: aliases are five-token phrases like "prickly cyan mouse"; samples
  use much more varied statement templates.

Example:
    python -m data.generate_cvdb --variant synthetic
"""

import random
from dataclasses import dataclass

from datasets import Dataset
from simple_parsing import parse
from transformers import AutoTokenizer

from .aliases import generate_letter_aliases, generate_phrase_aliases
from .cvdb import DEFAULT_FIELDS, cvdb_qa_generator
from .cvdb_natural import generate_statement


@dataclass
class CvdbConfig:
    """Config for generating the CVDB QA dataset."""

    csv_path: str = "cache/cvdb/cross-verified-database.csv"
    """Path to the raw CVDB corpus csv."""

    out_path: str = ""
    """Where to save the HF dataset. Defaults to data/cvdb_<variant>.hf."""

    variant: str = "synthetic"
    """'synthetic' (3-token letter aliases, Q/A pairs) or 'natural'
    (5-token phrase aliases, varied statement templates)."""

    num_ents: int = 16000
    """Number of entities; 4 QA pairs each, so 16000 -> 64000 samples."""

    equalize_gender: bool = True
    """Take the top num_ents/2 male and female entities by readership."""

    tokenizer: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    """Tokenizer used to enforce alias token lengths."""

    asymmetric_answers: bool = False
    """Natural variant only: statements carry a different surface form than
    the query answer ("born in 1943" vs "1940s"; occupation/region synonyms)
    so a query's entailing statement cannot be found by string matching.
    Rows gain statement_value/asym columns; unmapped occupations keep
    symmetric forms with asym=False."""

    seed: int = 0


def generate_cvdb_dataset(run_cfg: CvdbConfig) -> Dataset:
    rng = random.Random(run_cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.tokenizer)

    if run_cfg.variant == "synthetic":
        aliases = generate_letter_aliases(
            run_cfg.num_ents, tokenizer, n_tokens=3, rng=rng
        )
    elif run_cfg.variant == "natural":
        aliases = generate_phrase_aliases(
            run_cfg.num_ents,
            tokenizer,
            adjective_path="data/names/adjective.txt",
            noun_path="data/names/noun.txt",
            n_tokens=5,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown variant: {run_cfg.variant!r}")

    if run_cfg.asymmetric_answers and run_cfg.variant != "natural":
        raise ValueError("asymmetric_answers requires --variant natural")

    rows = list(
        cvdb_qa_generator(
            run_cfg.csv_path,
            num_ents=run_cfg.num_ents,
            fields=DEFAULT_FIELDS,
            equalize_gender=run_cfg.equalize_gender,
            aliases=aliases,
            asymmetric=run_cfg.asymmetric_answers,
        )
    )

    if run_cfg.variant == "natural":
        for row in rows:
            statement_value = row.get("statement_value", row["answer"])
            row["text"] = generate_statement(
                rng, row["field"], f"<|{row['alias']}|>", statement_value
            )

    return Dataset.from_list(rows)


if __name__ == "__main__":
    run_cfg = parse(CvdbConfig)

    dataset = generate_cvdb_dataset(run_cfg)
    suffix = "_asym" if run_cfg.asymmetric_answers else ""
    out_path = run_cfg.out_path or f"data/cvdb_{run_cfg.variant}{suffix}.hf"
    dataset.save_to_disk(out_path)
    print(f"Saved {len(dataset)} samples to {out_path}")
    print(
        "Reproduce with: python -m data.generate_cvdb"
        f" --csv_path {run_cfg.csv_path} --variant {run_cfg.variant}"
        f" --num_ents {run_cfg.num_ents} --tokenizer {run_cfg.tokenizer}"
        f" --asymmetric_answers {run_cfg.asymmetric_answers}"
        f" --seed {run_cfg.seed}"
    )
