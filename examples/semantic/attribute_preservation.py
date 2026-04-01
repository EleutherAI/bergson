"""Attribute Preservation Under Style Suppression Experiment.

This module tests whether style suppression preconditioners preserve the ability
to match on content attributes (not just exact facts). This is a harder test -
we want to surgically remove style signal without damaging attribute signal.

Key insight: Current synthetic data has largely independent facts. For a meaningful
test, we need data where attributes actually correlate or cluster.

Design:
- Create occupational clusters (Scientists, Business, Creative)
- Each cluster has correlated attributes (institution types, degree types, etc.)
- Assign different styles to different clusters in training
- Query in "wrong" style but matching occupation
- Style suppression should preserve attribute matching
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from bergson import IndexConfig
from examples.semantic.data import (
    HF_ANALYSIS_MODEL,
    load_experiment_data,
)

# ==============================================================================
# Attribute Cluster Definitions
# ==============================================================================

# Occupational clusters with correlated attributes
OCCUPATION_CLUSTERS = {
    "scientist": {
        "employers": [
            "MIT",
            "Stanford Research Institute",
            "NASA",
            "CERN",
            "Caltech",
            "Lawrence Berkeley Lab",
            "Fermilab",
            "Max Planck Institute",
            "Cambridge Research",
            "Oxford Physics Lab",
        ],
        "universities": [
            "MIT",
            "Stanford University",
            "Caltech",
            "Princeton University",
            "Harvard University",
            "UC Berkeley",
            "Cambridge University",
            "Oxford University",
            "ETH Zurich",
            "Imperial College London",
        ],
        "degrees": [
            "PhD in Physics",
            "PhD in Chemistry",
            "PhD in Biology",
            "MSc in Mathematics",
        ],
        "titles": ["Dr.", "Professor", "Research Scientist", "Principal Investigator"],
    },
    "business": {
        "employers": [
            "Goldman Sachs",
            "JPMorgan Chase",
            "McKinsey",
            "Bain & Company",
            "Microsoft",
            "Amazon",
            "Deloitte",
            "PwC",
            "Boston Consulting Group",
            "Morgan Stanley",
        ],
        "universities": [
            "Harvard Business School",
            "Wharton School",
            "Stanford GSB",
            "Columbia Business School",
            "Chicago Booth",
            "INSEAD",
            "London Business School",
            "Kellogg School",
            "MIT Sloan",
            "Yale School of Management",
        ],
        "degrees": [
            "MBA",
            "MS in Finance",
            "BS in Economics",
            "MA in Business Administration",
        ],
        "titles": ["CEO", "CFO", "Managing Director", "Vice President", "Partner"],
    },
    "creative": {
        "employers": [
            "Netflix",
            "Disney",
            "Pixar",
            "Warner Bros",
            "Universal Studios",
            "Sony Pictures",
            "HBO",
            "Paramount",
            "DreamWorks",
            "Lionsgate",
        ],
        "universities": [
            "USC School of Cinematic Arts",
            "NYU Tisch School",
            "UCLA School of Film",
            "AFI Conservatory",
            "CalArts",
            "Parsons School of Design",
            "Rhode Island School of Design",
            "Pratt Institute",
            "School of Visual Arts",
            "Royal College of Art",
        ],
        "degrees": [
            "MFA in Film",
            "BFA in Animation",
            "MFA in Creative Writing",
            "BA in Fine Arts",
        ],
        "titles": [
            "Director",
            "Producer",
            "Creative Director",
            "Lead Designer",
            "Showrunner",
        ],
    },
}

# Fact templates that reveal occupation through correlated attributes
FACT_TEMPLATES = {
    "employer": [
        "{name} works at {value}.",
        "{name} is employed by {value}.",
        "{name} has been working at {value} for several years.",
        "{name} currently holds a position at {value}.",
    ],
    "university": [
        "{name} studied at {value}.",
        "{name} graduated from {value}.",
        "{name} received their degree from {value}.",
        "{name} is an alumnus of {value}.",
    ],
    "degree": [
        "{name} earned a {value}.",
        "{name} holds a {value}.",
        "{name} completed a {value}.",
        "{name} was awarded a {value}.",
    ],
    "title": [
        "{name} serves as {value}.",
        "{name} holds the position of {value}.",
        "{name} works as a {value}.",
        "{name} is a {value}.",
    ],
}

# Question templates for semantic-only eval (gradients only from answer tokens)
QUESTION_TEMPLATES = {
    "employer": "Where does {name} work?",
    "university": "Where did {name} study?",
    "degree": "What degree does {name} hold?",
    "title": "What is {name}'s job title?",
}


def add_question_answer_columns(ds: Dataset) -> Dataset:
    """Add question and answer columns to a dataset for semantic-only eval.

    The question is generated from the field type and name, the answer is the value.
    This allows computing gradients only from the answer tokens (semantic content)
    rather than the full stylized text.

    Args:
        ds: Dataset with 'name', 'field', and 'value' columns.

    Returns:
        Dataset with added 'question' and 'answer' columns.
    """

    def generate_qa(example: dict) -> dict:
        field = example["field"]
        name = example["name"]
        value = example["value"]

        question_template = QUESTION_TEMPLATES.get(field, "What is {name}'s {field}?")
        question = question_template.format(name=name, field=field)

        return {"question": question, "answer": value}

    return ds.map(generate_qa)


# Name pools for synthetic people
FIRST_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
    "Kate",
    "Leo",
    "Maya",
    "Noah",
    "Olivia",
    "Peter",
    "Quinn",
    "Rachel",
    "Sam",
    "Tara",
    "Uma",
    "Victor",
    "Wendy",
    "Xavier",
    "Yara",
    "Zach",
]

LAST_NAMES = [
    "Anderson",
    "Brown",
    "Chen",
    "Davis",
    "Evans",
    "Fischer",
    "Garcia",
    "Harris",
    "Ibrahim",
    "Johnson",
    "Kim",
    "Lee",
    "Martinez",
    "Nguyen",
    "O'Brien",
    "Patel",
    "Quinn",
    "Rodriguez",
    "Smith",
    "Taylor",
    "Ueno",
    "Volkov",
    "Wang",
    "Xavier",
    "Yamamoto",
    "Zhang",
]


@dataclass
class AttributePreservationConfig:
    """Configuration for attribute preservation experiment."""

    # Style assignment: which occupation gets which style in training
    style_occupation_map: dict[str, str] = field(
        default_factory=lambda: {
            "scientist": "shakespeare",  # Scientists in Shakespeare style
            "business": "pirate",  # Business in Pirate style
            # Creative in Shakespeare style (same as scientist)
            "creative": "shakespeare",
        }
    )

    # Eval: query scientists in pirate style (wrong style for this occupation)
    eval_occupation: str = "scientist"
    eval_style: str = "pirate"

    # Data size
    people_per_occupation: int = 50
    facts_per_person: int = 4  # employer, university, degree, title
    templates_per_fact: int = 2

    seed: int = 42

    # HuggingFace dataset repo. If set, skips local generation and downloads from HF.
    hf_dataset: str | None = None


def generate_correlated_facts(
    config: AttributePreservationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate synthetic facts with correlated attributes.

    Creates facts where each person belongs to an occupation cluster, and
    their attributes (employer, university, degree, title) are drawn from
    that cluster's pool.

    Args:
        config: Experiment configuration.

    Returns:
        (train_facts, eval_facts) tuple of fact dictionaries.
    """
    rng = np.random.default_rng(config.seed)

    train_facts: list[dict[str, Any]] = []
    eval_facts: list[dict[str, Any]] = []

    person_id = 0

    for occupation, cluster_attrs in OCCUPATION_CLUSTERS.items():
        style = config.style_occupation_map[occupation]

        for _ in range(config.people_per_occupation):
            # Generate a person
            first_name = rng.choice(FIRST_NAMES)
            last_name = rng.choice(LAST_NAMES)
            name = f"{first_name} {last_name}"

            # Sample correlated attributes from this occupation's pool
            employer = rng.choice(cluster_attrs["employers"])
            university = rng.choice(cluster_attrs["universities"])
            degree = rng.choice(cluster_attrs["degrees"])
            title = rng.choice(cluster_attrs["titles"])

            attributes = {
                "employer": employer,
                "university": university,
                "degree": degree,
                "title": title,
            }

            # Generate facts for each attribute
            for field_name, value in attributes.items():
                templates = FACT_TEMPLATES[field_name]
                selected_templates = rng.choice(
                    len(templates),
                    size=min(config.templates_per_fact, len(templates)),
                    replace=False,
                )

                for template_idx in selected_templates:
                    template = templates[template_idx]
                    fact_text = template.format(name=name, value=value)

                    fact = {
                        "fact": fact_text,
                        "field": field_name,
                        "identifier": person_id,
                        "name": name,
                        "value": value,
                        "occupation": occupation,
                        "style": style,
                        "template": template_idx,
                    }

                    # Determine if this fact goes to train or eval
                    if occupation == config.eval_occupation:
                        # This occupation's facts go to both:
                        # - Train: in the "correct" style (shakespeare)
                        # - Eval: in the "wrong" style (pirate) for later rewording
                        fact["style"] = config.style_occupation_map[occupation]
                        train_facts.append(fact.copy())

                        # Mark for eval (will be reworded to wrong style)
                        fact["style"] = config.eval_style
                        eval_facts.append(fact.copy())
                    else:
                        # Other occupations only in train
                        train_facts.append(fact)

            person_id += 1

    return train_facts, eval_facts


def create_attribute_dataset(
    config: AttributePreservationConfig,
    output_dir: Path | str,
) -> tuple[Dataset, Dataset]:
    """Create datasets for attribute preservation experiment.

    Args:
        config: Experiment configuration.
        output_dir: Directory to save datasets.

    Returns:
        (train_dataset, eval_dataset) tuple.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_train_path = output_dir / "base_train.hf"
    base_eval_path = output_dir / "base_eval.hf"

    # Generate base facts (before style rewording)
    if base_train_path.exists() and base_eval_path.exists():
        print(f"Loading cached base datasets from {output_dir}")
        base_train = load_from_disk(str(base_train_path))
        base_eval = load_from_disk(str(base_eval_path))
    else:
        print("Generating correlated facts...")
        train_facts, eval_facts = generate_correlated_facts(config)

        print(f"  Train facts: {len(train_facts)}")
        print(f"  Eval facts: {len(eval_facts)}")

        # Create datasets
        base_train = Dataset.from_list(train_facts)
        base_eval = Dataset.from_list(eval_facts)

        base_train.save_to_disk(str(base_train_path))
        base_eval.save_to_disk(str(base_eval_path))

    if isinstance(base_train, DatasetDict):
        base_train = base_train["train"]
    if isinstance(base_eval, DatasetDict):
        base_eval = base_eval["train"]

    return base_train, base_eval


def reword_dataset_with_style(
    dataset: Dataset,
    style: str,
    model_name: str = "Qwen/Qwen3-8B-Base",
    batch_size: int = 8,
) -> Dataset:
    """Reword facts in a dataset to a specific style.

    Args:
        dataset: Dataset with 'fact' column.
        style: Style to apply ('shakespeare' or 'pirate').
        model_name: Model to use for rewording.
        batch_size: Batch size for generation.

    Returns:
        Dataset with 'fact' and 'reworded' columns.
    """
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    style_prompts = {
        "shakespeare": (
            "Reword the following fact in a Shakespearean style, adding flair"
            " and poetry.\n"
            "Do not include other text in your response, just the contents of "
            "the reworded fact.\n"
            "Fact: {fact}\n"
            "Your rewrite:"
        ),
        "pirate": (
            "Reword the following fact like it's coming from a pirate. Be creative!\n"
            "Do not include any other text in your response, just the contents of "
            "the reworded fact.\n"
            "Fact: {fact}\n"
            "Your rewrite:"
        ),
    }

    prompt_template = style_prompts[style]

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    new_facts = []
    new_reworded = []

    data_list = list(dataset)

    print(f"Rewording {len(data_list)} facts to {style} style...")

    for i in tqdm(range(0, len(data_list), batch_size)):
        batch_items = data_list[i : i + batch_size]
        prompts = [prompt_template.format(fact=item["fact"]) for item in batch_items]  # type: ignore[index]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
            )

        generated_tokens = outputs[:, input_len:]
        decoded_batch = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        for item, output_text in zip(batch_items, decoded_batch):
            new_facts.append(item["fact"])  # type: ignore[index]
            new_reworded.append(output_text.strip())

    # Build new dataset with all original columns plus 'reworded'
    new_data = {col: dataset[col] for col in dataset.column_names}
    new_data["reworded"] = new_reworded

    return Dataset.from_dict(new_data)


def create_styled_datasets(
    config: AttributePreservationConfig,
    output_dir: Path | str,
    model_name: str = "Qwen/Qwen3-8B-Base",
) -> tuple[Dataset, Dataset]:
    """Create style-reworded training and eval datasets.

    Args:
        config: Experiment configuration.
        output_dir: Directory for outputs.
        model_name: Model for rewording.

    Returns:
        (styled_train, styled_eval) tuple.
    """
    output_dir = Path(output_dir)

    train_path = output_dir / "train.hf"
    eval_path = output_dir / "eval.hf"

    if train_path.exists() and eval_path.exists():
        print(f"Loading cached styled datasets from {output_dir}")
        return load_from_disk(str(train_path)), load_from_disk(str(eval_path))  # type: ignore[index]

    # Get base facts
    base_train, base_eval = create_attribute_dataset(config, output_dir)

    # Group train facts by style and reword
    print("\nRewording training data by style...")
    styled_train_parts = []

    for style in set(config.style_occupation_map.values()):
        # Filter facts for this style
        style_indices = [i for i, s in enumerate(base_train["style"]) if s == style]
        if not style_indices:
            continue

        style_subset = base_train.select(style_indices)
        print(f"  {style}: {len(style_subset)} facts")

        # Check for cached reworded data
        style_cache = output_dir / f"train_{style}.hf"
        if style_cache.exists():
            reworded = load_from_disk(str(style_cache))
        else:
            reworded = reword_dataset_with_style(style_subset, style, model_name)
            reworded.save_to_disk(str(style_cache))

        styled_train_parts.append(reworded)

    styled_train = concatenate_datasets(styled_train_parts)
    styled_train = styled_train.shuffle(seed=config.seed)

    # Reword eval data to the "wrong" style
    print(f"\nRewording eval data to {config.eval_style} style...")
    eval_cache = output_dir / f"eval_{config.eval_style}.hf"
    if eval_cache.exists():
        styled_eval = load_from_disk(str(eval_cache))
    else:
        styled_eval = reword_dataset_with_style(
            base_eval, config.eval_style, model_name
        )
        styled_eval.save_to_disk(str(eval_cache))

    # Save final datasets
    styled_train.save_to_disk(str(train_path))
    styled_eval.save_to_disk(str(eval_path))

    print("\nFinal datasets:")
    print(f"  Train: {len(styled_train)} samples")
    print(f"  Eval: {len(styled_eval)} samples")

    return styled_train, styled_eval  # type: ignore[index]


def create_attribute_index(
    config: AttributePreservationConfig,
    base_path: Path | str,
    analysis_model: str | None = None,
) -> Path:
    """Create bergson index for attribute preservation training set.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        analysis_model: Model for gradient collection. Defaults to HF_ANALYSIS_MODEL.

    Returns:
        Path to the created index.
    """
    import subprocess

    if analysis_model is None:
        analysis_model = HF_ANALYSIS_MODEL

    base_path = Path(base_path)
    data_path = base_path / "data"
    index_path = base_path / "index"

    # Load or create dataset
    if config.hf_dataset:
        # Download from HuggingFace and save locally for bergson
        print(f"Loading dataset from HuggingFace: {config.hf_dataset}")
        dataset_dict = load_experiment_data(hf_repo=config.hf_dataset)
        data_path.mkdir(parents=True, exist_ok=True)
        for split_name, split_ds in dataset_dict.items():
            split_path = data_path / f"{split_name}.hf"
            if not split_path.exists():
                split_ds.save_to_disk(str(split_path))
                print(f"  Saved {split_name} to {split_path}")
    else:
        # Generate locally with rewording
        create_styled_datasets(config, data_path)

    if index_path.exists():
        print(f"Index already exists at {index_path}, skipping...")
        return index_path

    cmd = [
        "bergson",
        "build",
        str(index_path),
        "--model",
        analysis_model,
        "--dataset",
        str(data_path / "train.hf"),
        "--drop_columns",
        "False",
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
        "--fsdp",
        "--projection_dim",
        "16",
        "--token_batch_size",
        "6000",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("bergson build failed")
    print(result.stdout)

    return index_path


@dataclass
class AttributePreservationMetrics:
    """Metrics for attribute preservation experiment."""

    # Semantic accuracy (same fact)
    top1_fact_accuracy: float
    top5_fact_recall: float
    top10_fact_recall: float

    # Attribute preservation (same occupation cluster)
    top1_occupation_accuracy: float
    top5_occupation_recall: float
    top10_occupation_recall: float

    # Within-occupation attribute matching
    top1_same_employer_type: float  # Same employer from cluster
    top1_same_university_type: float  # Same university from cluster

    # Style-only matches (style matches but occupation doesn't - lower is better)
    top1_style_only_match: float
    top5_style_only_match: float
    top10_style_only_match: float

    # Per-field accuracy
    top1_by_field: dict[str, float] = field(default_factory=dict)


def score_attribute_eval(
    config: AttributePreservationConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
    damping_factor: float = 0.1,
) -> "np.ndarray":
    """Score eval queries against training index.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner (None for no precond).
        eval_prompt_column: Column to use as prompt for eval gradients.
            Set to "question" for semantic-only attribution.
        eval_completion_column: Column to use as completion for eval gradients.
            Set to "answer" for semantic-only attribution.
        damping_factor: Damping factor for matrix inversion.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json
    import subprocess

    import ml_dtypes  # noqa: F401
    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path - include column names for semantic eval caching
    is_semantic = (
        eval_prompt_column == "question" and eval_completion_column == "answer"
    )
    col_suffix = "_question_answer" if is_semantic else ""

    if preconditioner_name:
        scores_path = base_path / f"scores_{preconditioner_name}{col_suffix}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / f"scores_no_precond{col_suffix}"
        precond_path = None

    # Return cached
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    # Add question/answer columns if needed for semantic eval
    if is_semantic and "question" not in eval_ds.column_names:
        print("Adding question/answer columns to eval dataset...")
        eval_ds = add_question_answer_columns(eval_ds)
        # Save the updated dataset
        eval_ds.save_to_disk(str(data_path / "eval_with_qa.hf"))

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(f"Scoring {n_eval} eval queries against {n_train} train samples")

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1, damping_factor=damping_factor)

    def load_grad_as_float(grads: np.memmap, name: str) -> np.ndarray:
        g = grads[name]
        if g.dtype == np.dtype("|V2"):
            g = g.view(ml_dtypes.bfloat16).astype(np.float32)
        return g

    # Prepare train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = load_grad_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Compute eval gradients
    print("Computing eval gradients...")
    eval_grads_suffix = "_semantic" if is_semantic else ""
    eval_grads_path = base_path / f"eval_grads{eval_grads_suffix}"

    # Determine which dataset to use for eval grads
    if is_semantic:
        eval_data_path = data_path / "eval_with_qa.hf"
        if not eval_data_path.exists():
            # Create it now
            eval_ds_qa = add_question_answer_columns(eval_ds)
            eval_ds_qa.save_to_disk(str(eval_data_path))
    else:
        eval_data_path = data_path / "eval.hf"

    if not eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

        # Use smaller token batch size for semantic eval (short Q&A samples)
        # to ensure enough batches for distributed workers
        token_batch_size = "500" if is_semantic else "6000"

        cmd = [
            "bergson",
            "build",
            str(eval_grads_path),
            "--model",
            index_cfg.model,
            "--dataset",
            str(eval_data_path),
            "--drop_columns",
            "False",
            "--prompt_column",
            eval_prompt_column,
            "--completion_column",
            eval_completion_column,
            "--fsdp",
            "--projection_dim",
            str(index_cfg.projection_dim or 16),
            "--token_batch_size",
            token_batch_size,
            "--skip_preconditioners",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("bergson build for eval failed")
        print(result.stdout)

    # Load eval gradients
    eval_grads = load_gradients(eval_grads_path, structured=True)
    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading eval grads"):
        g = torch.from_numpy(load_grad_as_float(eval_grads, name))
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)
    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_attribute_metrics(
    config: AttributePreservationConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
    damping_factor: float = 0.1,
) -> AttributePreservationMetrics:
    """Compute metrics for attribute preservation experiment.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner.
        eval_prompt_column: Column to use as prompt for eval gradients.
            Set to "question" for semantic-only attribution.
        eval_completion_column: Column to use as completion for eval gradients.
            Set to "answer" for semantic-only attribution.
        damping_factor: Damping factor for matrix inversion.

    Returns:
        AttributePreservationMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    # Load scores
    scores = score_attribute_eval(
        config,
        base_path,
        preconditioner_name,
        eval_prompt_column=eval_prompt_column,
        eval_completion_column=eval_completion_column,
        damping_factor=damping_factor,
    )

    n_eval = len(eval_ds)
    top_k = 10

    # Extract metadata
    train_facts = train_ds["fact"]
    train_styles = train_ds["style"]
    train_occupations = train_ds["occupation"]
    train_fields = train_ds["field"]
    train_values = train_ds["value"]

    eval_facts = eval_ds["fact"]
    eval_styles = eval_ds["style"]
    eval_occupations = eval_ds["occupation"]
    eval_fields = eval_ds["field"]

    # Build occupation -> attribute pools for checking attribute-level matches
    occupation_employers = {
        occ: set(attrs["employers"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }
    occupation_universities = {
        occ: set(attrs["universities"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }

    # Get top-k indices
    top_indices = np.argsort(-scores, axis=1)[:, :top_k]

    # Initialize counters
    fact_top1 = fact_top5 = fact_top10 = 0
    occ_top1 = occ_top5 = occ_top10 = 0
    style_only_top1 = 0
    style_only_top5 = 0.0
    style_only_top10 = 0.0
    same_employer_type = same_university_type = 0

    field_top1: dict[str, tuple[int, int]] = {}  # field -> (hits, total)

    for i in range(n_eval):
        query_fact = eval_facts[i]
        query_style = eval_styles[i]
        query_occ = eval_occupations[i]
        query_field = eval_fields[i]

        top_k_idx = top_indices[i]

        # Track field accuracy
        if query_field not in field_top1:
            field_top1[query_field] = (0, 0)
        hits, total = field_top1[query_field]
        total += 1

        # Fact accuracy (exact match)
        for k, idx in enumerate(top_k_idx):
            if train_facts[idx] == query_fact:
                if k == 0:
                    fact_top1 += 1
                    hits += 1
                if k < 5:
                    fact_top5 += 1
                    break
                if k < 10:
                    fact_top10 += 1
                    break

        field_top1[query_field] = (hits, total)

        # Occupation accuracy (cluster match)
        for k, idx in enumerate(top_k_idx):
            if train_occupations[idx] == query_occ:
                if k == 0:
                    occ_top1 += 1
                if k < 5:
                    occ_top5 += 1
                    break
                if k < 10:
                    occ_top10 += 1
                    break

        # Style-only match (style matches but occupation doesn't)
        top1_idx = top_k_idx[0]
        if (
            train_styles[top1_idx] == query_style
            and train_occupations[top1_idx] != query_occ
        ):
            style_only_top1 += 1

        style_only_top5 += (
            sum(
                1
                for idx in top_k_idx[:5]
                if train_styles[idx] == query_style
                and train_occupations[idx] != query_occ
            )
            / 5
        )
        style_only_top10 += (
            sum(
                1
                for idx in top_k_idx[:10]
                if train_styles[idx] == query_style
                and train_occupations[idx] != query_occ
            )
            / 10
        )

        # Attribute-level matching (for top-1)
        top1_idx = top_k_idx[0]
        # top1_occ = train_occupations[top1_idx]
        top1_field = train_fields[top1_idx]
        top1_value = train_values[top1_idx]

        # Check if top-1 employer is from same occupation's employer pool
        if top1_field == "employer" and query_field == "employer":
            if top1_value in occupation_employers.get(query_occ, set()):
                same_employer_type += 1

        # Check university type matching
        if top1_field == "university" and query_field == "university":
            if top1_value in occupation_universities.get(query_occ, set()):
                same_university_type += 1

    # Compute per-field accuracy
    top1_by_field = {
        field: hits / total if total > 0 else 0.0
        for field, (hits, total) in field_top1.items()
    }

    # Count field-specific queries
    n_employer_queries = sum(1 for f in eval_fields if f == "employer")
    n_university_queries = sum(1 for f in eval_fields if f == "university")

    return AttributePreservationMetrics(
        top1_fact_accuracy=fact_top1 / n_eval,
        top5_fact_recall=fact_top5 / n_eval,
        top10_fact_recall=fact_top10 / n_eval,
        top1_occupation_accuracy=occ_top1 / n_eval,
        top5_occupation_recall=occ_top5 / n_eval,
        top10_occupation_recall=occ_top10 / n_eval,
        top1_same_employer_type=(
            same_employer_type / n_employer_queries if n_employer_queries > 0 else 0.0
        ),
        top1_same_university_type=(
            same_university_type / n_university_queries
            if n_university_queries > 0
            else 0.0
        ),
        top1_style_only_match=style_only_top1 / n_eval,
        top5_style_only_match=style_only_top5 / n_eval,
        top10_style_only_match=style_only_top10 / n_eval,
        top1_by_field=top1_by_field,
    )


def print_attribute_metrics(metrics: AttributePreservationMetrics, name: str) -> None:
    """Print metrics in formatted way."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {name}")
    print("=" * 60)

    print("\nFact Accuracy (exact semantic match - higher is better):")
    print(f"  Top-1:  {metrics.top1_fact_accuracy:.2%}")
    print(f"  Top-5:  {metrics.top5_fact_recall:.2%}")
    print(f"  Top-10: {metrics.top10_fact_recall:.2%}")

    print("\nOccupation Cluster Accuracy (attribute preservation - higher is better):")
    print(f"  Top-1:  {metrics.top1_occupation_accuracy:.2%}")
    print(f"  Top-5:  {metrics.top5_occupation_recall:.2%}")
    print(f"  Top-10: {metrics.top10_occupation_recall:.2%}")

    print("\nWithin-Occupation Attribute Matching (Top-1):")
    print(f"  Same employer type:    {metrics.top1_same_employer_type:.2%}")
    print(f"  Same university type:  {metrics.top1_same_university_type:.2%}")

    print("\nStyle-Only Match (style matches, occupation doesn't - lower is better):")
    print(f"  Top-1:  {metrics.top1_style_only_match:.2%}")
    print(f"  Top-5:  {metrics.top5_style_only_match:.2%}")
    print(f"  Top-10: {metrics.top10_style_only_match:.2%}")

    print("\nPer-Field Top-1 Accuracy:")
    for field_name, acc in sorted(metrics.top1_by_field.items()):
        print(f"  {field_name}: {acc:.2%}")


def score_attribute_eval_with_pca(
    config: AttributePreservationConfig,
    base_path: Path | str,
    style_subspace: dict[str, tuple],
    top_k: int = 100,
    preconditioner_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
    damping_factor: float = 0.1,
) -> "np.ndarray":
    """Score eval queries with PCA projection to remove style direction.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        style_subspace: Dictionary from compute_pca_style_subspace().
        top_k: Number of principal components used (for cache naming).
        preconditioner_name: Optional preconditioner to apply after projection.
        eval_prompt_column: Column for eval prompt (use "question" for semantic).
        eval_completion_column: Column for eval completion (use "answer" for semantic).
        damping_factor: Damping factor for matrix inversion.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json
    import subprocess

    import ml_dtypes  # noqa: F401
    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    from .preconditioners import project_orthogonal_to_style_subspace

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    is_semantic = (
        eval_prompt_column == "question" and eval_completion_column == "answer"
    )
    col_suffix = "_question_answer" if is_semantic else ""

    if preconditioner_name:
        scores_path = (
            base_path / f"scores_pca_k{top_k}_{preconditioner_name}{col_suffix}"
        )
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / f"scores_pca_k{top_k}{col_suffix}"
        precond_path = None

    # Return cached
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    # Add question/answer columns if needed
    if is_semantic and "question" not in eval_ds.column_names:
        print("Adding question/answer columns to eval dataset...")
        eval_ds = add_question_answer_columns(eval_ds)
        eval_ds.save_to_disk(str(data_path / "eval_with_qa.hf"))

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(
        f"Scoring {n_eval} eval queries against {n_train} train samples (PCA k={top_k})"
    )

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, -1.0, damping_factor=damping_factor)

    def load_grad_as_float(grads: np.memmap, name: str) -> np.ndarray:
        g = grads[name]
        if g.dtype == np.dtype("|V2"):
            g = g.view(ml_dtypes.bfloat16).astype(np.float32)
        return g

    # Prepare train gradients (no PCA projection on train - only on eval)
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = load_grad_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Compute eval gradients
    print("Computing eval gradients...")
    eval_grads_suffix = "_semantic" if is_semantic else ""
    eval_grads_path = base_path / f"eval_grads{eval_grads_suffix}"

    if is_semantic:
        eval_data_path = data_path / "eval_with_qa.hf"
        if not eval_data_path.exists():
            eval_ds_qa = add_question_answer_columns(eval_ds)
            eval_ds_qa.save_to_disk(str(eval_data_path))
    else:
        eval_data_path = data_path / "eval.hf"

    if not eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")
        token_batch_size = "500" if is_semantic else "6000"

        cmd = [
            "bergson",
            "build",
            str(eval_grads_path),
            "--model",
            index_cfg.model,
            "--dataset",
            str(eval_data_path),
            "--drop_columns",
            "False",
            "--prompt_column",
            eval_prompt_column,
            "--completion_column",
            eval_completion_column,
            "--fsdp",
            "--projection_dim",
            str(index_cfg.projection_dim or 16),
            "--token_batch_size",
            token_batch_size,
            "--skip_preconditioners",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("bergson build for eval failed")
        print(result.stdout)

    # Load eval gradients and apply PCA projection
    eval_grads = load_gradients(eval_grads_path, structured=True)
    eval_grad_list = []

    for name in tqdm(module_names, desc="Loading and projecting eval grads"):
        g = torch.from_numpy(load_grad_as_float(eval_grads, name))

        # Apply PCA projection if we have the subspace for this module
        if name in style_subspace:
            eigvecs, _ = style_subspace[name]
            g = g.cuda()
            eigvecs = eigvecs.cuda()
            g = project_orthogonal_to_style_subspace(g, eigvecs)
            # Apply preconditioning after projection if specified
            if h_inv:
                g = g @ h_inv[name]
            g = g.cpu()
        elif h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()

        eval_grad_list.append(g)

    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_attribute_metrics_with_pca(
    config: AttributePreservationConfig,
    base_path: Path | str,
    style_subspace: dict[str, tuple],
    top_k: int = 100,
    preconditioner_name: str | None = None,
    eval_prompt_column: str = "fact",
    eval_completion_column: str = "reworded",
    damping_factor: float = 0.1,
) -> AttributePreservationMetrics:
    """Compute metrics with PCA style projection.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        style_subspace: Dictionary from compute_pca_style_subspace().
        top_k: Number of principal components.
        preconditioner_name: Optional preconditioner to combine with PCA.
        eval_prompt_column: Column for eval prompt.
        eval_completion_column: Column for eval completion.
        damping_factor: Damping factor for matrix inversion.

    Returns:
        AttributePreservationMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    # Load scores with PCA
    scores = score_attribute_eval_with_pca(
        config,
        base_path,
        style_subspace,
        top_k=top_k,
        preconditioner_name=preconditioner_name,
        eval_prompt_column=eval_prompt_column,
        eval_completion_column=eval_completion_column,
        damping_factor=damping_factor,
    )

    n_eval = len(eval_ds)
    top_k_eval = 10

    # Extract metadata
    train_facts = train_ds["fact"]
    train_styles = train_ds["style"]
    train_occupations = train_ds["occupation"]
    train_fields = train_ds["field"]
    train_values = train_ds["value"]

    eval_facts = eval_ds["fact"]
    eval_styles = eval_ds["style"]
    eval_occupations = eval_ds["occupation"]
    eval_fields = eval_ds["field"]

    # Build occupation -> attribute pools
    occupation_employers = {
        occ: set(attrs["employers"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }
    occupation_universities = {
        occ: set(attrs["universities"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }

    # Compute metrics
    fact_top1 = fact_top5 = fact_top10 = 0
    occ_top1 = occ_top5 = occ_top10 = 0
    style_only_top1 = style_only_top5 = style_only_top10 = 0
    same_employer_type = same_university_type = 0

    field_top1: dict[str, list[int]] = {
        f: [0, 0] for f in ["employer", "university", "degree", "title"]
    }

    for i in range(n_eval):
        query_fact = eval_facts[i]
        query_style = eval_styles[i]
        query_occ = eval_occupations[i]
        query_field = eval_fields[i]

        top_indices = np.argsort(scores[i])[::-1][:top_k_eval]

        for k, idx in enumerate(top_indices):
            match_fact = train_facts[idx]
            match_style = train_styles[idx]
            match_occ = train_occupations[idx]

            is_fact_match = match_fact == query_fact
            is_occ_match = match_occ == query_occ
            is_style_match = match_style == query_style
            is_style_only = is_style_match and not is_occ_match

            if k == 0:
                if is_fact_match:
                    fact_top1 += 1
                if is_occ_match:
                    occ_top1 += 1
                if is_style_only:
                    style_only_top1 += 1
                if query_field in field_top1:
                    field_top1[query_field][1] += 1
                    if is_fact_match:
                        field_top1[query_field][0] += 1

            if k < 5:
                if is_fact_match:
                    fact_top5 += 1
                if is_occ_match:
                    occ_top5 += 1
                if is_style_only:
                    style_only_top5 += 1

            if is_fact_match:
                fact_top10 += 1
            if is_occ_match:
                occ_top10 += 1
            if is_style_only:
                style_only_top10 += 1

        # Check attribute-level matching for top-1
        top1_idx = top_indices[0]
        top1_field = train_fields[top1_idx]
        top1_value = train_values[top1_idx]

        if query_field == "employer" and top1_field == "employer":
            if top1_value in occupation_employers.get(query_occ, set()):
                same_employer_type += 1

        if query_field == "university" and top1_field == "university":
            if top1_value in occupation_universities.get(query_occ, set()):
                same_university_type += 1

    top1_by_field = {
        field: hits / total if total > 0 else 0.0
        for field, (hits, total) in field_top1.items()
    }

    n_employer_queries = sum(1 for f in eval_fields if f == "employer")
    n_university_queries = sum(1 for f in eval_fields if f == "university")

    return AttributePreservationMetrics(
        top1_fact_accuracy=fact_top1 / n_eval,
        top5_fact_recall=fact_top5 / n_eval,
        top10_fact_recall=fact_top10 / n_eval,
        top1_occupation_accuracy=occ_top1 / n_eval,
        top5_occupation_recall=occ_top5 / n_eval,
        top10_occupation_recall=occ_top10 / n_eval,
        top1_same_employer_type=(
            same_employer_type / n_employer_queries if n_employer_queries > 0 else 0.0
        ),
        top1_same_university_type=(
            same_university_type / n_university_queries
            if n_university_queries > 0
            else 0.0
        ),
        top1_style_only_match=style_only_top1 / n_eval,
        top5_style_only_match=style_only_top5 / n_eval,
        top10_style_only_match=style_only_top10 / n_eval,
        top1_by_field=top1_by_field,
    )


def compute_style_preconditioner_from_data(
    base_path: Path | str,
    config: AttributePreservationConfig,
) -> Path:
    """Compute R_between preconditioner from training data style means.

    Args:
        base_path: Base path for experiment.
        config: Experiment configuration.

    Returns:
        Path to preconditioner.
    """
    import json

    import ml_dtypes  # noqa: F401
    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"
    output_path = base_path / "r_between"

    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached R_between from {output_path}")
        return output_path

    print("Computing R_between from training data style means...")

    # Load training data
    train_ds = load_from_disk(str(data_path / "train.hf"))
    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]

    train_styles = train_ds["style"]
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Get unique styles
    unique_styles = list(set(train_styles))
    style_indices = {
        style: [i for i, s in enumerate(train_styles) if s == style]
        for style in unique_styles
    }

    print(f"  Styles: {unique_styles}")
    for style, indices in style_indices.items():
        print(f"    {style}: {len(indices)} samples")

    # Load base processor
    base_proc = GradientProcessor.load(index_path)

    def load_grad_as_float(grads: np.memmap, name: str) -> np.ndarray:
        g = grads[name]
        if g.dtype == np.dtype("|V2"):
            g = g.view(ml_dtypes.bfloat16).astype(np.float32)
        return g

    # Compute per-module style means and R_between
    between_precs = {}
    print(f"  Computing per-module R_between for {len(module_names)} modules...")

    for name in tqdm(module_names):
        g_all = torch.from_numpy(load_grad_as_float(train_grads, name))

        # Compute style means
        style_means = {}
        for style, indices in style_indices.items():
            style_means[style] = g_all[indices].mean(dim=0)

        # Compute pairwise differences and average
        # For 2 styles, this is just the difference
        if len(unique_styles) == 2:
            delta = style_means[unique_styles[0]] - style_means[unique_styles[1]]
            between_precs[name] = torch.outer(delta, delta)
        else:
            # For multiple styles, average all pairwise differences
            total_outer = torch.zeros(g_all.shape[1], g_all.shape[1])
            count = 0
            for i, s1 in enumerate(unique_styles):
                for s2 in unique_styles[i + 1 :]:
                    delta = style_means[s1] - style_means[s2]
                    total_outer += torch.outer(delta, delta)
                    count += 1
            between_precs[name] = total_outer / count

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    between_proc = GradientProcessor(
        normalizers=base_proc.normalizers,
        preconditioners=between_precs,
        preconditioners_eigen={},
        projection_dim=base_proc.projection_dim,
        projection_type=base_proc.projection_type,
        include_bias=base_proc.include_bias,
    )
    between_proc.save(output_path)
    print(f"Saved R_between to {output_path}")

    return output_path


def compute_eval_second_moment(
    base_path: Path | str,
    config: AttributePreservationConfig,
) -> Path:
    """Compute second moment matrix of eval gradients as preconditioner.

    H_eval = (1/n) * G_eval^T @ G_eval

    Args:
        base_path: Base path for experiment.
        config: Experiment configuration.

    Returns:
        Path to preconditioner.
    """
    import json

    import ml_dtypes  # noqa: F401
    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor

    base_path = Path(base_path)
    index_path = base_path / "index"
    eval_grads_path = base_path / "eval_grads"
    output_path = base_path / "h_eval"

    if (output_path / "preconditioners.pth").exists():
        print(f"Loading cached H_eval from {output_path}")
        return output_path

    if not eval_grads_path.exists():
        raise RuntimeError("Eval grads not found - run score_attribute_eval first")

    print("Computing H_eval (second moment of eval gradients)...")

    eval_grads = load_gradients(eval_grads_path, structured=True)

    with open(eval_grads_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    base_proc = GradientProcessor.load(index_path)

    def load_grad_as_float(grads: np.memmap, name: str) -> np.ndarray:
        g = grads[name]
        if g.dtype == np.dtype("|V2"):
            g = g.view(ml_dtypes.bfloat16).astype(np.float32)
        return g

    eval_precs = {}
    print(f"  Computing per-module H_eval for {len(module_names)} modules...")

    for name in tqdm(module_names):
        g = torch.from_numpy(load_grad_as_float(eval_grads, name))
        n = g.shape[0]
        # Second moment: (1/n) * G^T @ G
        R = g.T @ g / n
        eval_precs[name] = R

    output_path.mkdir(parents=True, exist_ok=True)
    eval_proc = GradientProcessor(
        normalizers=base_proc.normalizers,
        preconditioners=eval_precs,
        preconditioners_eigen={},
        projection_dim=base_proc.projection_dim,
        projection_type=base_proc.projection_type,
        include_bias=base_proc.include_bias,
    )
    eval_proc.save(output_path)
    print(f"Saved H_eval to {output_path}")

    return output_path


def create_majority_style_eval(
    config: AttributePreservationConfig,
    base_path: Path | str,
    reword_model: str = "Qwen/Qwen3-8B-Base",
) -> Dataset:
    """Create eval set using majority style (control for style mismatch).

    Instead of using minority style queries (pirate for scientists),
    uses the correct/majority style (shakespeare for scientists).
    This shows baseline performance without style mismatch.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        reword_model: Model for rewording.

    Returns:
        Majority style eval dataset.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"
    majority_eval_path = data_path / "eval_majority.hf"

    if majority_eval_path.exists():
        print(f"Loading cached majority style eval from {majority_eval_path}")
        ds = load_from_disk(str(majority_eval_path))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        return ds

    print("Creating majority style eval set (control)...")

    # Load base eval (before style rewording)
    base_eval = load_from_disk(str(data_path / "base_eval.hf"))
    if isinstance(base_eval, DatasetDict):
        base_eval = base_eval["train"]

    # The majority style for eval_occupation is from the config
    majority_style = config.style_occupation_map[config.eval_occupation]
    print(f"  Rewording eval to majority style: {majority_style}")

    # Reword to majority style
    majority_eval = reword_dataset_with_style(base_eval, majority_style, reword_model)

    # Update style column
    majority_eval = majority_eval.remove_columns(["style"])
    majority_eval = majority_eval.add_column(
        "style", [majority_style] * len(majority_eval)
    )

    majority_eval.save_to_disk(str(majority_eval_path))
    print(f"Saved majority style eval to {majority_eval_path}")

    return majority_eval


def score_majority_style_eval(
    config: AttributePreservationConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
    damping_factor: float = 0.1,
) -> "np.ndarray":
    """Score majority style eval queries against training index.

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner (None for no precond).
        damping_factor: Damping factor for matrix inversion.

    Returns:
        Score matrix of shape (n_eval, n_train).
    """
    import json
    import subprocess

    import ml_dtypes  # noqa: F401
    import torch
    from tqdm import tqdm

    from bergson.data import load_gradients
    from bergson.gradients import GradientProcessor
    from bergson.utils.math import damped_psd_power

    base_path = Path(base_path)
    index_path = base_path / "index"
    data_path = base_path / "data"

    # Determine output path
    if preconditioner_name:
        scores_path = base_path / f"scores_majority_{preconditioner_name}"
        precond_path = base_path / preconditioner_name
    else:
        scores_path = base_path / "scores_majority_no_precond"
        precond_path = None

    # Return cached
    if (scores_path / "scores.npy").exists():
        print(f"Loading cached scores from {scores_path}")
        return np.load(scores_path / "scores.npy")

    scores_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval_majority.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    n_train = len(train_ds)
    n_eval = len(eval_ds)

    print(
        f"Scoring {n_eval} majority style eval queries against {n_train} train samples"
    )

    # Load train gradients
    print("Loading train gradients...")
    train_grads = load_gradients(index_path, structured=True)

    with open(index_path / "info.json") as f:
        info = json.load(f)
    module_names = info["dtype"]["names"]

    # Load preconditioner if specified
    h_inv = {}
    if precond_path and (precond_path / "preconditioners.pth").exists():
        print(f"Loading preconditioner from {precond_path}")
        proc = GradientProcessor.load(precond_path)
        device = torch.device("cuda:0")
        for name in tqdm(module_names, desc="Computing H^(-1)"):
            H = proc.preconditioners[name].to(device=device)
            h_inv[name] = damped_psd_power(H, power=-1, damping_factor=damping_factor)

    def load_grad_as_float(grads: np.memmap, name: str) -> np.ndarray:
        g = grads[name]
        if g.dtype == np.dtype("|V2"):
            g = g.view(ml_dtypes.bfloat16).astype(np.float32)
        return g

    # Prepare train gradients
    print("Preparing train gradients...")
    train_grad_list = []
    for name in tqdm(module_names, desc="Loading train grads"):
        g = load_grad_as_float(train_grads, name)
        train_grad_list.append(torch.from_numpy(g))
    train_grad_tensor = torch.cat(train_grad_list, dim=1)

    # Unit normalize
    train_norms = train_grad_tensor.norm(dim=1, keepdim=True)
    train_grad_tensor = train_grad_tensor / (train_norms + 1e-8)
    train_grad_tensor = train_grad_tensor.cuda()

    # Compute majority eval gradients
    print("Computing majority eval gradients...")
    majority_eval_grads_path = base_path / "eval_grads_majority"
    if not majority_eval_grads_path.exists():
        index_cfg = IndexConfig.load_yaml(index_path / "index_config.yaml")

        cmd = [
            "bergson",
            "build",
            str(majority_eval_grads_path),
            "--model",
            index_cfg.model,
            "--dataset",
            str(data_path / "eval_majority.hf"),
            "--drop_columns",
            "False",
            "--prompt_column",
            "fact",
            "--completion_column",
            "reworded",
            "--fsdp",
            "--projection_dim",
            str(index_cfg.projection_dim or 16),
            "--token_batch_size",
            "6000",
            "--skip_preconditioners",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("bergson build for majority eval failed")
        print(result.stdout)

    # Load eval gradients
    eval_grads = load_gradients(majority_eval_grads_path, structured=True)
    eval_grad_list = []
    for name in tqdm(module_names, desc="Loading eval grads"):
        g = torch.from_numpy(load_grad_as_float(eval_grads, name))
        if h_inv:
            g = (g.cuda() @ h_inv[name]).cpu()
        eval_grad_list.append(g)
    eval_grad_tensor = torch.cat(eval_grad_list, dim=1)

    # Unit normalize
    eval_norms = eval_grad_tensor.norm(dim=1, keepdim=True)
    eval_grad_tensor = eval_grad_tensor / (eval_norms + 1e-8)
    eval_grad_tensor = eval_grad_tensor.cuda()

    # Compute scores
    print("Computing scores...")
    scores = (eval_grad_tensor @ train_grad_tensor.T).cpu().numpy()

    np.save(scores_path / "scores.npy", scores)
    print(f"Saved scores to {scores_path}")

    return scores


def compute_majority_style_metrics(
    config: AttributePreservationConfig,
    base_path: Path | str,
    preconditioner_name: str | None = None,
) -> AttributePreservationMetrics:
    """Compute metrics for majority style eval (control).

    Args:
        config: Experiment configuration.
        base_path: Base path for experiment outputs.
        preconditioner_name: Name of preconditioner.

    Returns:
        AttributePreservationMetrics dataclass.
    """
    base_path = Path(base_path)
    data_path = base_path / "data"

    # Load datasets
    train_ds = load_from_disk(str(data_path / "train.hf"))
    eval_ds = load_from_disk(str(data_path / "eval_majority.hf"))

    if isinstance(train_ds, DatasetDict):
        train_ds = train_ds["train"]
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]

    # Load scores
    scores = score_majority_style_eval(config, base_path, preconditioner_name)

    n_eval = len(eval_ds)
    top_k = 10

    # Extract metadata
    train_facts = train_ds["fact"]
    train_styles = train_ds["style"]
    train_occupations = train_ds["occupation"]
    train_fields = train_ds["field"]
    train_values = train_ds["value"]

    eval_facts = eval_ds["fact"]
    eval_styles = eval_ds["style"]
    eval_occupations = eval_ds["occupation"]
    eval_fields = eval_ds["field"]

    # Build occupation -> attribute pools
    occupation_employers = {
        occ: set(attrs["employers"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }
    occupation_universities = {
        occ: set(attrs["universities"]) for occ, attrs in OCCUPATION_CLUSTERS.items()
    }

    # Get top-k indices
    top_indices = np.argsort(-scores, axis=1)[:, :top_k]

    # Initialize counters
    fact_top1 = fact_top5 = fact_top10 = 0
    occ_top1 = occ_top5 = occ_top10 = 0
    style_only_top1 = 0
    style_only_top5 = 0.0
    style_only_top10 = 0.0
    same_employer_type = same_university_type = 0

    field_top1: dict[str, tuple[int, int]] = {}

    for i in range(n_eval):
        query_fact = eval_facts[i]
        query_style = eval_styles[i]
        query_occ = eval_occupations[i]
        query_field = eval_fields[i]

        top_k_idx = top_indices[i]

        # Track field accuracy
        if query_field not in field_top1:
            field_top1[query_field] = (0, 0)
        hits, total = field_top1[query_field]
        total += 1

        # Fact accuracy
        for k, idx in enumerate(top_k_idx):
            if train_facts[idx] == query_fact:
                if k == 0:
                    fact_top1 += 1
                    hits += 1
                if k < 5:
                    fact_top5 += 1
                    break
                if k < 10:
                    fact_top10 += 1
                    break

        field_top1[query_field] = (hits, total)

        # Occupation accuracy
        for k, idx in enumerate(top_k_idx):
            if train_occupations[idx] == query_occ:
                if k == 0:
                    occ_top1 += 1
                if k < 5:
                    occ_top5 += 1
                    break
                if k < 10:
                    occ_top10 += 1
                    break

        # Style-only match (style matches but occupation doesn't)
        top1_idx = top_k_idx[0]
        if (
            train_styles[top1_idx] == query_style
            and train_occupations[top1_idx] != query_occ
        ):
            style_only_top1 += 1

        style_only_top5 += (
            sum(
                1
                for idx in top_k_idx[:5]
                if train_styles[idx] == query_style
                and train_occupations[idx] != query_occ
            )
            / 5
        )
        style_only_top10 += (
            sum(
                1
                for idx in top_k_idx[:10]
                if train_styles[idx] == query_style
                and train_occupations[idx] != query_occ
            )
            / 10
        )

        # Attribute-level matching
        top1_idx = top_k_idx[0]
        top1_field = train_fields[top1_idx]
        top1_value = train_values[top1_idx]

        if top1_field == "employer" and query_field == "employer":
            if top1_value in occupation_employers.get(query_occ, set()):
                same_employer_type += 1

        if top1_field == "university" and query_field == "university":
            if top1_value in occupation_universities.get(query_occ, set()):
                same_university_type += 1

    top1_by_field = {
        field: hits / total if total > 0 else 0.0
        for field, (hits, total) in field_top1.items()
    }

    n_employer_queries = sum(1 for f in eval_fields if f == "employer")
    n_university_queries = sum(1 for f in eval_fields if f == "university")

    return AttributePreservationMetrics(
        top1_fact_accuracy=fact_top1 / n_eval,
        top5_fact_recall=fact_top5 / n_eval,
        top10_fact_recall=fact_top10 / n_eval,
        top1_occupation_accuracy=occ_top1 / n_eval,
        top5_occupation_recall=occ_top5 / n_eval,
        top10_occupation_recall=occ_top10 / n_eval,
        top1_same_employer_type=(
            same_employer_type / n_employer_queries if n_employer_queries > 0 else 0.0
        ),
        top1_same_university_type=(
            same_university_type / n_university_queries
            if n_university_queries > 0
            else 0.0
        ),
        top1_style_only_match=style_only_top1 / n_eval,
        top5_style_only_match=style_only_top5 / n_eval,
        top10_style_only_match=style_only_top10 / n_eval,
        top1_by_field=top1_by_field,
    )


def run_attribute_preservation_experiment(
    config: AttributePreservationConfig | None = None,
    base_path: Path | str = "runs/attribute_preservation",
    analysis_model: str | None = None,
    reword_model: str = "Qwen/Qwen3-8B-Base",
    include_h_eval: bool = True,
    include_majority_control: bool = True,
    include_semantic_eval: bool = True,
    include_pca: bool = True,
    pca_top_k: int = 100,
    pca_subspace_path: Path | str | None = None,
    damping_factor: float = 0.1,
) -> dict[str, AttributePreservationMetrics]:
    """Run the full attribute preservation experiment.

    Tests whether style suppression damages the ability to match on
    content attributes (occupation clusters).

    Args:
        config: Experiment configuration. Set config.hf_dataset to load data
            from HuggingFace instead of generating locally.
        base_path: Base path for outputs.
        analysis_model: Model for gradient collection. Defaults to HF_ANALYSIS_MODEL.
        reword_model: Model for style rewording (only used if not using HF dataset).
        include_h_eval: Whether to include H_eval preconditioner.
        include_majority_control: Whether to include majority style control.
        include_semantic_eval: Whether to include semantic-only eval (Q&A format).
        include_pca: Whether to include PCA style projection strategies.
        pca_top_k: Number of principal components for PCA projection.
        pca_subspace_path: Path to precomputed PCA subspace. If None, uses
            runs/asymmetric_style/pca_subspace by default.
        damping_factor: Damping factor for matrix inversion.

    Returns:
        Dictionary mapping preconditioner names to metrics.
    """
    if config is None:
        config = AttributePreservationConfig()

    base_path = Path(base_path)

    print("=" * 70)
    print("ATTRIBUTE PRESERVATION UNDER STYLE SUPPRESSION EXPERIMENT")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Style-occupation mapping:")
    for occ, style in config.style_occupation_map.items():
        print(f"    {occ}: {style}")
    print(
        f"  Eval occupation: {config.eval_occupation} "
        f"(queried in {config.eval_style} style)"
    )
    print(f"  People per occupation: {config.people_per_occupation}")

    # Step 1: Create data and index
    print("\n" + "-" * 60)
    print("STEP 1: Creating attribute-correlated dataset and index")
    print("-" * 60)
    create_styled_datasets(config, base_path / "data", reword_model)
    create_attribute_index(config, base_path, analysis_model)

    # Step 2: Compute style suppression preconditioner
    print("\n" + "-" * 60)
    print("STEP 2: Computing style suppression preconditioner (R_between)")
    print("-" * 60)
    compute_style_preconditioner_from_data(base_path, config)

    # Step 3: Evaluate minority style (style mismatch) with different preconditioners
    print("\n" + "-" * 60)
    print("STEP 3: Evaluating preconditioner strategies (minority style eval)")
    print("-" * 60)

    strategies = [
        (None, "no_precond"),
        ("r_between", "r_between"),
    ]

    all_metrics: dict[str, AttributePreservationMetrics] = {}

    for precond_name, display_name in strategies:
        print(f"\n--- Strategy: {display_name} ---")
        metrics = compute_attribute_metrics(config, base_path, precond_name)
        print_attribute_metrics(metrics, display_name)
        all_metrics[display_name] = metrics

    # Step 3b: Compute and evaluate H_eval preconditioner
    if include_h_eval:
        print("\n" + "-" * 60)
        print("STEP 3b: Computing H_eval (second moment of eval gradients)")
        print("-" * 60)
        compute_eval_second_moment(base_path, config)

        print("\n--- Strategy: h_eval ---")
        metrics = compute_attribute_metrics(config, base_path, "h_eval")
        print_attribute_metrics(metrics, "h_eval")
        all_metrics["h_eval"] = metrics

    # Step 4: Majority style control (no style mismatch)
    if include_majority_control:
        print("\n" + "-" * 60)
        print("STEP 4: Majority style control (no style mismatch)")
        print("-" * 60)
        create_majority_style_eval(config, base_path, reword_model)

        print("\n--- Control: majority_style_no_precond ---")
        metrics = compute_majority_style_metrics(config, base_path, None)
        print_attribute_metrics(metrics, "majority_no_precond")
        all_metrics["majority_no_precond"] = metrics

    # Step 5: Semantic-only eval (gradients only from answer tokens)
    if include_semantic_eval:
        print("\n" + "-" * 60)
        print("STEP 5: Semantic-only eval (gradients from answer tokens only)")
        print("-" * 60)

        # Semantic strategies to test
        semantic_strategies = [
            ("index", "semantic_index"),  # Standard IF with H_train
            (None, "semantic_no_precond"),
            ("r_between", "semantic_r_between"),
        ]

        if include_h_eval:
            semantic_strategies.append(("h_eval", "semantic_h_eval"))

        for precond_name, display_name in semantic_strategies:
            print(f"\n--- Strategy: {display_name} ---")
            metrics = compute_attribute_metrics(
                config,
                base_path,
                precond_name,
                eval_prompt_column="question",
                eval_completion_column="answer",
                damping_factor=damping_factor,
            )
            print_attribute_metrics(metrics, display_name)
            all_metrics[display_name] = metrics

    # Step 6: PCA style projection (using external style subspace)
    style_subspace = None
    if include_pca:
        from .preconditioners import compute_pca_style_subspace

        print("\n" + "-" * 60)
        print(f"STEP 6: PCA style projection (k={pca_top_k})")
        print("-" * 60)

        # Load or compute PCA subspace from asymmetric experiment
        if pca_subspace_path is None:
            pca_subspace_path = Path("runs/asymmetric_style/pca_subspace")

        pca_subspace_path = Path(pca_subspace_path)
        pirate_idx = Path("runs/precond_comparison/pirate")
        shakespeare_idx = Path("runs/precond_comparison/shakespeare")

        if pirate_idx.exists() and shakespeare_idx.exists():
            print(f"Loading PCA style subspace from {pca_subspace_path}")
            style_subspace = compute_pca_style_subspace(
                pirate_idx,
                shakespeare_idx,
                pca_subspace_path,
                top_k=pca_top_k,
            )

            # PCA without preconditioner
            print(f"\n--- Strategy: pca_k{pca_top_k} ---")
            metrics = compute_attribute_metrics_with_pca(
                config,
                base_path,
                style_subspace,
                top_k=pca_top_k,
                damping_factor=damping_factor,
            )
            print_attribute_metrics(metrics, f"pca_k{pca_top_k}")
            all_metrics[f"pca_k{pca_top_k}"] = metrics

            # PCA + H_train (index)
            print(f"\n--- Strategy: pca_k{pca_top_k}_index ---")
            metrics = compute_attribute_metrics_with_pca(
                config,
                base_path,
                style_subspace,
                top_k=pca_top_k,
                preconditioner_name="index",
                damping_factor=damping_factor,
            )
            print_attribute_metrics(metrics, f"pca_k{pca_top_k}_index")
            all_metrics[f"pca_k{pca_top_k}_index"] = metrics

            # Semantic + PCA
            if include_semantic_eval:
                print(f"\n--- Strategy: semantic_pca_k{pca_top_k} ---")
                metrics = compute_attribute_metrics_with_pca(
                    config,
                    base_path,
                    style_subspace,
                    top_k=pca_top_k,
                    eval_prompt_column="question",
                    eval_completion_column="answer",
                    damping_factor=damping_factor,
                )
                print_attribute_metrics(metrics, f"semantic_pca_k{pca_top_k}")
                all_metrics[f"semantic_pca_k{pca_top_k}"] = metrics

                # Semantic + PCA + H_train
                print(f"\n--- Strategy: semantic_pca_k{pca_top_k}_index ---")
                metrics = compute_attribute_metrics_with_pca(
                    config,
                    base_path,
                    style_subspace,
                    top_k=pca_top_k,
                    preconditioner_name="index",
                    eval_prompt_column="question",
                    eval_completion_column="answer",
                    damping_factor=damping_factor,
                )
                print_attribute_metrics(metrics, f"semantic_pca_k{pca_top_k}_index")
                all_metrics[f"semantic_pca_k{pca_top_k}_index"] = metrics

            # k=500: test full subspace removal (capped at dim=256)
            pca_k500 = 500
            style_subspace_k500 = compute_pca_style_subspace(
                pirate_idx,
                shakespeare_idx,
                pca_subspace_path,
                top_k=pca_k500,
            )

            print(f"\n--- Strategy: pca_k{pca_k500} ---")
            metrics = compute_attribute_metrics_with_pca(
                config,
                base_path,
                style_subspace_k500,
                top_k=pca_k500,
                damping_factor=damping_factor,
            )
            print_attribute_metrics(metrics, f"pca_k{pca_k500}")
            all_metrics[f"pca_k{pca_k500}"] = metrics

            print(f"\n--- Strategy: pca_k{pca_k500}_index ---")
            metrics = compute_attribute_metrics_with_pca(
                config,
                base_path,
                style_subspace_k500,
                top_k=pca_k500,
                preconditioner_name="index",
                damping_factor=damping_factor,
            )
            print_attribute_metrics(metrics, f"pca_k{pca_k500}_index")
            all_metrics[f"pca_k{pca_k500}_index"] = metrics

            if include_semantic_eval:
                print(f"\n--- Strategy: semantic_pca_k{pca_k500} ---")
                metrics = compute_attribute_metrics_with_pca(
                    config,
                    base_path,
                    style_subspace_k500,
                    top_k=pca_k500,
                    eval_prompt_column="question",
                    eval_completion_column="answer",
                    damping_factor=damping_factor,
                )
                print_attribute_metrics(metrics, f"semantic_pca_k{pca_k500}")
                all_metrics[f"semantic_pca_k{pca_k500}"] = metrics

                print(f"\n--- Strategy: semantic_pca_k{pca_k500}_index ---")
                metrics = compute_attribute_metrics_with_pca(
                    config,
                    base_path,
                    style_subspace_k500,
                    top_k=pca_k500,
                    preconditioner_name="index",
                    eval_prompt_column="question",
                    eval_completion_column="answer",
                    damping_factor=damping_factor,
                )
                print_attribute_metrics(metrics, f"semantic_pca_k{pca_k500}_index")
                all_metrics[f"semantic_pca_k{pca_k500}_index"] = metrics
        else:
            print("WARNING: PCA indices not found at runs/precond_comparison/")
            print(
                "  Run the asymmetric style experiment first to generate PCA subspace."
            )

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Style Suppression vs Attribute Preservation Trade-off")
    print("=" * 70)

    print(
        f"\n{'Strategy':<25} {'Fact Acc':<12} {'Occ Acc':<12} "
        f"{'Style Only':<12} {'Trade-off':<12}"
    )
    print("-" * 73)

    for name, m in all_metrics.items():
        # Trade-off: we want high occupation accuracy and low style-only matches
        # A good trade-off is when occ_acc is high and style_only is low
        trade_off = m.top1_occupation_accuracy - m.top1_style_only_match
        print(
            f"{name:<25} {m.top1_fact_accuracy:<12.2%} "
            f"{m.top1_occupation_accuracy:<12.2%} "
            f"{m.top1_style_only_match:<12.2%} {trade_off:<12.2%}"
        )

    print("\nInterpretation:")
    print("  - Fact Accuracy: How well we match exact facts (semantic matching)")
    print(
        "  - Occupation Accuracy: How well we match occupation cluster "
        "(attribute preservation)"
    )
    print(
        "  - Style Only: Matches where style matches but occupation doesn't "
        "(should be LOW)"
    )
    print("  - Trade-off: Occ Acc - Style Only (higher is better)")
    print(
        "  - majority_no_precond: Control showing baseline when eval "
        "style matches training"
    )
    if include_semantic_eval:
        print("  - semantic_*: Eval gradients only from answer tokens (Q&A format)")

    baseline = all_metrics.get("no_precond")
    r_between = all_metrics.get("r_between")
    h_eval = all_metrics.get("h_eval")
    majority = all_metrics.get("majority_no_precond")

    print("\n" + "-" * 60)
    print("KEY FINDINGS")
    print("-" * 60)

    if baseline and r_between:
        # Check if R_between reduced style-only matches
        style_reduction = (
            baseline.top1_style_only_match - r_between.top1_style_only_match
        )
        print(f"\nR_between Style-Only Match Reduction: {style_reduction:.2%}")

        # Check if attribute preservation was damaged
        occ_change = (
            r_between.top1_occupation_accuracy - baseline.top1_occupation_accuracy
        )
        print(f"R_between Occupation Accuracy Change: {occ_change:+.2%}")

    if h_eval and baseline:
        style_reduction_h = (
            baseline.top1_style_only_match - h_eval.top1_style_only_match
        )
        occ_change_h = (
            h_eval.top1_occupation_accuracy - baseline.top1_occupation_accuracy
        )
        print(f"\nH_eval Style-Only Match Reduction: {style_reduction_h:.2%}")
        print(f"H_eval Occupation Accuracy Change: {occ_change_h:+.2%}")

    if majority:
        print("\nMajority Style Control (upper bound):")
        print(f"  Fact Accuracy: {majority.top1_fact_accuracy:.2%}")
        print(f"  Occupation Accuracy: {majority.top1_occupation_accuracy:.2%}")

    if baseline and r_between:
        style_reduction = (
            baseline.top1_style_only_match - r_between.top1_style_only_match
        )
        occ_change = (
            r_between.top1_occupation_accuracy - baseline.top1_occupation_accuracy
        )
        if style_reduction > 0 and occ_change >= -0.05:
            print(
                "\n✓ SUCCESS: Style suppression works without "
                "damaging attribute preservation!"
            )
        elif style_reduction > 0 and occ_change < -0.05:
            print("\n⚠ PARTIAL: Style suppressed but attribute preservation damaged")
        elif style_reduction <= 0:
            print("\n✗ FAILURE: Style suppression not effective")

    return all_metrics


if __name__ == "__main__":
    run_attribute_preservation_experiment()
