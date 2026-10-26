"""Dataset creation and rewording utilities for semantic experiments."""

from pathlib import Path
from typing import Any, cast

import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default HuggingFace repos for experiments
HF_ASYMMETRIC_STYLE = "EleutherAI/bergson-asymmetric-style"
HF_ATTRIBUTE_PRESERVATION = "EleutherAI/bergson-attribute-preservation"
HF_ANALYSIS_MODEL = "EleutherAI/bergson-asymmetric-style-qwen3-8b-lora"


def load_experiment_data(
    base_path: Path | str | None = None,
    hf_repo: str | None = None,
    splits: list[str] | None = None,
) -> DatasetDict:
    """Load experiment data from HuggingFace or local disk.

    Args:
        base_path: Local path containing data/*.hf directories.
            Required if hf_repo is None.
        hf_repo: HuggingFace dataset repo ID
            (e.g., "EleutherAI/bergson-asymmetric-style").
            If provided, downloads from HF and ignores base_path.
        splits: Optional list of splits to load. If None, loads all available splits.

    Returns:
        DatasetDict with the requested splits.

    Examples:
        # Load from HuggingFace
        data = load_experiment_data(hf_repo="EleutherAI/bergson-asymmetric-style")

        # Load from local disk
        data = load_experiment_data(base_path="runs/asymmetric_style")

        # Load specific splits
        data = load_experiment_data(hf_repo="...", splits=["train", "eval"])
    """
    if hf_repo:
        loaded = load_dataset(hf_repo)
        if not isinstance(loaded, DatasetDict):
            raise TypeError(f"Expected DatasetDict from HF, got {type(loaded)}")
        dataset_dict: DatasetDict = loaded
        if splits:
            filtered = {k: dataset_dict[k] for k in splits if k in dataset_dict}
            dataset_dict = DatasetDict(cast(Any, filtered))
        return dataset_dict

    if base_path is None:
        raise ValueError("Either base_path or hf_repo must be provided")

    base_path = Path(base_path)
    data_path = base_path / "data"

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Discover available splits
    available_splits = [p.stem for p in data_path.glob("*.hf") if p.is_dir()]

    if splits:
        available_splits = [s for s in splits if s in available_splits]

    if not available_splits:
        raise FileNotFoundError(f"No .hf datasets found in {data_path}")

    result: dict[str, Dataset] = {}
    for split in available_splits:
        ds = load_from_disk(str(data_path / f"{split}.hf"))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        result[split] = ds
    return DatasetDict(cast(Any, result))


def reword(
    dataset: Dataset, model_name: str, prompt_template: str, batch_size: int = 8
) -> Dataset:
    """Reword facts in a dataset using a language model.

    Args:
        dataset: Dataset containing a "fact" column.
        model_name: HuggingFace model name to use for rewording.
        prompt_template: Template string with {fact} placeholder.
        batch_size: Batch size for generation.

    Returns:
        Dataset with "fact" and "reworded" columns.
    """
    device = "cuda:3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # REQUIRED for batched generation with Llama/Qwen/Mistral
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

    # Convert dataset to list for easy slicing
    # (Assuming the dataset is small enough to fit in RAM, which 1000 items is)
    data_list = list(dataset)

    print(f"Starting generation with batch size: {batch_size}...")

    for i in tqdm(range(0, len(data_list), batch_size)):
        # 1. Prepare the batch
        batch_items = data_list[i : i + batch_size]
        prompts = [prompt_template.format(fact=item["fact"]) for item in batch_items]  # type: ignore[index]

        # 2. Tokenize (Batch mode)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        # 3. Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                min_p=0.0,
            )

        # 4. Slice output to remove prompt (all at once)
        # With left-padding, the prompt is always the first 'input_len' tokens
        generated_tokens = outputs[:, input_len:]

        # 5. Decode batch
        decoded_batch = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # 6. Store results
        for item, output_text in zip(batch_items, decoded_batch):
            new_facts.append(item["fact"])  # type: ignore[index]
            new_reworded.append(output_text.strip())

    # Reconstruct dataset
    return Dataset.from_dict({"fact": new_facts, "reworded": new_reworded})


def create_data() -> None:
    """Create reworded datasets in Shakespeare and Pirate styles."""
    dataset = load_from_disk("data/facts_dataset.hf")
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    for model_name in ["Qwen/Qwen3-8B-Base", "Meta-Llama/Meta-Llama-3-8B"]:
        model_short = model_name.split("/")[-1]

        # 1. Shakespeare
        shake_path = f"data/facts_dataset_shakespeare-{model_short}.hf"
        if not Path(shake_path).exists():
            prompt_shake = (
                "Reword the following fact in a Shakespearean style, adding flair and "
                "poetry.\n"
                "Do not include other text in your response, just the contents of the "
                "reworded fact.\n"
                "Fact: {fact}\n"
                "Your rewrite:"
            )

            ds_shake = reword(dataset, model_name, prompt_shake, batch_size=8)
            ds_shake.save_to_disk(shake_path)
            print("Shakespearean processing done.")

        # 2. Pirate
        pirate_path = f"data/facts_dataset_pirate-{model_short}.hf"
        if not Path(pirate_path).exists():
            prompt_pirate = (
                "Reword the following fact like it's coming from a pirate. "
                "Be creative!\n"
                "Do not include any other text in your response, "
                "just the contents of the "
                "reworded fact.\n"
                "Fact: {fact}\n"
                "Your rewrite:"
            )

            ds_pirate = reword(dataset, model_name, prompt_pirate, batch_size=8)
            ds_pirate.save_to_disk(pirate_path)
            print("Pirate processing done.")


def create_qwen_only_dataset() -> Path:
    """Create a merged dataset with only Qwen-generated styles (pirate + shakespeare).

    Returns:
        Path to the created dataset.
    """
    qwen_dataset_path = Path("data/facts_dataset_reworded_qwen.hf")

    if qwen_dataset_path.exists():
        print(f"Qwen-only dataset already exists at {qwen_dataset_path}")
        return qwen_dataset_path

    print("Creating Qwen-only merged dataset...")
    original = load_from_disk("data/facts_dataset.hf")
    if isinstance(original, DatasetDict):
        original = original["train"]

    qwen_paths = [
        "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf",
        "data/facts_dataset_pirate-Qwen3-8B-Base.hf",
    ]

    merged_datasets = []
    for path in qwen_paths:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]

        # Add back any dropped columns from original
        for col in original.column_names:
            if col not in ds.column_names:
                orig_map = {row["fact"]: row for row in original}  # type: ignore[index]
                restored_col = [orig_map[row["fact"]][col] for row in ds]  # type: ignore[index]
                ds = ds.add_column(col, restored_col)

        merged_datasets.append(ds)

    final_dataset = concatenate_datasets(merged_datasets)
    final_dataset = final_dataset.shuffle(seed=42)
    final_dataset.save_to_disk(str(qwen_dataset_path))
    print(f"Qwen-only dataset saved to: {qwen_dataset_path}")

    return qwen_dataset_path
