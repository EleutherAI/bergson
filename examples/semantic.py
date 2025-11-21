from pathlib import Path
import subprocess

import torch
from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer


def reword(dataset, model_name: str, prompt_template: str, batch_size: int = 8):
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
        prompts = [prompt_template.format(fact=item["fact"]) for item in batch_items]

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
            new_facts.append(item["fact"])
            new_reworded.append(output_text.strip())

    # Reconstruct dataset
    return Dataset.from_dict({"fact": new_facts, "reworded": new_reworded})


def create_data():
    dataset = load_from_disk("data/facts_dataset.hf")

    for model_name in ["Qwen/Qwen3-8B-Base", "Meta-Llama/Meta-Llama-3-8B"]:

        # 1. Shakespeare
        prompt_shake = (
            "Reword the following fact in a Shakespearean style, adding flair and poetry.\n"
            "Do not include other text in your response, just the contents of the reworded fact.\n"
            "Fact: {fact}\n"
            "Your rewrite:"
        )

        ds_shake = reword(dataset, model_name, prompt_shake, batch_size=8)
        ds_shake.save_to_disk(
            f"data/facts_dataset_shakespeare-{model_name.split('/')[-1]}.hf"
        )
        print("Shakespearean processing done.")

        # 2. Pirate
        prompt_pirate = (
            "Reword the following fact like it's coming from a pirate. Be creative!\n"
            "Do not include any other text in your response, just the contents of the reworded fact.\n"
            "Fact: {fact}\n"
            "Your rewrite:"
        )

        ds_pirate = reword(dataset, model_name, prompt_pirate, batch_size=8)
        ds_pirate.save_to_disk(
            f"data/facts_dataset_pirate-{model_name.split('/')[-1]}.hf"
        )
        print("Pirate processing done.")


def create_index(dataset_name, analysis_model_name):
    run_path = Path(f"runs/{dataset_name}")
    cmd = [
        "bergson",
        "build",
        str(run_path),
        "--model",
        analysis_model_name,
        "--dataset",
        dataset_name,
        "--drop_columns",
        "False",
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
        "--fsdp",
        "--projection_dim",
        "128",
        "--skip_preconditioners",
    ]

    print(" ".join(cmd))
    if not run_path.exists():
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)


def finetune(dataset_path, analysis_model_name, finetuned_model_path):
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29500",
        "--standalone",
        "examples/train_lora.py",
        # "examples/finetune_sem.py",
        "--dataset_name",
        dataset_path,
        "--finetuned_model_path",
        finetuned_model_path,
        "--model_name",
        analysis_model_name,
        "--prompt_column",
        "fact",
        "--completion_column",
        "reworded",
    ]
    print(" ".join(cmd))
    with subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,  # "Pipe" the output to us
        stderr=subprocess.STDOUT, # Merge errors into the standard output stream
        text=True,                # Decode bytes to string automatically
        bufsize=1                 # Line buffering (updates every line)
    ) as process:
        # Iterate over the output line by line as it comes in
        for line in process.stdout: # type: ignore
            print(line.strip()) 
            
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def main():
    # create_data()
    dataset_paths = [
        "data/facts_dataset_shakespeare-Qwen3-8B-Base.hf",
        "data/facts_dataset_pirate-Qwen3-8B-Base.hf",
        "data/facts_dataset_shakespeare-Meta-Llama-3-8B.hf",
        "data/facts_dataset_pirate-Meta-Llama-3-8B.hf",
    ]

    original = load_from_disk("data/facts_dataset.hf")

    merged_datasets = []

    for path in dataset_paths:
        ds = load_from_disk(path)

        # Add back any dropped columns from original
        for col in original.column_names:
            if col not in ds.column_names:
                # Align ds length with original by matching on "fact"
                # Create a mapping from fact → row
                orig_map = {row["fact"]: row for row in original}

                # Build list for restored column
                restored_col = [orig_map[row["fact"]][col] for row in ds]

                ds = ds.add_column(col, restored_col)

        merged_datasets.append(ds)

    final_dataset = concatenate_datasets(merged_datasets)
    final_dataset = final_dataset.shuffle(seed=42)

    final_dataset_path = "data/facts_dataset_reworded.hf"
    final_dataset.save_to_disk(final_dataset_path)
    print(f"Merged dataset saved to: {final_dataset_path}")

    analysis_model_name = "Qwen/Qwen3-4B"

    finetuned_model_path = f"finetuned-{final_dataset_path.split('/')[-1].split('.')[0]}-{analysis_model_name}"
    # Finetune model on dataset
    finetune(final_dataset_path, analysis_model_name, finetuned_model_path)

    # Build index with finetuned model
    create_index(final_dataset_path, finetuned_model_path)


if __name__ == "__main__":
    main()
