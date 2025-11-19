# python -m data.generate_facts --num_facts 1000

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def reword(dataset, model_name: str, prompt_template: str):
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    def generate(text: str):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                min_p=0.0,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt prefix.
        return generated[len(text) :].strip()

    # Process each item in dataset
    new_items = []
    for item in dataset:
        fact = item["fact"]

        prompt = prompt_template.format(fact=fact)
        output = generate(prompt)

        item["reworded"] = output
        print(output)

        new_items.append(item)

    return Dataset.from_list(new_items)


def main():
    dataset = load_from_disk("data/facts_dataset.hf")
    # model_name = "Meta-Llama/Meta-Llama-3-8B-Instruct"
    model_name = "Qwen/Qwen3-8B-Base"

    prompt = (
        "Reword the following fact in a Shakespearean style, adding "
        "flair and poetry.\n "
        "Do not include other text in your response, "
        "just the contents of the reworded fact.\n "
        "Fact: {fact}\n "
        "Your rewrite: (remember, no notes or explanations):"
    )

    reword(dataset, model_name, prompt).to_disk("data/facts_dataset_shakespeare.hf")

    prompt = (
        "Reword the following fact like it's coming from a pirate. Be creative!\n "
        "Do not include any other text in your response, "
        "just the contents of the reworded fact.\n "
        "Fact: {fact}\n "
        "Your rewrite: (remember, no notes or explanations):"
    )

    reword(dataset, model_name, prompt).to_disk("data/facts_dataset_pirate.hf")


if __name__ == "__main__":
    main()
