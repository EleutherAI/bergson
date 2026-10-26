"""
Setup utilities for the induction head experiment.

Contains model creation, data loading, and induction head dataset generation.
"""

import random

from datasets import Dataset, load_dataset

from bergson import AttentionConfig
from bergson.utils import assert_type
from examples.experiments.find_induction_heads_experiment.attn_only_transformer import (  # noqa: F401
    AttnOnlyConfig,
    AttnOnlyForCausalLM,
)

HEAD_CFGS = {
    "h.0.attn.c_attn": AttentionConfig(12, 192, 2),
    "h.0.attn.c_proj": AttentionConfig(12, 64, 2),
    "h.1.attn.c_attn": AttentionConfig(12, 192, 2),
    "h.1.attn.c_proj": AttentionConfig(12, 64, 2),
}


def check_logins():
    """Check if user is logged into HF hub and wandb."""
    print("Checking authentication...")

    # Check HF hub login
    try:
        from huggingface_hub import whoami

        whoami()
        print("✓ Logged into Hugging Face Hub")
    except Exception as e:
        print("✗ Not logged into Hugging Face Hub. Please run: huggingface-cli login")
        raise e

    # Check wandb login
    try:
        import wandb

        wandb.login()
        print("✓ Logged into Weights & Biases")
    except Exception as e:
        print("✗ Not logged into Weights & Biases. Please run: wandb login")
        raise e


def create_model(tokenizer, special_pos_embed):
    """Create an attention-only transformer."""
    cfg = AttnOnlyConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        max_position_embeddings=1024,
        layer_norm=False,
        special_pos_embed=special_pos_embed,
    )
    model = AttnOnlyForCausalLM(cfg)

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model


def load_data(
    tokenizer, N: int | None = None, name="EleutherAI/SmolLM2-135M-10B", max_length=512
):
    """Load and preprocess dataset."""
    split = f"train[:{N}]" if N is not None else "train"
    dataset = load_dataset(name, split=split)
    dataset = assert_type(Dataset, dataset)

    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )

        # For language modeling, labels are the same as input_ids
        # TODO probably remove this
        # tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Split into train/eval
    train_eval = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_eval["train"]
    eval_dataset = train_eval["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def build_single_token_vocab(tokenizer, wordlist, max_words=500):
    singles = []
    for w in wordlist:
        toks = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            singles.append(w)
        if len(singles) >= max_words:
            break
    return singles


def create_induction_ds(tokenizer, seed, num_prompts=100):
    random.seed(seed)

    # Separate words into appropriate A and B categories for sensible bigrams
    A_words = [
        "blue",
        "green",
        "red",
        "gold",
        "happy",
        "sad",
        "big",
        "small",
        "fast",
        "slow",
        "smart",
        "kind",
        "brave",
        "wise",
        "young",
        "old",
    ]
    B_words = [
        "cat",
        "dog",
        "bird",
        "wolf",
        "bear",
        "sun",
        "moon",
        "star",
        "book",
        "tree",
        "car",
        "road",
        "sky",
        "song",
        "king",
        "queen",
        "child",
        "story",
        "house",
        "river",
        "mountain",
        "flower",
        "cloud",
    ]

    A_vocab = build_single_token_vocab(tokenizer, A_words)
    B_vocab = build_single_token_vocab(tokenizer, B_words)
    print(f"A vocab size: {len(A_vocab)}")
    print(f"B vocab size: {len(B_vocab)}")

    # Verify that all words are indeed single tokens
    print("A vocab:", A_vocab)
    print("B vocab:", B_vocab)

    patterns = [
        "The {A} {B} was happy. The {A} {B}",
        "Once the {A} {B} played, later the {A} {B}",
        "In the story the {A} {B} ran fast. The {A} {B}",
        "My favorite is the {A} {B} that sings. The {A} {B}",
        "Everyone said the {A} {B} is smart. The {A} {B}",
    ]

    dataset = []
    for _ in range(num_prompts):
        try:
            A = random.choice(A_vocab)
            B = random.choice(B_vocab)
        except ValueError:
            print(f"A vocab size: {len(A_vocab)}, B vocab size: {len(B_vocab)}")
            raise ValueError("Not enough unique tokens in vocab")

        template = random.choice(patterns)
        text = template.format(A=A, B=B)
        toks = tokenizer(
            text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        input_ids = toks["input_ids"]
        labels = [-100] * len(input_ids)

        # Set the last non-padding token as the target
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != tokenizer.pad_token_id:
                labels[i] = input_ids[i]
                break

        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": toks["attention_mask"],
                "labels": labels,
                "text": text,
            }
        )
    return Dataset.from_list(dataset)


def test_induction_head_labels(tokenizer):
    dataset = create_induction_ds(tokenizer, seed=0, num_prompts=3)

    for ex in dataset:
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        A_id = tokenizer(ex["A"], add_special_tokens=False)["input_ids"][0]
        B_id = tokenizer(ex["B"], add_special_tokens=False)["input_ids"][0]

        # check only {A, B, -100} appear
        allowed = {A_id, B_id, -100}
        assert set(labels.tolist()).issubset(allowed)

        # every A in input_ids must be in labels
        for pos in (input_ids == A_id).nonzero(as_tuple=True)[0]:
            assert labels[pos] == A_id

        # every B in input_ids must be in labels
        for pos in (input_ids == B_id).nonzero(as_tuple=True)[0]:
            assert labels[pos] == B_id

        # final token must be B
        assert labels[-1].item() == B_id


def upload_to_hub(model, tokenizer, model_name="2layer-transformer-tinystories"):
    """Upload the trained model to Hugging Face Hub."""
    print(f"Uploading model to Hugging Face Hub as {model_name}...")

    try:
        # Push model and tokenizer
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
        print(f"✓ Successfully uploaded to https://huggingface.co/{model_name}")
    except Exception as e:
        print(f"✗ Failed to upload to HF Hub: {e}")
        raise e
