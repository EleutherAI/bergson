from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments

from bergson.replay.replay_trainer import ReplayTrainer
from bergson.utils import assert_type

model_name = "EleutherAI/pythia-14m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("NeelNanda/pile-10k", split="train")
dataset = assert_type(Dataset, dataset)
num_items = len(dataset)


def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result


train_dataset = dataset.select(range(int(num_items * 0.9)))
eval_dataset = dataset.select(range(int(num_items * 0.9), num_items))

train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
eval_dataset = eval_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="test_replay",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
)
trainer = ReplayTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
