from pathlib import Path
import subprocess

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling

from bergson.replay.replay_trainer import ReplayTrainer
from bergson.replay.sqrt_callback import SaveEverySqrtStepsCallback
from bergson.utils import assert_type
from bergson.config import IndexConfig, ReduceConfig, DataConfig
from bergson.worker_utils import setup_data_pipeline, setup_model_and_peft

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
finetuned_model_name = "EleutherAI/test-SmolLM2-135M-Instruct"
output_path = Path("runs/test_replay_ckpts")
ds_name = "NeelNanda/pile-10k"
ds_split = "train"
ds_N = 1000

# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id

model, target_modules = setup_model_and_peft(
    IndexConfig(
        run_path=str(output_path),
        model=model_name,
        tokenizer=model_name,
        data=DataConfig(
            dataset=ds_name,
            split=ds_split,
            subset=None,
        ),
    ),
    rank=dist.get_rank() if dist.is_initialized() else 0,
)

ds = setup_data_pipeline(
    IndexConfig(
        run_path=str(output_path),
        model=model_name,
        tokenizer=model_name,
        data=DataConfig(
            dataset=ds_name,
            split=ds_split,
            subset=None,
        ),
    )
)
ds = assert_type(Dataset, ds)
ds = ds.select(range(ds_N))
num_items = len(ds)


# Add labels to the dataset
ds = ds.add_column("labels", ds["input_ids"])
print("input_ids type:", type(ds[0]["input_ids"]))
print("input_ids[0] element type:", type(ds[0]["input_ids"][0]))
print("labels type:", type(ds[0]["labels"]))
print("labels[0] element type:", type(ds[0]["labels"][0]))
exit()
print(ds.column_names)
print(ds[0])

train_ds = ds.select(range(int(num_items * 0.9)))
eval_ds = ds.select(range(int(num_items * 0.9), num_items))

training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    fp16=True,
)
trainer = ReplayTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    # processing_class=tokenizer,
    callbacks=[SaveEverySqrtStepsCallback()],
)

trainer.train()

# Get the last checkpoint
last_checkpoint_path = list(output_path.glob("checkpoint-*"))[-1]

# Load and push to hub
model.push_to_hub(finetuned_model_name)
AutoTokenizer.from_pretrained(model_name).push_to_hub(finetuned_model_name)

query_index_cfg = IndexConfig(
    run_path="runs/test_replay_reduce",
    model=finetuned_model_name,
    tokenizer=model_name,
    data=DataConfig(
        dataset="cais/wmdp",
        split="test",
        subset="wmdp-bio",
        prompt_column="question",
        completion_column="choices",
    ),
)

# Load the final model checkpoint and reduce a query from the eval set cais/wmdp

cmd = [
    "python",
    "-m",
    "bergson",
    "reduce",
    query_index_cfg.run_path,
    "--model",
    last_checkpoint_path,
    "--tokenizer",
    query_index_cfg.tokenizer,
    "--dataset",
    query_index_cfg.data.dataset,
    "--split",
    query_index_cfg.data.split,
    "--subset",
    query_index_cfg.data.subset,
    "--prompt_column",
    query_index_cfg.data.prompt_column,
    "--completion_column",
    query_index_cfg.data.completion_column,
    "--method",
    ReduceConfig.method,
    "--unit_normalize",
]

result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
