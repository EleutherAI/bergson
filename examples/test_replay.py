import shutil
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.training_args import TrainingArguments

from bergson.config import DataConfig, IndexConfig, ReduceConfig
from bergson.data import load_gradients
from bergson.replay.replay_callbacks import SaveEverySqrtStepsCallback
from bergson.replay.replay_trainer import ReplayTrainer
from bergson.utils import assert_type
from bergson.worker_utils import setup_data_pipeline, setup_model_and_peft

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
finetuned_model_name = "EleutherAI/test-SmolLM2-135M-Instruct"
checkpoint_path = Path("runs/test_replay_ckpts_5")
reduce_path = Path("runs/test_replay_reduce")
ds_name = "NeelNanda/pile-10k"
ds_split = "train"
ds_N = 1000

training_args = TrainingArguments(
    output_dir=checkpoint_path,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    bf16=True,  # Use bf16 instead of fp16 to avoid gradient scaler issues
    report_to="none",
)

index_cfg = IndexConfig(
    run_path=str(checkpoint_path),
    model=model_name,
    tokenizer=model_name,
    data=DataConfig(
        dataset=ds_name,
        split=ds_split,
        subset=None,
        truncation=True,
    ),
    token_batch_size=2048,  # Smaller token batch size to avoid OOM
    fsdp=False,
    projection_dim=0,
    skip_preconditioners=True,
)

ds = setup_data_pipeline(index_cfg)
ds = assert_type(Dataset, ds)
ds = ds.select(range(ds_N))
num_items = len(ds)
train_ds = ds.select(range(int(num_items * 0.9)))
eval_ds = ds.select(range(int(num_items * 0.9), num_items))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def finetune(checkpoint_path: Path):
    model, target_modules = setup_model_and_peft(
        index_cfg,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        freeze=False,
    )

    trainer = ReplayTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[SaveEverySqrtStepsCallback()],
    )
    trainer.train()

    # Push to hub
    model.push_to_hub(finetuned_model_name)
    AutoTokenizer.from_pretrained(model_name).push_to_hub(finetuned_model_name)


def reduce(reduce_path: Path):
    query_index_cfg = IndexConfig(
        run_path=str(reduce_path),
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

    reduce_part_path = Path(query_index_cfg.partial_run_path)
    if reduce_path.exists():
        shutil.rmtree(reduce_path)
    if reduce_part_path.exists():
        shutil.rmtree(reduce_part_path)

    # Load the final model checkpoint and reduce a query from the eval set cais/wmdp
    cmd = [
        "python",
        "-m",
        "bergson",
        "reduce",
        query_index_cfg.run_path,
        "--model",
        finetuned_model_name,
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
        "--skip_preconditioners",
        "--projection_dim",
        "0",
    ]
    print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd, cwd=Path(__file__).parent, capture_output=True, text=True
        )
        print("Reduce command output:")
        print(result.stdout)
        if result.returncode != 0:
            print(result.stdout)
            print("Reduce command errors:")
            print(result.stderr)
            print(f"\nReduce command failed with return code {result.returncode}")
            print(
                "This is a known issue with distributed file creation in the reduce command."
            )
        else:
            print("Reduce command completed successfully!")
    except Exception as e:
        print(f"Error running reduce command: {e}")
        print("Training completed successfully, but reduce step failed.")


def test_attribute(reduce_path: Path, checkpoint_path: Path):
    from copy import deepcopy

    finetuned_index_cfg = deepcopy(index_cfg)
    finetuned_index_cfg.model = finetuned_model_name

    model, target_modules = setup_model_and_peft(
        finetuned_index_cfg,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        freeze=False,
    )

    trainer = ReplayTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    target_modules = list(load_gradients(reduce_path, structured=True).dtype.names)

    query = torch.tensor(load_gradients(reduce_path, structured=False))

    num_training_items = len(train_ds)

    training_jacobian = trainer.attribute(
        query, checkpoint_path, target_modules, num_training_items
    )

    assert training_jacobian is not None


def main():
    # finetune(checkpoint_path)
    # reduce(reduce_path)
    test_attribute(reduce_path, checkpoint_path)


if __name__ == "__main__":
    main()
