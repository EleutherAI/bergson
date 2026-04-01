import os
from typing import List, Literal, Optional, Union

import backoff
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    dataset: str = Field(..., description="Dataset")
    split: str = Field(..., description="Split")

    prompt_column: str = Field("prompt", description="Prompt column")
    completion_column: str = Field("completion", description="Completion column")

    # Training type configuration
    loss: Literal["dpo", "orpo", "sft"] = Field(
        ..., description="Loss function / training type"
    )

    # Output model
    finetuned_model_id: Optional[str] = Field(
        None, description="File ID of the finetuned model"
    )

    # Model configuration
    max_seq_length: int = Field(
        2048, description="Maximum sequence length for training"
    )
    load_in_4bit: bool = Field(
        False, description="Whether to load model in 4-bit quantization"
    )

    # PEFT configuration
    is_peft: bool = Field(True, description="Whether to use PEFT for training")
    target_modules: Optional[List[str]] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA",
    )
    lora_bias: Literal["all", "none"] = Field(
        "none", description="Value for FastLanguageModel.get_peft_model(bias=?)"
    )

    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    merge_before_push: bool = Field(
        True,
        # description="Whether to merge model before pushing to Hub. Only merged models
        # can be used as parent models for further finetunes. Only supported for
        # bf16 models.",
    )
    push_to_private: bool = Field(True, description="Whether to push to private Hub")

    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: int = Field(-1, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(
        2, description="Training batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        8, description="Number of gradient accumulation steps"
    )
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(
        1e-4, description="Learning rate or string expression"
    )
    logging_steps: int = Field(1, description="Number of steps between logging")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field(
        "./tmp", description="Output directory for training checkpoints"
    )

    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(
                f"You have set org={org}, but it must be an org you have access to"
            )
        return v

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed_schedulers = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v


# def process(df, prompt_column: str = "prompt", completion_column: str = "completion"):
#     def format_chat_data(example):
#         old_example = example
#         example["prompt"] = [{"role": "user", "content": old_example[prompt_column]}]
#         example["completion"] = [
#             {"role": "assistant", "content": old_example[completion_column]}
#         ]
#         return example

#     df = df.map(format_chat_data)
#     return df


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset):  # type: ignore[override]
        sampler = SequentialSampler(train_dataset)  # type: ignore[arg-type]

        return sampler


def train(training_cfg: TrainingConfig, dataset: Dataset):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 3. Define LoRA config
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )

    # dataset = process(
    #     dataset,
    #     prompt_column=training_cfg.prompt_column,
    #     completion_column=training_cfg.completion_column,
    # )
    if training_cfg.seed is not None:
        dataset = dataset.shuffle(seed=training_cfg.seed)

    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            completion_only_loss=True,
            ddp_find_unused_parameters=False,
            fp16=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=float(training_cfg.learning_rate),
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            num_train_epochs=training_cfg.epochs,
            label_names=["labels"],
            optim=training_cfg.optim,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            report_to=None,
            save_steps=training_cfg.save_steps,
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
        ),
        peft_config=peft_config,
        callbacks=[],
    )
    trainer.train()

    if rank == 0:
        if training_cfg.finetuned_model_id is not None:
            push_model(training_cfg, training_cfg.finetuned_model_id, model, tokenizer)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # model_name = "Qwen/Qwen2.5-7B"
    parser.add_argument("--finetuned_model_path", type=str, default="finetuned-model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--completion_column", type=str, default="completion")
    parser.add_argument(
        "--no_push_to_private", action="store_false", dest="push_to_private"
    )

    args = parser.parse_args()

    training_config = TrainingConfig(  # type: ignore
        finetuned_model_id=args.finetuned_model_path,  # type: ignore
        model=args.model_name,  # type: ignore
        dataset=args.dataset_name,  # type: ignore
        split=args.split,  # type: ignore
        loss="sft",  # type: ignore
        prompt_column=args.prompt_column,  # type: ignore
        completion_column=args.completion_column,  # type: ignore
        merge_before_push=False,
        push_to_private=args.push_to_private,
    )  # type: ignore

    dataset, _ = setup_data_pipeline(
        IndexConfig(
            run_path=f"runs/{args.finetuned_model_path}",
            model=args.model_name,
            data=DataConfig(
                dataset=args.dataset_name,
                split=args.split,
                prompt_column=args.prompt_column,
                completion_column=args.completion_column,
            ),
        )
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")
    train(training_config, dataset)


if __name__ == "__main__":
    main()
