import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from simple_parsing import parse
from torch import Tensor
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from trl import SFTConfig, SFTTrainer, setup_chat_format

from bergson.collection import GradientCollector, GradientProcessor
from bergson.config import DataConfig
from bergson.data import load_gradient_dataset, tokenize, unflatten
from bergson.utils import assert_type


@dataclass
class FilterConfig:
    """Config for building the index and running the model/dataset pipeline."""

    filter: Literal["classification", "attribution", "loss", "random"] = "attribution"
    """Filter to apply to the training set before finetuning."""

    model: str = "HuggingFaceTB/SmolLM2-1.7B"
    """Name of the model to load."""

    dataset: str = "argilla/magpie-ultra-v0.1"
    """Dataset identifier to finetune on."""

    index_dataset: str = ""
    """Bergson index to use for attribution and loss filtering."""

    query_dataset: str = ""
    """
    Use the mean of this dataset's gradients as the query for attribution
    filtering. If unspecified the query is calculated over the index dataset.
    """

    query_scores: bool = False
    """Use the top-scored dataset items for the attribution query."""

    precondition: bool = False
    """Whether to use preconditioner for attribution filtering."""

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    num_examples: int = 30_000
    """Number of items to select from the training set."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    batch_size: int = 512
    """Batch size for processing the dataset."""

    seed: int = 42
    """Seed for reproducibility."""

    lowest: bool = False
    """Select the lowest scores."""

    sample: bool = False
    """Filter by sampling from the dataset without replacement with
    probability proportional to the filtering criteria."""

    temperature: float = 0.1
    """Temperature for sampling, used to control the distribution of
    the sampling probabilities. Lower values make the distribution more
    uniform, while higher values make it more peaked."""

    num_epochs: int = 1
    """Number of epochs to train for."""

    hf_token: str | None = None
    """Hugging Face token to use for the dataset."""

    dry_run: bool = False
    """Whether to run the script in dry run mode."""

    revision: str | None = None
    """Revision of the model to use."""

    query_method: Literal["mean", "nearest"] = "mean"
    """Method to use for computing the query."""


def worker(rank: int, world_size: int, cfg: FilterConfig, train, eval, run_name):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype="bfloat16",
        revision=cfg.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, max_length=8192)
    model, tokenizer = setup_chat_format(model, tokenizer)

    # https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config.yaml
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=eval,
        args=SFTConfig(
            max_length=8192,
            # max_seq_length=8192,
            output_dir=f"examples/runs/{run_name}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            learning_rate=3e-4,
            num_train_epochs=cfg.num_epochs,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            eval_steps=20,
            save_steps=100,
            save_total_limit=3,
            group_by_length=True,
            completion_only_loss=True,
            ddp_find_unused_parameters=False,
            seed=cfg.seed,
        ),
    )

    if cfg.dry_run:
        print("Dry run mode, exiting...")
        exit()

    trainer.train()

    if rank == 0:
        trainer.save_model(f"examples/runs/{run_name}")
        tokenizer.save_pretrained(f"examples/runs/{run_name}")

        if cfg.hf_token is not None:
            os.environ["HF_TOKEN"] = cfg.hf_token
            trainer.push_to_hub(repo_id=f"EleutherAI/{run_name}")
            tokenizer.push_to_hub(repo_id="EleutherAI")


def dist_worker(rank: int, world_size: int, cfg: FilterConfig, train, eval, run_name):
    try:
        worker(rank, world_size, cfg, train, eval, run_name)
    finally:
        dist.destroy_process_group()


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def add_index(
    dataset: Dataset,
    index: Path | None = None,
) -> Dataset:
    assert index is not None, "Index must be provided for attribution or loss filtering"

    gradient_ds = load_gradient_dataset(index).with_format("torch")

    def generator():
        for row, grad_row in zip(dataset, gradient_ds):
            yield {**dict(row), **dict(grad_row)}

    return assert_type(Dataset, Dataset.from_generator(generator))


def precondition(
    grad: Tensor, processor: GradientProcessor, shapes: dict[str, Sequence[int]]
) -> Tensor:
    named_grads = unflatten(grad, shapes)

    def precondition_module_grad(name: str, g: Tensor):
        P = processor.preconditioners[name]
        g = g.flatten(1).type_as(P)
        # Figure out why we use this method and if it makes sense in unbatched context
        g = torch.cholesky_solve(g.mT, P).mT

        return g

    return torch.cat(
        [precondition_module_grad(k, v) for k, v in named_grads.items()], dim=1
    )


def attribution_filter(
    args: FilterConfig,
    train: Dataset,
    model: PreTrainedModel,
    run_name: str,
    projection_dim: int = 16,
    query_method: Literal["mean", "nearest"] = "mean",
) -> Dataset:
    if args.query_scores:
        query_dataset = train.filter(lambda x: x["quality"] == "excellent")
    elif args.query_dataset:
        query_dataset = load_gradient_dataset(Path(args.query_dataset)).with_format(
            "torch"
        )
    else:
        query_dataset = train

    # Compute the mean of the normalized gradients in the query index
    if query_method == "mean":
        acc = {"sum": torch.zeros_like(query_dataset[0]["gradients"], device="cuda")}

        def sum_(col):
            acc["sum"] += col.cuda().sum(0)

        # RAM usage climbs here; it's intentionally only evicted under pressure
        # Do not use num_proc here because we are accumulating in a single variable
        # nproc solution must use reduce as in
        # https://colab.research.google.com/drive/1jCLv31Y4cDfqD0lhO0AnqEv3Or-LLvWe?usp=sharing
        query_dataset.map(
            sum_, input_columns="gradients", batched=True, batch_size=args.batch_size
        )

        query = acc["sum"] / len(query_dataset)
    elif query_method == "nearest":
        query = assert_type(Tensor, query_dataset["gradients"]).cuda()

    if args.precondition:
        # Load the gradient processor
        index_processor = GradientProcessor.load(
            Path(args.index_dataset), map_location="cuda"
        )
        target_info = GradientCollector(
            model.base_model, lambda _: _, index_processor
        ).target_info
        shapes: dict[str, Sequence[int]] = {
            k: [projection_dim, projection_dim] for k in target_info.keys()
        }

        query_processor = (
            GradientProcessor.load(Path(args.query_dataset), map_location="cuda")
            if args.query_dataset
            else index_processor
        )

        query = precondition(query.unsqueeze(0), query_processor, shapes).squeeze(0)
    else:
        index_processor, shapes = None, {}

    query /= query.norm()

    del query_dataset

    # Score the training set
    acc = {"scores": []}

    def score(batch):
        gradients_batch = batch.cuda()

        if index_processor and shapes:
            gradients_batch = precondition(gradients_batch, index_processor, shapes)

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query

        acc["scores"].append(batch_scores)

    def score_nearest(batch):
        gradients_batch = batch.cuda()

        if index_processor and shapes:
            gradients_batch = precondition(gradients_batch, index_processor, shapes)

        gradients_batch /= gradients_batch.norm(dim=1, keepdim=True)
        batch_scores = gradients_batch @ query.T

        # Take the maximum batch score for each item in the batch
        # (query has multiple rows)
        batch_scores = batch_scores.max(dim=-1).values

        acc["scores"].append(batch_scores)

    score_fn = score_nearest if query_method == "nearest" else score
    train.map(
        score_fn, input_columns="gradients", batched=True, batch_size=args.batch_size
    )
    importance_scores = torch.cat(acc["scores"], dim=0).cuda()

    print("Saving importance scores to disk.")
    os.makedirs(f"examples/runs/{run_name}", exist_ok=True)
    torch.save(importance_scores, f"examples/runs/{run_name}/importance_scores.pt")

    if args.sample:
        probs = torch.softmax(importance_scores / args.temperature, dim=0)
        selected_indices = torch.multinomial(
            probs, args.num_examples, replacement=False
        )
    else:
        # Select the top-k items
        sorted_scores = torch.argsort(importance_scores)
        selected_indices = (
            sorted_scores[: args.num_examples]
            if args.lowest
            else sorted_scores[-args.num_examples :]
        )

    return train.select(selected_indices)


def main(
    args: FilterConfig,
):
    set_seeds(args.seed)
    print("Running")

    if args.name is None:
        run_name = (
            f"{args.model.split('/')[-1]}-{args.dataset.split('/')[-1]}-{args.filter}"
            f"{'-lowest' if args.lowest else ''}"
            f"-s={args.seed}"
        )
    else:
        run_name = args.name

    if args.filter == "attribution" or args.filter == "loss":
        dataset = load_gradient_dataset(Path(args.index_dataset))
    else:
        dataset = assert_type(Dataset, load_dataset(args.dataset, split="train"))

    dataset.shuffle(args.seed)

    print("Spliting...")
    train, eval = dataset.train_test_split(
        test_size=0.05,
        seed=args.seed,
    ).values()

    train.set_format("torch")
    eval.set_format("torch")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16",
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)
    model, tokenizer = setup_chat_format(model, tokenizer)

    print("Filtering...")
    if args.num_examples == 0:
        pass
    elif args.filter == "attribution":
        train = attribution_filter(
            args, train, model, run_name, query_method=args.query_method
        )
    elif args.filter == "classification":
        if "score" in train.column_names:
            train = train.sort("score", reverse=not args.lowest)
            train = train.select(range(min(args.num_examples, len(train))))
        else:
            ranks = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "very poor": 0}

            def add_rank(ex):
                q = ex.get("quality")
                return {"_q": ranks.get(q, -1)}

            train = (
                train.map(add_rank)
                .filter(lambda x: x["_q"] >= 0)
                .sort("_q", reverse=not args.lowest)
                .select(range(min(args.num_examples, len(train))))
                .remove_columns("_q")
            )
    elif args.filter == "loss":
        train = train.map(
            lambda x: {"loss": x["loss"].item()},
            remove_columns=["gradients"],
        )
        # Select the top-k items
        sorted_scores = torch.argsort(train[:]["loss"])
        selected_indices = (
            sorted_scores[: args.num_examples]
            if args.lowest
            else sorted_scores[-args.num_examples :]
        )
        train = train.select(selected_indices)
    elif args.filter == "random":
        train = train.select(range(min(args.num_examples, len(train))))
    else:
        raise ValueError(f"Invalid filter: {args.filter}")

    # Create DataConfig for tokenization
    data_config = DataConfig(
        prompt_column=args.prompt_column,
        completion_column=args.completion_column,
        conversation_column=args.conversation_column,
    )

    train = train.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )
    eval = eval.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, args, train, eval, run_name)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "build",
            dist_worker,
            args={
                i: (i, world_size, args, train, eval, run_name)
                for i in range(world_size)
            },
            envs={
                i: {
                    "LOCAL_RANK": str(i),
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(port),
                }
                for i in range(world_size)
            },
            logs_specs=DefaultLogsSpecs(),
        )
        ctx.wait()


if __name__ == "__main__":
    args = parse(FilterConfig)

    main(args)
