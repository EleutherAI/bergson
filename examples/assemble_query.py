# TODAY
# Assemble dataset!! 6 queries
# Try multi node generation
# I believe the MCQA and Cloze setups are pulled from the same eval and are
# both roughly 1k rows, like the original wmdp-bio as a whole.

import subprocess

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)

from bergson import DataConfig, IndexConfig, ReduceConfig, load_gradients
from bergson.data import tokenize
from bergson.utils import assert_type


def lm_eval_harness_format(x):
    """Format the MCQA as they are for models without a chat template
    in LM Eval Harness, but with the answer appended after a space.'"""

    question = x["question"]
    choices = x["choices"]

    prompt = (
        f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\n"
        f"C. {choices[2]}\nD. {choices[3]}\nAnswer: {x['answer']}"
    )

    return {
        "text": prompt,
        "subset": x["subset"],
    }


def load_mcqa_dataset(dataset_name="EleutherAI/wmdp_bio_robust_mcqa"):
    subsets = get_dataset_config_names(dataset_name)
    mcqa_datasets = []
    for subset in subsets:
        ds = assert_type(Dataset, load_dataset(dataset_name, subset, split="robust"))
        ds = ds.add_column("subset", [subset] * len(ds))
        mcqa_datasets.append(ds)

    return concatenate_datasets(mcqa_datasets)


def tokenize_mcqa(
    batch: dict,
    *,
    tokenizer,
    args: DataConfig,
    answer_marker: str = "Answer:",
):
    """
    Custom tokenizer for this MCQA experiment that only keeps labels on the
    final answer span so gradient collection ignores the rest of the prompt.

    Codex wrote this.
    """
    # TODO integrate custom masking into tokenize if necessary
    return tokenize(batch, args=args, tokenizer=tokenizer, apply_chat_template=False)


# def create_query_index(
#     query_ds: Dataset,
#     run_path: str,
#     assembled_dataset_path: str,
#     index_dtype: np.dtype
# ):
#     structured_mmap = load_gradients(run_path)
#     mmap_dtype = structured_mmap.dtype

#     # Copy into memory
#     gradient_tensor = torch.tensor(structured_to_unstructured(structured_mmap)).to(
#         torch.float32
#     )

#     print("mmap sum", gradient_tensor.sum())
#     print("mmap sum", gradient_tensor.abs().sum())

#     # Group mmap gradient rows by the subset they came from
#     subset_gradients = defaultdict(list)
#     for grads_row, ds_row in zip(gradient_tensor, query_ds):
#         subset_gradients[ds_row["subset"]].append(grads_row)

#     subset_mean_gradients = {"overall": gradient_tensor.mean(dim=0)}
#     for subset, gradients in subset_gradients.items():
#         mean_gradient = torch.stack(gradients).mean(dim=0)
#         subset_mean_gradients[subset] = mean_gradient

#     # Copy everything from the origin run path to the new path
#     # except gradients.bin and data.hf
#     os.makedirs(assembled_dataset_path, exist_ok=True)
#     for item in os.listdir(run_path):
#         if item not in ["gradients.bin", "data.hf"]:
#             dest = Path(assembled_dataset_path) / item
#             shutil.copy(Path(run_path) / item, dest)

#     if (Path(assembled_dataset_path) / "data.hf").exists():
#         if (Path(assembled_dataset_path) / "data.hf").is_file():
#             (Path(assembled_dataset_path) / "data.hf").unlink()
#         else:
#             shutil.rmtree(Path(assembled_dataset_path) / "data.hf")

#     # Write structured mean queries to data.hf
#     np_mean_grads = np.stack(
#         [item.numpy() for item in list(subset_mean_gradients.values())], axis=0
#     )
#     # structured_np_mean_grads = unstructured_to_structured(np_mean_grads, mmap_dtype)
#     # data = [
#     #     {
#     #         name: structured_np_mean_grads[name][i].tolist()
#     #         for name in mmap_dtype.names
#     #     }
#     #     for i in range(structured_np_mean_grads.shape[0])
#     # ]

#     means_dataset = Dataset.from_dict(
#         {
#             "scores": [0.0] * len(subset_mean_gradients),
#         }
#     )
#     means_dataset.save_to_disk(Path(assembled_dataset_path) / "data.hf")

#     mean_grad_stack = torch.stack(list(subset_mean_gradients.values()))
#     first_query_grad = gradient_tensor[0].unsqueeze(0).expand_as(mean_grad_stack)
#     cosine_sims = torch.nn.functional.cosine_similarity(
#         mean_grad_stack, first_query_grad, dim=1
#     )

#     # Assemble grad sizes
#     grad_sizes = {}
#     for name in mmap_dtype.names:
#         field_dtype = mmap_dtype.fields[name][0]
#         subdtype = field_dtype.subdtype
#         assert subdtype is not None

#         _, shape = subdtype
#         grad_sizes[name] = int(np.prod(shape))

#     # Create and populate the index
#     index_grads = create_index(
#         str(assembled_dataset_path),
#         len(subset_mean_gradients),
#         grad_sizes,
#         index_dtype
#     )
#     index_grads[:] = unstructured_to_structured(
#         np_mean_grads.astype(index_dtype), mmap_dtype
#     )
#     index_grads.flush()

#     load_gradient_dataset(assembled_dataset_path)

#     mean_grad_stack = torch.stack(list(subset_mean_gradients.values()))
#     first_query_grad = gradient_tensor[1].unsqueeze(0).expand_as(mean_grad_stack)
#     cosine_sims = torch.nn.functional.cosine_similarity(
#         mean_grad_stack, first_query_grad, dim=1
#     )
#     if torch.any(cosine_sims <= 0.09):
#         raise ValueError(
#             f"Cosine similarity between mean gradients and the first query gradient "
#             f"is not greater than 0.09. Cosine sims: {cosine_sims}"
#         )
#     else:
#         print(f"Cosine sims: {cosine_sims}")


def main():
    # TODO migrate to a larger model
    model_name = "EleutherAI/deep_ignorance_pretraining_baseline_small"
    ds_path = "runs/ds_wmdp_bio_robust_mcqa"
    projection_dim = 256

    # Spend all day on getting a setup without FSDP working.
    index_path = f"runs/wmdp_bio_robust_mcqa_query_{projection_dim}"

    mcqa_ds = assert_type(Dataset, load_mcqa_dataset())
    mcqa_ds = mcqa_ds.map(
        lm_eval_harness_format, remove_columns=["choices", "answer", "question"]
    )
    mcqa_ds.save_to_disk(ds_path)

    # Add chat template following whatever the original deep ignorance project did
    data_config = DataConfig(
        dataset=ds_path,
        prompt_column="text",
    )

    cfg = IndexConfig(
        run_path=index_path,
        # precision="fp16",
        data=data_config,
        # fsdp=True,
        model=model_name,
        projection_dim=projection_dim,
        skip_preconditioners=False,
        token_batch_size=800,
        precision="fp16",
    )
    reduce_cfg = ReduceConfig(
        method="mean",
        unit_normalize=True,
    )

    cmd = [
        "bergson",
        "reduce",
        index_path,
        "--dataset",
        cfg.data.dataset,
        "--prompt_column",
        cfg.data.prompt_column,
        "--model",
        cfg.model,
        "--projection_dim",
        str(cfg.projection_dim),
        "--token_batch_size",
        str(cfg.token_batch_size),
        "--method",
        reduce_cfg.method,
        "--unit_normalize",
        str(reduce_cfg.unit_normalize),
        "--precision",
        cfg.precision,
        "--fsdp",  # Need more memory available when computing the preconditioner
    ]

    print(" ".join(cmd))
    exit()

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Trackstar uses 2**16 with an 8B model
    # We are collecting gradients for a ~2.7B model
    # We are using ~2**13 I think
    modules = set(load_gradients(cfg.run_path).dtype.names)
    print(
        f"Full projection dim: {len(modules) * cfg.projection_dim * cfg.projection_dim}"
    )

    exit()


if __name__ == "__main__":
    main()
