"""Tools for tokenizing and manipulating text datasets."""

import math
from multiprocessing import cpu_count
from typing import TypeVar, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from .utils import assert_type

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def tokenize_and_pad(raw_ds, tokenizer, max_seq_len):
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # keep as lists for HF datasets
        )

    ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)
    ds.set_format("torch")
    return ds


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = assert_type(str, tokenizer.eos_token or "<|endoftext|>")
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        tokens = assert_type(list, output["input_ids"])
        output_batch_size = len(tokens)

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data2 = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    data2 = data2.with_format(format, columns=["input_ids"])

    # Make sure HuggingFace hasn't changed the type of the dataset
    assert isinstance(data2, type(data))
    return data2


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names
