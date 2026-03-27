"""Tests for truncation and max_length handling."""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from bergson import GradientProcessor, collect_gradients
from bergson.config import DataConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.utils.worker_utils import setup_data_pipeline

# GPT-2: model_max_length=max_position_embeddings
# Pythia-14m: model_max_length=very large
GPT2 = "openai-community/gpt2"
PYTHIA = "EleutherAI/pythia-14m"


def get_max_position_embeddings(model_name):
    """The maximum supported context length."""
    return AutoConfig.from_pretrained(model_name).max_position_embeddings


GPT2_MAX_POS_EMB = get_max_position_embeddings(GPT2)
PYTHIA_MAX_POS_EMB = get_max_position_embeddings(PYTHIA)


def create_documents_file(tmp_path, model, doc_tokens):
    """Create a JSON file with documents of exactly `doc_tokens` tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Generate exact token count by decoding repeated token IDs
    token_id = tokenizer.encode("hello")[0]
    tokens = [token_id] * doc_tokens
    text = tokenizer.decode(tokens)
    assert len(tokenizer.encode(text)) == doc_tokens
    data = [{"text": text}]
    path = tmp_path / "docs.json"
    path.write_text("\n".join(json.dumps(d) for d in data))
    return str(path)


def run_pipeline(tmp_path, model_name, token_batch_size, doc_tokens, truncation):
    documents_file = create_documents_file(tmp_path, model_name, doc_tokens)
    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        model=model_name,
        token_batch_size=token_batch_size,
        skip_preconditioners=True,
        data=DataConfig(dataset=documents_file, truncation=truncation),
    )
    ds, _ = setup_data_pipeline(cfg)
    batches = allocate_batches(ds["length"][:], token_batch_size)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    collect_gradients(
        model=model,
        data=ds,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
        batches=batches,
    )
    return ds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("truncation", [True, False])
@pytest.mark.parametrize(
    "model,token_batch_size",
    [
        # token_batch_size < max_position_embeddings
        (GPT2, GPT2_MAX_POS_EMB // 2),
        (PYTHIA, PYTHIA_MAX_POS_EMB // 2),
        # token_batch_size = max_position_embeddings
        (GPT2, GPT2_MAX_POS_EMB),
        (PYTHIA, PYTHIA_MAX_POS_EMB),
        # token_batch_size > max_position_embeddings
        (PYTHIA, PYTHIA_MAX_POS_EMB * 2),
    ],
)
def test_short_documents(tmp_path, model, token_batch_size, truncation):
    """Short documents (fit within token_batch_size and max_position_embeddings)
    work regardless of truncation setting.
    """
    max_position_embeddings = get_max_position_embeddings(model)
    doc_tokens = min(token_batch_size, max_position_embeddings) // 2
    ds = run_pipeline(tmp_path, model, token_batch_size, doc_tokens, truncation)

    assert max(ds["length"]) == doc_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "model,token_batch_size",
    [
        # token_batch_size < max_position_embeddings: truncates to token_batch_size
        (GPT2, GPT2_MAX_POS_EMB // 2),
        (PYTHIA, PYTHIA_MAX_POS_EMB // 2),
        # token_batch_size = max_position_embeddings: truncates to both
        (GPT2, GPT2_MAX_POS_EMB),
        (PYTHIA, PYTHIA_MAX_POS_EMB),
        # token_batch_size > max_position_embeddings:
        # truncates to max_position_embeddings
        # (for GPT2, see `test_token_batch_size_exceeds_model_max_length`)
        (PYTHIA, PYTHIA_MAX_POS_EMB * 2),
    ],
)
def test_long_documents_truncated(tmp_path, model, token_batch_size):
    """Long documents get truncated to
    min(token_batch_size, max_position_embeddings).
    """
    doc_tokens = token_batch_size * 2
    max_position_embeddings = get_max_position_embeddings(model)
    expected_length = min(token_batch_size, max_position_embeddings)
    assert doc_tokens > expected_length
    ds = run_pipeline(tmp_path, model, token_batch_size, doc_tokens, truncation=True)
    assert max(ds["length"]) == expected_length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "model,token_batch_size",
    [
        # token_batch_size < max_position_embeddings
        (GPT2, GPT2_MAX_POS_EMB // 2),
        (PYTHIA, PYTHIA_MAX_POS_EMB // 2),
        # token_batch_size = max_position_embeddings
        (GPT2, GPT2_MAX_POS_EMB),
        (PYTHIA, PYTHIA_MAX_POS_EMB),
        # token_batch_size > max_position_embeddings
        # (for GPT2, see `test_token_batch_size_exceeds_model_max_length`)
        (PYTHIA, PYTHIA_MAX_POS_EMB * 2),
    ],
)
def test_long_documents_fail_without_truncation(tmp_path, model, token_batch_size):
    """Without truncation, we fail when a document exceeds token_batch_size."""
    doc_tokens = token_batch_size + 1
    with pytest.warns(UserWarning, match="longer than the model can handle"):
        with pytest.raises(RuntimeError, match="too long"):
            run_pipeline(
                tmp_path, model, token_batch_size, doc_tokens, truncation=False
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_long_documents_warn_without_truncation(tmp_path):
    """Without truncation, we warn when a document exceeds max_position_embeddings
    but not token_batch_size.
    """
    # Only possible when token_batch_size > max_position_embeddings (requires PYTHIA)
    token_batch_size = PYTHIA_MAX_POS_EMB * 3
    doc_tokens = (
        PYTHIA_MAX_POS_EMB * 2
    )  # max_position_embeddings < doc_tokens < token_batch_size
    with pytest.warns(UserWarning, match="longer than the model can handle"):
        run_pipeline(tmp_path, PYTHIA, token_batch_size, doc_tokens, truncation=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_batch_size_exceeds_model_max_length(tmp_path):
    """token_batch_size > model_max_length raises an error
    (GPT2 has model_max_length=max_position_embeddings).
    """
    token_batch_size = GPT2_MAX_POS_EMB * 2
    doc_tokens = GPT2_MAX_POS_EMB // 2
    with pytest.raises(ValueError, match="exceeds model max length"):
        run_pipeline(tmp_path, GPT2, token_batch_size, doc_tokens, truncation=True)
