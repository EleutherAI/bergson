"""TKFAC (Gao et al. 2021) approximates the Fisher by ``δ Φ ⊗ Ψ``, with traces
taken per position and ``δ = 1 / tr(F)`` so the approximation keeps ``tr(F)``."""

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig

from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import fit_factored_hessians
from bergson.utils.utils import get_device

NUM_DOCUMENTS = 32


def make_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    return model.eval().to(get_device(0))


def fit(tmp_path, data: Dataset, docs_per_batch: int, ev_correction: bool) -> str:
    index_cfg = IndexConfig(run_path=str(tmp_path / "run"))
    index_cfg.partial_run_path.mkdir(parents=True)
    path = str(tmp_path / "hessian")
    fit_factored_hessians(
        make_model(),
        data,
        index_cfg,
        HessianConfig(
            method="tkfac", ev_correction=ev_correction, use_dataset_labels=True
        ),
        target_modules=None,
        attention_cfgs={},
        batches=[
            list(range(i, i + docs_per_batch))
            for i in range(0, len(data), docs_per_batch)
        ],
        path=path,
        do_eigendecomposition=True,
    )
    return path


@pytest.mark.parametrize("docs_per_batch", [1, 8])
def test_tkfac_keeps_the_fisher_trace(tmp_path, docs_per_batch):
    """With one gradient-carrying position per document, a module's gradient is
    ``g aᵀ``, so the EK-FAC corrected eigenvalues sum to the exact Fisher trace
    ``E[|a|² |g|²]``. TKFAC's eigenvalue grid must sum to the same, however the
    documents are batched."""
    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": torch.randint(0, 64, (NUM_DOCUMENTS, 2)).tolist()}
    )
    path = fit(tmp_path, data, docs_per_batch, ev_correction=True)

    tkfac = load_file(f"{path}/eigenvalue_sharded/shard_0.safetensors")
    exact = load_file(f"{path}/eigenvalue_correction_sharded/shard_0.safetensors")
    for name in tkfac:
        torch.testing.assert_close(
            tkfac[name].sum(), exact[name].sum(), rtol=1e-4, atol=1e-12, msg=name
        )


def test_tkfac_factors_do_not_depend_on_batching(tmp_path):
    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": torch.randint(0, 64, (NUM_DOCUMENTS, 6)).tolist()}
    )
    factors = []
    for docs_per_batch in (1, 8):
        path = fit(tmp_path / str(docs_per_batch), data, docs_per_batch, False)
        factors.append(
            {
                sub: load_file(f"{path}/{sub}/shard_0.safetensors")
                for sub in ("activation_sharded", "gradient_sharded")
            }
        )

    for sub, tensors in factors[0].items():
        for name, tensor in tensors.items():
            # Batching only reorders the sums, so compare relative to the scale.
            atol = 1e-5 * tensor.abs().max().item()
            torch.testing.assert_close(
                factors[1][sub][name], tensor, rtol=1e-4, atol=atol, msg=name
            )
