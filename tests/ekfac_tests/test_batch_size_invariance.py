"""Test that covariance traces are batch-size invariant after normalization."""

import tempfile
from pathlib import Path

import pytest
import torch

from bergson.collector.collector import CollectorComputer, fwd_bwd_hessian_factory
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances, set_all_seeds
from tests.ekfac_tests.toy_model import (
    ToyDataConfig,
    ToyLM,
    ToyLMConfig,
    generate_batches,
    generate_dataset,
)


@pytest.mark.parametrize(
    "seq_lengths, num_batches",
    [
        ((16,), 20),  # Single sequence length
        ((16, 8), 20),  # Mixed sequence lengths
        ((64,), 10),  # Longer sequences
    ],
)
def test_trace_batch_invariant(seq_lengths, num_batches, tmp_path):
    """Normalized covariance traces should be the same regardless of batch size."""
    set_all_seeds(42)

    config = ToyDataConfig(
        vocab_size=16,
        hidden_size=8,
        seq_lengths=seq_lengths,
        num_batches=num_batches,
    )
    device = torch.device(get_device())

    dataset = generate_dataset(config)
    batches = generate_batches(config)

    model_config = ToyLMConfig(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size
    )
    model = ToyLM(
        model_config,
        training_data=dataset,
        training_batches=batches,
        device=device,
    )

    # Flatten all indices from batches
    indices = [idx for batch in batches for idx in batch]

    # B=1 vs B=2 batches
    batches_b1 = [[i] for i in indices]
    batches_b2 = [indices[i : i + 2] for i in range(0, len(indices), 2)]

    def compute_traces(batches: list[list[int]]) -> tuple[float, float]:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "run"
            index_cfg = IndexConfig(run_path=str(run_path), loss_reduction="sum")

            collector = CovarianceCollector(
                model=model.base_model,
                target_modules={"linear"},
                dtype=torch.float32,
                path=str(index_cfg.partial_run_path),
            )

            hessian_cfg = HessianConfig()
            computer = CollectorComputer(
                model=model,
                data=dataset,
                batches=batches,
                collector=collector,
                cfg=index_cfg,
            )
            computer.forward_backward = fwd_bwd_hessian_factory(index_cfg, hessian_cfg)
            computer.run_with_collector_hooks()

            # Load covariances
            A = load_sharded_covariances(
                index_cfg.partial_run_path / "activation_sharded"
            )
            G = load_sharded_covariances(
                index_cfg.partial_run_path / "gradient_sharded"
            )
            n = torch.load(index_cfg.partial_run_path / "total_processed.pt").item()

            return (
                sum(v.trace().item() / n for v in A.values()),
                sum(v.trace().item() / n for v in G.values()),
            )

    model.eval()
    A1, G1 = compute_traces(batches_b1)
    A2, G2 = compute_traces(batches_b2)

    torch.testing.assert_close(A1, A2, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(G1, G2, rtol=1e-2, atol=1e-4)
