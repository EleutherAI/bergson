"""Test that covariance traces are batch-size invariant after normalization."""

import tempfile
from typing import Any

import torch

from bergson.data import IndexConfig
from bergson.hessians.ekfac_compute import EkfacComputer
from tests.ekfac_tests.test_utils import load_covariances


def test_trace_batch_invariant(ground_truth_setup: dict[str, Any]):
    """Normalized covariance traces should be the same regardless of batch size."""
    setup = ground_truth_setup
    indices = [
        idx
        for worker_batches in setup["batches_world"]
        for batch in worker_batches
        for idx in batch
    ]

    # B=1 vs B=2 batches
    batches_b1 = [[i] for i in indices]
    batches_b2 = [indices[i : i + 2] for i in range(0, len(indices), 2)]

    def compute_traces(batches):
        with tempfile.TemporaryDirectory() as tmpdir:
            ekfac = EkfacComputer(
                model=setup["model"],
                data=setup["data"],
                batches=batches,
                target_modules=setup["target_modules"],
                cfg=IndexConfig(run_path=tmpdir, data=None),
            )
            ekfac.compute_covariance()

            A, G, n = load_covariances(tmpdir)

            return (
                sum(v.trace().item() / n for v in A.values()),
                sum(v.trace().item() / n for v in G.values()),
            )

    setup["model"].eval()
    A1, G1 = compute_traces(batches_b1)
    A2, G2 = compute_traces(batches_b2)

    torch.testing.assert_close(A1, A2, rtol=0.01, atol=0)
    torch.testing.assert_close(G1, G2, rtol=0.2, atol=0)
