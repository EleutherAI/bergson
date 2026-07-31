"""Segment eigenvalues are normalized per document, matching kronfluence's
``lambda_matrix.div_(num_lambda_processed)``."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from bergson.approx_unrolling.segment_aggregation import (
    DOCUMENTS_PROCESSED_FILENAME,
    _sum_my_shard,
    lambda_denominator,
)

DOCS = 4599


def test_denominator_pools_over_checkpoints(tmp_path):
    dirs = []
    for i in range(2):
        d = tmp_path / f"ckpt_{i}"
        d.mkdir()
        torch.save(torch.tensor(DOCS), d / DOCUMENTS_PROCESSED_FILENAME)
        dirs.append(d)
    assert lambda_denominator(dirs) == 2 * DOCS


def test_denominator_reports_missing_counts(tmp_path):
    d = tmp_path / "ckpt_0"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="lambda step"):
        lambda_denominator([d])


def test_sum_my_shard_divides(tmp_path):
    ins = []
    for i in range(2):
        p = tmp_path / f"in_{i}.safetensors"
        save_file({"w": torch.full((2, 3), float(i + 1))}, p)
        ins.append(p)

    out = tmp_path / "out.safetensors"
    _sum_my_shard(ins, out, device="cpu", divisor=6.0)
    # (1 + 2) / 6
    assert torch.allclose(load_file(out)["w"], torch.full((2, 3), 0.5))
