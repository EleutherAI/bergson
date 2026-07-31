"""Normalization of the segment EK-FAC eigenvalues fed to SOURCE's eigenfunctions.

kronfluence divides its lambda matrix by a per-sample count before use
(``kronfluence/factor/config.py`` ``lambda_matrix.div_(NUM_LAMBDA_PROCESSED)``,
counted in ``module/tracker/factor.py`` as ``per_sample_gradient.size(0)``).
``LambdaCollector`` accumulates an unnormalized sum, so the segment aggregation
divides by the pooled count to match.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from bergson.approx_unrolling.segment_aggregation import (
    LAMBDA_COUNTS_FILENAME,
    _sum_my_shard,
    lambda_denominator,
    write_lambda_counts,
)

DOCS, TOKENS = 4599, 2354682


def _ckpt_dirs(tmp_path, n=2):
    dirs = []
    for i in range(n):
        d = tmp_path / f"ckpt_{i}"
        d.mkdir()
        write_lambda_counts(d, documents=DOCS, tokens=TOKENS)
        dirs.append(d)
    return dirs


def test_counts_roundtrip(tmp_path):
    write_lambda_counts(tmp_path, documents=DOCS, tokens=TOKENS)
    with open(tmp_path / LAMBDA_COUNTS_FILENAME) as f:
        assert json.load(f) == {"documents": DOCS, "tokens": TOKENS}


@pytest.mark.parametrize(
    "normalization, expected",
    [("document", 2 * DOCS), ("token", 2 * TOKENS), ("none", 1.0)],
)
def test_denominator_pools_over_checkpoints(tmp_path, normalization, expected):
    """The lambdas are summed over checkpoints, so the denominator sums too."""
    assert lambda_denominator(_ckpt_dirs(tmp_path), normalization) == expected


def test_denominator_reports_missing_counts(tmp_path):
    d = tmp_path / "ckpt_0"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="fisher_normalization"):
        lambda_denominator([d], "document")


def test_denominator_ignores_missing_counts_when_unnormalized(tmp_path):
    """ "none" reproduces pre-normalization runs, which have no counts.json."""
    d = tmp_path / "ckpt_0"
    d.mkdir()
    assert lambda_denominator([d], "none") == 1.0


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


def test_sum_my_shard_defaults_to_plain_sum(tmp_path):
    p = tmp_path / "in.safetensors"
    save_file({"w": torch.ones(2, 3)}, p)
    out = tmp_path / "out.safetensors"
    _sum_my_shard([p], out, device="cpu")
    assert torch.allclose(load_file(out)["w"], torch.ones(2, 3))
