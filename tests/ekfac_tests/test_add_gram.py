"""Covariance products of half-precision rows sum the same products as
casting the rows to fp32 first."""

import pytest
import torch

from bergson.hessians.kfac import add_gram


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_half_precision_rows_add_their_fp32_gram(device: str, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(512, 96, generator=generator).to(device, dtype)
    start = torch.randn(96, 96, generator=generator).to(device)
    out = start.clone()
    add_gram(out, x)
    expected = start.double() + x.double().mT @ x.double()
    # Products are exact in fp32, so only the order of the fp32 sums differs
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-4)
