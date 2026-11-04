"""Covariance products, summed by upper row blocks and mirrored, of
half-precision rows sum the same products as casting the rows to fp32 first."""

import pytest
import torch

from bergson.hessians.kfac import add_gram, mirror_upper


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
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
def test_rows_add_their_fp32_gram(device: str, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(512, 96, generator=generator).to(device, dtype)
    start = torch.randn(96, 96, generator=generator)
    start = (start + start.mT).to(device)
    out = start.clone()
    # Blocks that don't divide the width
    add_gram(out, x, block=40)
    add_gram(out, x, block=40)
    mirror_upper(out)
    expected = start.double() + 2 * x.double().mT @ x.double()
    # Products are exact in fp32, so only the order of the fp32 sums differs
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-4)
