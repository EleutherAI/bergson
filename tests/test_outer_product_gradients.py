"""OuterProductGradients gives the same results as the gradients it forms."""

import pytest
import torch

from bergson.gradients import OuterProductGradients


# With 9 inputs and 5 or 12 outputs, 3 queries contract each way and 64 form
# the gradients.
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("divisor", [False, True])
@pytest.mark.parametrize("num_queries", [3, 64])
@pytest.mark.parametrize("o", [5, 12])
def test_random_vectors_match_formed_gradients(bias, divisor, num_queries, o):
    """Every combination of terms, including ones a model's gradients never
    produce, such as a divisor with a projected bias column."""
    torch.manual_seed(0)
    T, w = 7, 9
    grads = OuterProductGradients(
        g=torch.randn(T, o),
        a=torch.randn(T, w),
        bias=torch.randn(T, o) if bias else None,
        bias_col=torch.randn(w) if bias else None,
        divisor=torch.rand(o, w) + 0.5 if divisor else None,
    )
    formed = grads.materialize().flatten(1)
    torch.testing.assert_close(grads.sq_norm(), formed.pow(2).sum(1))
    q = torch.randn(num_queries, o * w)
    torch.testing.assert_close(grads.dot(q), formed @ q.T)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_queries", [3, 64])
def test_half_precision_with_small_adam_denominators(dtype, num_queries):
    """Adam's denominators stay in float32: inverted, they overflow float16."""
    torch.manual_seed(0)
    T, o, w = 7, 5, 9
    grads = OuterProductGradients(
        g=torch.randn(T, o) * 1e-3,
        a=torch.randn(T, w) * 1e-3,
        bias=torch.randn(T, o),
        bias_col=torch.randn(w),
        divisor=torch.rand(o, w) * 1e-6 + 1e-6,
    )
    formed = grads.materialize().flatten(1)
    half = grads.to(torch.device("cpu"), dtype)
    assert half.divisor is not None and half.divisor.dtype == torch.float32

    torch.testing.assert_close(
        half.sq_norm(), formed.pow(2).sum(1), check_dtype=False, atol=0.0, rtol=0.05
    )
    q = torch.randn(num_queries, o * w)
    expected = formed @ q.T
    torch.testing.assert_close(
        half.dot(q.to(dtype)).float(),
        expected,
        atol=0.05 * expected.abs().max(),
        rtol=0,
    )
