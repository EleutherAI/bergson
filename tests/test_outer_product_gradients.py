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
