"""ASTRA refinement of the Adam SOURCE variant's per-segment EK-FAC inverse."""

import pytest

from bergson.config.config import (
    ApproxUnrollingConfig,
    AstraConfig,
    DistributedConfig,
    HessianConfig,
    IndexConfig,
)
from bergson.hessians.pipeline import HessianPipelineConfig  # noqa: F401


def _cfgs(**astra):
    index = IndexConfig(run_path="runs/x", model="gpt2")
    unroll = ApproxUnrollingConfig(
        checkpoints=["runs/x/checkpoint-1", "runs/x/checkpoint-2"],
        segments=2,
        astra=AstraConfig(**astra),
    )
    return index, HessianConfig(method="kfac", ev_correction=True), unroll


def test_astra_is_off_by_default():
    _, _, unroll = _cfgs()
    assert unroll.astra.num_steps == 0


def test_astra_needs_the_adam_variant():
    """The SGD variant applies no inverse, so there is nothing to refine."""
    from bergson.approx_unrolling.pipeline import approx_unrolling_pipeline

    index, hessian, unroll = _cfgs(num_steps=10)
    assert not unroll.use_adam_preconditioner
    with pytest.raises(ValueError, match="use_adam_preconditioner"):
        approx_unrolling_pipeline(index, hessian, unroll)


def test_astra_rejects_projected_gradients():
    from bergson.approx_unrolling.pipeline import approx_unrolling_pipeline

    index, hessian, unroll = _cfgs(num_steps=10)
    unroll.use_adam_preconditioner = True
    index.projection_dim = 16
    index.distributed = DistributedConfig(nproc_per_node=1)
    with pytest.raises(ValueError, match="projection_dim"):
        approx_unrolling_pipeline(index, hessian, unroll)
