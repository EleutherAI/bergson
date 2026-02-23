"""Test that second-order gradients flow through DTensor redistribute + to_local.

MAGIC attribution (Trainer.backward) requires computing d(new_params)/d(weights)
where new_params = params - lr * grad(loss, params). This is a second-order
derivative: the first-order grad(loss, params) is computed with create_graph=True,
and then we differentiate the resulting update w.r.t. DataStream.weights.

Parameters are DTensors with Shard(0) placement. During forward,
redistribute(Replicate()) + to_local() all-gathers them. The second-order
gradient must flow back through these DTensor ops to reach DataStream.weights.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import (
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
    init_device_mesh,
)


def _worker(global_rank: int, rank: int, world_size: int, results_path: str):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    addr = os.environ.get("MASTER_ADDR", "localhost")
    port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{addr}:{port}",
        device_id=torch.device(device),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=2),
    )

    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(42)

    # Shard a weight matrix across ranks (like Trainer params under FSDP)
    W_full = torch.randn(8, 4, device=device)
    with mesh:
        W_dt = nn.Parameter(distribute_tensor(W_full, placements=(Shard(0),)))

    # Per-example weights — this is DataStream.weights
    weights = nn.Parameter(torch.ones(4, device=device))

    # All-gather W for the forward pass (redistribute + to_local)
    W_local = W_dt.redistribute(placements=(Replicate(),)).to_local(
        grad_placements=(Partial(reduce_op="avg"),)
    )

    # Each rank processes its data slice (like DataStream.__getitem__)
    x = torch.randn(4, 4, device=device)
    local_x = x[rank::world_size]
    local_w = weights[rank::world_size]

    logits = local_x @ W_local.T
    loss = (logits.sum(dim=-1) * local_w).mean()

    # First-order gradient with create_graph (like Trainer.step with trace=True)
    (grad_W,) = torch.autograd.grad(loss, W_dt, create_graph=True)

    # SGD update
    W_new = W_dt - 0.01 * grad_W

    # Second-order: d(W_new)/d(weights) — Trainer.backward needs this to be non-None
    (w_grad,) = torch.autograd.grad(
        W_new,
        weights,
        grad_outputs=torch.ones_like(W_new),
        allow_unused=True,
    )

    if rank == 0:
        torch.save(w_grad, results_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need >= 2 GPUs")
def test_dtensor_second_order_grad(tmp_path):
    """d(W - lr * grad(loss, W)) / d(weights) must be non-None through DTensor ops."""
    from bergson.distributed import launch_distributed_run

    results_path = str(tmp_path / "w_grad.pt")
    launch_distributed_run("test-dt-grad", _worker, [results_path])

    w_grad = torch.load(results_path, weights_only=True)
    assert w_grad is not None, (
        "Second-order gradient through DTensor redistribute() + to_local() is "
        "None. This breaks Trainer.backward (MAGIC attribution) because "
        "DataStream.weights gradients cannot flow through the DTensor ops."
    )
    assert w_grad.any(), "Gradient is all zeros"
