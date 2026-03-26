"""Test that DDP MAGIC produces the same attribution scores as single-process."""

import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchopt
from datasets import Dataset
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce


def _make_model():
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    return model


def _make_dataset():
    return Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            "labels": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
        }
    )


def _run_magic(model, dataset, device="cpu", ckpt_dir=None):
    """Run full MAGIC pipeline and return attribution scores."""
    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    batch_size = len(dataset)
    stream = DataStream(dataset, batch_size=batch_size, device=device)
    assert len(stream) >= 1

    _tmpdir = tempfile.TemporaryDirectory() if ckpt_dir is None else None
    if _tmpdir is not None:
        ckpt_dir = _tmpdir.name

    try:
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpt_dir)

        with fwd_state.activate(model) as params:
            batch = stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }

        # Average query gradients across ranks (matches compute_query_gradients)
        if dist.is_initialized():
            for g in query_grads.values():
                dist.all_reduce(g, op=dist.ReduceOp.AVG)

        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(fwd_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        bwd_state = BackwardState(
            query_grads, opt_grads, torch.zeros_like(stream.weights)
        )

        stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir, stream, bwd_state, fwd_state, inplace=True
        )
    finally:
        if _tmpdir is not None:
            _tmpdir.cleanup()

    scores = bwd_state.weight_grads.detach()
    if dist.is_initialized():
        dist.all_reduce(scores, op=dist.ReduceOp.SUM)
    return scores.cpu()


def _ddp_worker(rank, world_size, port, dataset, result_dict, ckpt_dir):
    """Worker function for distributed MAGIC test."""
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}"),
        )

        model = _make_model().to(f"cuda:{rank}")
        scores = _run_magic(model, dataset, device=f"cuda:{rank}", ckpt_dir=ckpt_dir)
        result_dict[rank] = scores
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need >= 2 GPUs for DDP test",
)
def test_ddp_matches_single_process():
    """DDP MAGIC scores should match single-process scores."""
    dataset = _make_dataset()

    # ── Single-process baseline (on GPU for identical numerics) ──
    model = _make_model().to("cuda:0")
    expected = _run_magic(model, dataset, device="cuda:0")
    del model

    # ── Distributed run ──
    world_size = min(torch.cuda.device_count(), len(dataset))
    assert len(dataset) % world_size == 0, "Dataset must be divisible by world_size"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    result_dict = manager.dict()

    with tempfile.TemporaryDirectory() as shared_ckpt_dir:
        mp.spawn(
            _ddp_worker,
            args=(world_size, port, dataset, result_dict, shared_ckpt_dir),
            nprocs=world_size,
            join=True,
        )

    actual = result_dict[0]

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-4,
        rtol=1e-3,
        msg="DDP attribution scores diverged from single-process",
    )
