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
from bergson.magic.cli import compute_query_gradients
from bergson.magic.data_stream import pad_dataset_to_batch_size
from bergson.utils.math import weighted_causal_lm_ce
from bergson.validate import mean_query_loss, per_doc_query_losses


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


def _run_magic(model, dataset, device="cpu", ckpt_dir=None, batch_size=None, lr=1e-4):
    """Run full MAGIC pipeline and return attribution scores."""
    optimizer = torchopt.adamw(lr, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    if batch_size is None:
        batch_size = len(dataset)
    stream = DataStream(dataset, batch_size=batch_size, device=device)
    assert len(stream) >= 1

    _tmpdir = tempfile.TemporaryDirectory() if ckpt_dir is None else None
    if _tmpdir is not None:
        ckpt_dir = _tmpdir.name
    assert ckpt_dir is not None

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


def _make_multistep_dataset(num_rows=16, seq_len=8):
    """Deterministic dataset with several batches worth of rows.

    All rows have equal length so that the mean loss over a batch equals the
    mean of the per-rank microbatch means, keeping single- and multi-process
    forward trajectories identical.
    """
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 100, (num_rows, seq_len), generator=g).tolist()
    return Dataset.from_dict(
        {
            "input_ids": ids,
            "labels": ids,
            "attention_mask": [[1] * seq_len] * num_rows,
        }
    )


def _cpu_ddp_worker(
    rank, world_size, port, dataset, batch_size, lr, result_dict, ckpt_dir
):
    """Gloo/CPU worker for the multi-step DDP test."""
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        model = _make_model()
        scores = _run_magic(
            model, dataset, ckpt_dir=ckpt_dir, batch_size=batch_size, lr=lr
        )
        result_dict[rank] = scores
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_ddp_matches_single_process_multistep():
    """Multi-step DDP MAGIC scores should match single-process scores.

    With more than one training step, the backward pass propagates the
    query gradient through earlier steps via each batch's gradient
    all-reduce. That collective must be autograd-aware: if the gradient
    passes through it rank-locally (as with the non-differentiable
    functional collectives), every rank drops the other ranks'
    curvature terms and the scores silently diverge from the exact
    single-process metagradient. A single-step run (the GPU test below)
    cannot catch this, because the last step's backward happens to be
    exact without any cross-rank exchange.

    The lr is large so the curvature terms are big enough that dropping
    the cross-rank ones moves scores well past the tolerance (~2% at
    lr=5e-2 vs ~0.03% at lr=1e-3 for this model).
    """
    batch_size = 4
    lr = 5e-2
    dataset = _make_multistep_dataset(num_rows=16)

    model = _make_model()
    expected = _run_magic(model, dataset, batch_size=batch_size, lr=lr)
    del model

    world_size = 2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    result_dict = manager.dict()

    with tempfile.TemporaryDirectory() as shared_ckpt_dir:
        mp.spawn(
            _cpu_ddp_worker,
            args=(
                world_size,
                port,
                dataset,
                batch_size,
                lr,
                result_dict,
                shared_ckpt_dir,
            ),
            nprocs=world_size,
            join=True,
        )

    actual = result_dict[0]

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-4,
        rtol=1e-3,
        msg="Multi-step DDP attribution scores diverged from single-process",
    )


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


def _padded_query_eval(device):
    """Query loss, per-document losses and mean query gradient of a 3-document
    query set padded to 4 rows, so on two ranks one rank's shard is half pad."""
    model = _make_model().to(device)
    _, fwd_state = Trainer.initialize(model, torchopt.adamw(1e-4))
    model.eval()
    docs = _make_dataset().select(range(3))
    padded, n_docs, _, weight_pad = pad_dataset_to_batch_size(docs, 4, 3, "Q", 0)
    stream = DataStream(padded, 4, device=device, weight_shape=(n_docs,))
    stream.weights.data[-weight_pad:] = 0.0
    grads, loss = compute_query_gradients(fwd_state, model, stream)
    with fwd_state.activate(model):
        per_doc = per_doc_query_losses(model, stream, n_docs)[:3]
        mean = mean_query_loss(model, stream)
    return {k: g.cpu() for k, g in grads.items()}, loss, per_doc.cpu(), mean.cpu()


def _padded_query_worker(rank, world_size, port, result_dict):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}"),
        )
        result_dict[rank] = _padded_query_eval(f"cuda:{rank}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need >= 2 GPUs for DDP test",
)
def test_padded_query_eval_matches_single_process():
    """Every rank returns the single-process query loss, per-document losses and
    mean query gradient, even when part of its shard is zero-weight padding."""
    expected = _padded_query_eval("cuda:0")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    result_dict = mp.Manager().dict()
    mp.spawn(_padded_query_worker, args=(2, port, result_dict), nprocs=2, join=True)

    for rank in range(2):
        grads, loss, per_doc, mean = result_dict[rank]
        assert loss == pytest.approx(expected[1], rel=1e-5)
        torch.testing.assert_close(per_doc, expected[2])
        torch.testing.assert_close(mean, expected[3])
        for name, g in expected[0].items():
            torch.testing.assert_close(grads[name], g, atol=1e-6, rtol=1e-4)
