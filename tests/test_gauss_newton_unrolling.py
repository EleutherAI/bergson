"""Unrolled Gauss-Newton scores."""

import torch
import torchopt
from datasets import Dataset
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.unroll import unroll_scores
from bergson.utils.math import weighted_causal_lm_ce

MODEL = "EleutherAI/pythia-14m"
BETAS, EPS, EPS_ROOT, WD = (0.9, 0.99), 1e-8, 1e-8, 0.01


def lr(step):
    return 1e-3 * (1 + step) / 4


def _train(ckpt_dir: str):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float64, attn_implementation="eager"
    ).eval()
    model.loss_function = weighted_causal_lm_ce
    # Training only the output head makes the logits linear in the trained
    # weights, where the Gauss-Newton matrix is the loss Hessian.
    model.requires_grad_(False)
    model.get_output_embeddings().requires_grad_(True)
    optimizer = torchopt.adamw(
        lr, betas=BETAS, eps=EPS, eps_root=EPS_ROOT, weight_decay=WD
    )
    trainer, state = Trainer.initialize(model, optimizer)

    torch.manual_seed(1)
    rows = torch.randint(0, 1000, (8, 6)).tolist()
    stream = DataStream(Dataset.from_dict({"input_ids": rows, "labels": rows}), 2)
    state = trainer.train(
        state, stream, inplace=True, save_dir=ckpt_dir, save_mode="log"
    )

    queries = []
    for i in (0, 3):
        with state.activate(model) as params:
            batch = stream[i]
            del batch["example_weight"]
            grads = grad_tree(model(**batch).loss, params)
            queries.append({k: g.detach().clone() for k, g in grads.items()})
    return trainer, state, stream, queries


def test_unrolled_scores_match_magic_when_logits_are_linear(tmp_path):
    trainer, state, stream, queries = _train(str(tmp_path / "a"))
    scores = unroll_scores(
        trainer,
        str(tmp_path / "a"),
        stream,
        state,
        queries,
        lr=lr,
        betas=BETAS,
        eps=EPS,
        eps_root=EPS_ROOT,
        weight_decay=WD,
    )

    for q in range(len(queries)):
        trainer, state, stream, queries = _train(str(tmp_path / f"m{q}"))
        opt = [
            torch.zeros_like(t)
            for t in tree_iter(state.opt_state)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        stream.requires_grad = True
        bwd = trainer.backward(
            str(tmp_path / f"m{q}"),
            stream,
            BackwardState(queries[q], opt, torch.zeros_like(stream.weights)),
            state,
            inplace=True,
            save_mode="log",
        )
        exact = bwd.weight_grads.detach()
        assert exact.abs().sum() > 0
        torch.testing.assert_close(scores[:, q], exact)


def test_full_batch_is_every_rank_s_rows(tmp_path):
    _, _, stream, _ = _train(str(tmp_path / "s"))
    full = stream.full_batch(1)
    torch.testing.assert_close(full["input_ids"], stream[1]["input_ids"])
    assert full["example_weight"].shape[0] == stream.batch_size


def _sharded_worker(rank: int, root: str, port: int):
    import faulthandler
    import os

    import torch.distributed as dist

    faulthandler.dump_traceback_later(90, exit=True)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    trainer, state, stream, _ = _train(os.path.join(root, "ckpt"))
    queries = [torch.load(os.path.join(root, "query.pt"))] if rank == 0 else []
    scores = unroll_scores(
        trainer,
        os.path.join(root, "ckpt"),
        stream,
        state,
        queries,
        lr=lr,
        betas=BETAS,
        eps=EPS,
        eps_root=EPS_ROOT,
        weight_decay=WD,
    )
    if rank == 0:
        torch.save(scores[:, 0], os.path.join(root, "scores.pt"))
    dist.barrier()
    dist.destroy_process_group()


def test_fewer_queries_than_ranks(tmp_path):
    """One query on two ranks: the rank without a query replays alongside."""
    import torch.multiprocessing as mp

    trainer, state, stream, queries = _train(str(tmp_path / "ref"))
    reference = unroll_scores(
        trainer,
        str(tmp_path / "ref"),
        stream,
        state,
        queries[:1],
        lr=lr,
        betas=BETAS,
        eps=EPS,
        eps_root=EPS_ROOT,
        weight_decay=WD,
    )
    torch.save(queries[0], tmp_path / "query.pt")
    mp.spawn(_sharded_worker, args=(str(tmp_path), 29731), nprocs=2, join=True)
    torch.testing.assert_close(torch.load(tmp_path / "scores.pt"), reference[:, 0])
