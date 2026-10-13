"""Unrolled Gauss-Newton scores."""

from typing import Any

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


def _kwargs(**extra: Any) -> dict[str, Any]:
    return dict(
        lr=lr, betas=BETAS, eps=EPS, eps_root=EPS_ROOT, weight_decay=WD, **extra
    )


def test_gradient_outer_matches_per_document_outer_products(tmp_path):
    """One step with ``gradient_outer`` applies the mean outer product of the
    documents' mean-token loss gradients."""
    from bergson.magic.unroll import _gauss_newton_and_scores, unrolled_names

    trainer, state, stream, queries = _train(str(tmp_path / "g"))
    model = trainer.model
    params = {k: v.detach() for k, v in state.params.items()}
    names = unrolled_names(model, params)
    batch = stream.full_batch(0)
    torch.manual_seed(3)
    z = [torch.randn(1, *params[k].shape, dtype=torch.float64) for k in names]
    scores = torch.zeros(stream.weights.shape[0], 1, dtype=torch.float64)
    out = _gauss_newton_and_scores(
        model,
        params,
        state.buffers,
        batch,
        names,
        z,
        stream.weights,
        scores,
        micro_batch_size=1,
        bf16=False,
        curvature="gradient_outer",
    )

    expected = [torch.zeros_like(v[0]) for v in z]
    docs = batch["input_ids"]
    for d in range(len(docs)):
        with state.activate(model) as live:
            logits = model(input_ids=docs[d : d + 1]).logits[:, :-1]
            labels = batch["labels"][d : d + 1, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
            grads = torch.autograd.grad(loss, [live[k] for k in names])
        inner = sum((g * v[0]).sum() for g, v in zip(grads, z))
        for e, g in zip(expected, grads):
            e += g * inner / len(docs)
    for o, e in zip(out, expected):
        torch.testing.assert_close(o[0], e)


def test_all_parameters_track_magic_more_closely(tmp_path):
    """On a tied tiny GPT-2 trained end to end, unrolling every parameter lands
    closer to MAGIC's exact backward than unrolling the linear weights."""
    from transformers import GPT2Config, GPT2LMHeadModel

    def train(ckpt_dir):
        torch.manual_seed(0)
        config = GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            vocab_size=64,
            n_positions=8,
            bos_token_id=0,
            eos_token_id=0,
            attn_implementation="eager",
        )
        model = GPT2LMHeadModel(config).double().eval()
        model.loss_function = weighted_causal_lm_ce
        optimizer = torchopt.adamw(
            lr, betas=BETAS, eps=EPS, eps_root=EPS_ROOT, weight_decay=WD
        )
        trainer, state = Trainer.initialize(model, optimizer)
        torch.manual_seed(1)
        rows = torch.randint(0, 64, (8, 6)).tolist()
        stream = DataStream(Dataset.from_dict({"input_ids": rows, "labels": rows}), 2)
        state = trainer.train(
            state, stream, inplace=True, save_dir=ckpt_dir, save_mode="log"
        )
        with state.activate(model) as params:
            batch = stream[0]
            del batch["example_weight"]
            grads = grad_tree(model(**batch).loss, params)
            query = {k: g.detach().clone() for k, g in grads.items()}
        return trainer, state, stream, query

    trainer, state, stream, query = train(str(tmp_path / "m"))
    opt = [
        torch.zeros_like(t)
        for t in tree_iter(state.opt_state)
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    ]
    stream.requires_grad = True
    exact = trainer.backward(
        str(tmp_path / "m"),
        stream,
        BackwardState(query, opt, torch.zeros_like(stream.weights)),
        state,
        inplace=True,
        save_mode="log",
    ).weight_grads.detach()

    errors = {}
    for parameters in ("linear", "all"):
        trainer, state, stream, query = train(str(tmp_path / parameters))
        scores = unroll_scores(
            trainer,
            str(tmp_path / parameters),
            stream,
            state,
            [query],
            **_kwargs(parameters=parameters),
        )[:, 0]
        errors[parameters] = ((scores - exact).norm() / exact.norm()).item()
    assert errors["all"] < errors["linear"]
    assert errors["all"] < 0.1


def test_fisher_token_matches_per_token_outer_products(tmp_path):
    """One step with ``fisher_token`` applies the mean outer product of the
    tokens' loss gradients."""
    from bergson.magic.unroll import _gauss_newton_and_scores, unrolled_names

    trainer, state, stream, _ = _train(str(tmp_path / "f"))
    model = trainer.model
    params = {k: v.detach() for k, v in state.params.items()}
    names = unrolled_names(model, params)
    batch = stream.full_batch(0)
    torch.manual_seed(4)
    z = [torch.randn(1, *params[k].shape, dtype=torch.float64) for k in names]
    scores = torch.zeros(stream.weights.shape[0], 1, dtype=torch.float64)
    out = _gauss_newton_and_scores(
        model,
        params,
        state.buffers,
        batch,
        names,
        z,
        stream.weights,
        scores,
        micro_batch_size=1,
        bf16=False,
        curvature="fisher_token",
    )

    expected = [torch.zeros_like(v[0]) for v in z]
    docs, labels = batch["input_ids"], batch["labels"]
    tokens = (labels[:, 1:] != -100).sum().item()
    for d in range(len(docs)):
        for t in range(labels.shape[1] - 1):
            with state.activate(model) as live:
                logits = model(input_ids=docs[d : d + 1]).logits[0, t]
                loss = torch.nn.functional.cross_entropy(
                    logits[None], labels[d : d + 1, t + 1]
                )
                grads = torch.autograd.grad(loss, [live[k] for k in names])
            inner = sum((g * v[0]).sum() for g, v in zip(grads, z))
            for e, g in zip(expected, grads):
                e += g * inner / tokens
    for o, e in zip(out, expected):
        torch.testing.assert_close(o[0], e)
