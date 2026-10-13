"""Check that ``unroll_scores`` with ``curvature="gradient_outer"`` is the adjoint
of the AdamW-influence sweep of dattri-llm (commit 6a6930d) on a toy run.

Run with ``PYTHONPATH=<bergson checkout>:<dattri-llm checkout>``. The toy trains a
one-layer GPT-2 for four AdamW steps; the authors' ``adamw_influence_push``,
``adamw_influence_coupling`` and ``adamw_influence_transition`` carry the
summary matrix ``W`` down the steps with ``fisher_scale = |B|`` for the mean
reduction, and ``unroll_scores`` runs the same trajectory from checkpoints.
"""

import tempfile
import warnings

import torch
import torchopt
from datasets import Dataset
from dattri_llm.gradient.ops import optimizer as ops
from transformers import GPT2Config, GPT2LMHeadModel

from bergson.distributed import grad_tree
from bergson.magic import DataStream, Trainer
from bergson.magic.unroll import adam_moments, unroll_scores, unrolled_names
from bergson.utils.math import weighted_causal_lm_ce

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
BETAS, EPS, WEIGHT_DECAY, BATCH = (0.9, 0.99), 1e-8, 0.01, 2


def lr(step: int) -> float:
    return 1e-3 * (1 + step) / 4


def build():
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
        tie_word_embeddings=False,
    )
    model = GPT2LMHeadModel(config).double().eval()
    model.loss_function = weighted_causal_lm_ce
    optimizer = torchopt.adamw(
        lr, betas=BETAS, eps=EPS, eps_root=0.0, weight_decay=WEIGHT_DECAY
    )
    trainer, state = Trainer.initialize(model, optimizer)
    torch.manual_seed(1)
    rows = torch.randint(0, 64, (8, 6)).tolist()
    stream = DataStream(Dataset.from_dict({"input_ids": rows, "labels": rows}), BATCH)
    return trainer, state, stream, model


def flat(tensors: dict, names: list[str]) -> torch.Tensor:
    return torch.cat([tensors[k].flatten() for k in names])


def query_gradient(state, model, stream, names):
    with state.activate(model) as live:
        batch = stream[0]
        del batch["example_weight"]
        grads = grad_tree(model(**batch).loss, live)
        return {k: g.detach().clone() for k, g in grads.items()}, flat(grads, names)


def main() -> None:
    root = tempfile.mkdtemp()

    # Train step by step, recording what the authors' sweep reads at each step:
    # the per-sample shares of the batch gradient and the moments before and after.
    trainer, state, stream, model = build()
    names = unrolled_names(model, state.params, "linear")
    steps = len(stream)
    record = []
    for t in range(steps):
        with state.activate(model) as live:
            batch = stream.full_batch(t)
            shares = []
            for i in range(BATCH):
                logits = model(input_ids=batch["input_ids"][i : i + 1]).logits[:, :-1]
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), batch["labels"][i : i + 1, 1:].flatten()
                )
                grads = torch.autograd.grad(loss, [live[k] for k in names])
                shares.append(torch.cat([g.flatten() for g in grads]) / BATCH)
        mu, _ = adam_moments(state, names)
        m_pre = flat(mu, names).clone()
        state = trainer.step(state, stream[t], inplace=True)
        mu, nu = adam_moments(state, names)
        record.append(
            dict(
                g_z=torch.stack(shares),
                m_pre=m_pre,
                m_post=flat(mu, names).clone(),
                v_post=flat(nu, names).clone(),
                lr=lr(t),
                step=t + 1,
            )
        )
    _, query = query_gradient(state, model, stream, names)
    query = query.float()

    # The authors' train-side sweep with the summary matrix W = [W_theta | W_m | W_v]
    p = query.numel()
    w_theta = torch.eye(p, dtype=torch.float32)
    w_m = torch.zeros(p, p, dtype=torch.float32)
    w_v = torch.zeros(p, p, dtype=torch.float32)
    theirs = torch.zeros(steps, BATCH)
    for t in reversed(range(steps)):
        r = record[t]
        g_t = ((r["m_post"] - BETAS[0] * r["m_pre"]) / (1 - BETAS[0])).float()
        d, s = ops.adam_preconditioner(
            r["m_post"] / (1 - BETAS[0] ** r["step"]),
            r["v_post"] / (1 - BETAS[1] ** r["step"]),
            eps=EPS,
        )
        common = dict(lr=r["lr"], step=r["step"], beta1=BETAS[0], beta2=BETAS[1])
        g_z = r["g_z"].float()
        push = ops.adamw_influence_push(g_z, g_t, d, s, **common)
        carried = (
            push[:, :p] @ w_theta.T
            + push[:, p : 2 * p] @ w_m.T
            + push[:, 2 * p :] @ w_v.T
        )
        theirs[t] = -(carried @ query).double()
        response = ops.adamw_influence_coupling(
            w_theta, w_m, w_v, g_z, g_t, d, s, **common
        )
        w_theta, w_m, w_v = ops.adamw_influence_transition(
            w_theta, w_m, w_v, d, s, weight_decay=WEIGHT_DECAY, **common
        )
        w_theta = w_theta + float(BATCH) * (response.T @ g_z)

    # The same trajectory through bergson, replayed from per-step checkpoints
    trainer, state, stream, model = build()
    state = trainer.train(
        state,
        stream,
        inplace=True,
        save_dir=root,
        save_mode="interval",
        save_interval=1,
    )
    query_dict, _ = query_gradient(state, model, stream, names)
    ours = unroll_scores(
        trainer,
        root,
        stream,
        state,
        [query_dict],
        lr=lr,
        betas=BETAS,
        eps=EPS,
        eps_root=0.0,
        weight_decay=WEIGHT_DECAY,
        curvature="gradient_outer",
        parameters="linear",
    )[:, 0].reshape(steps, BATCH)

    print("theirs", [f"{x:.6e}" for x in theirs.flatten().tolist()])
    print("ours  ", [f"{x:.6e}" for x in ours.flatten().tolist()])
    error = (ours - theirs).abs().max() / theirs.abs().max()
    print(f"max relative error {error.item():.3e}")


if __name__ == "__main__":
    main()
