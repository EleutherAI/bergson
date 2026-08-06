"""Per-query MAGIC scoring (`query_method="none"`).

`compute_per_query_magic_scores` runs one backward per query, sharing the
forward, to produce a ``[num_train_docs, num_query_docs]`` score matrix. Because
the backward is linear in the query cotangent, the mean over queries of the
per-query scores must equal the aggregate-query MAGIC score (a single backward
whose cotangent is the mean of the per-query gradients) — exactly, when the
queries have equal token counts so that the aggregate mean-over-tokens loss is
the uniform mean of the per-query losses. That identity is the correctness gate.
"""

import tempfile

import pytest
import torch
import torchopt
from datasets import Dataset
from torchopt.pytree import tree_iter

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.cli import compute_per_query_magic_scores
from bergson.magic.config import MagicConfig
from bergson.utils.math import weighted_causal_lm_ce

TINY = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _model():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(TINY)
    m = AutoModelForCausalLM.from_config(
        cfg, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.loss_function = weighted_causal_lm_ce
    m.requires_grad_(True)
    return m


def _equal_length_docs(n, seqlen=5, start=1):
    ids = [[start + i * seqlen + j for j in range(seqlen)] for i in range(n)]
    return Dataset.from_dict(
        {"input_ids": ids, "labels": ids, "attention_mask": [[1] * seqlen] * n}
    )


def _aggregate_magic_score(trainer, model, fwd_state, stream, ckpt_dir, query_ds):
    """Single backward with cotangent = mean over the query docs' gradients."""
    with fwd_state.activate(model) as params:
        qstream = DataStream(query_ds, batch_size=len(query_ds), device="cpu")
        batch = qstream[0]
        del batch["example_weight"]
        loss = model(
            **batch
        ).loss  # mean over all query tokens (equal length → uniform)
        qgrads = {k: g.detach().clone() for k, g in grad_tree(loss, params).items()}
        opt_grads = [
            torch.zeros_like(b)
            for b in tree_iter(fwd_state.opt_state)
            if isinstance(b, torch.Tensor) and b.is_floating_point()
        ]
        bwd = BackwardState(qgrads, opt_grads, torch.zeros_like(stream.weights))
    stream.requires_grad = True
    bwd = trainer.backward(
        ckpt_dir, stream, bwd, fwd_state, inplace=True, cleanup=False
    )
    return bwd.weight_grads.detach().cpu()


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_per_query_mean_reproduces_aggregate(grad_accum_steps):
    model = _model()
    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    # Multi-step training so the backward is non-trivial.
    train_ds = _equal_length_docs(4, start=1)
    stream = DataStream(train_ds, batch_size=1, device="cpu")
    assert len(stream) == 4

    query_ds = _equal_length_docs(3, start=100)  # equal token counts

    with tempfile.TemporaryDirectory() as run_path:
        ckpts = f"{run_path}/checkpoints"
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpts)

        run_cfg = MagicConfig(
            run_path=run_path,
            query_method="none",
            grad_accum_steps=grad_accum_steps,
        )
        run_cfg.query.prompt_column = "input_ids"

        # Snapshot final state so the aggregate reference starts where per-query does.
        agg = _aggregate_magic_score(trainer, model, fwd_state, stream, ckpts, query_ds)

        stream.requires_grad = True
        per_query = compute_per_query_magic_scores(
            trainer,
            ckpts,
            stream,
            fwd_state,
            model,
            query_ds,
            num_query_docs=len(query_ds),
            run_cfg=run_cfg,
            world_size=1,
            global_rank=0,
            pad_count=0,
            weight_pad_count=0,
        )

    # Shape: rows = train docs, cols = queries (validate_scores layout).
    assert per_query.shape == (len(train_ds), len(query_ds))
    assert torch.isfinite(per_query).all()
    # Queries are distinct (different cotangents → different score columns).
    assert not torch.allclose(per_query[:, 0], per_query[:, 1])
    # Correctness gate: mean over queries == aggregate-query score.
    torch.testing.assert_close(per_query.mean(dim=1), agg, atol=1e-6, rtol=1e-4)


def test_per_query_scores_saved_incrementally():
    """Each query is written to per_query/q{i}.pt as it completes (crash-safe)."""
    import os

    model = _model()
    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(_equal_length_docs(3), batch_size=1, device="cpu")
    query_ds = _equal_length_docs(2, start=100)

    with tempfile.TemporaryDirectory() as run_path:
        ckpts = f"{run_path}/checkpoints"
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpts)
        run_cfg = MagicConfig(run_path=run_path, query_method="none")
        run_cfg.query.prompt_column = "input_ids"
        stream.requires_grad = True
        compute_per_query_magic_scores(
            trainer,
            ckpts,
            stream,
            fwd_state,
            model,
            query_ds,
            num_query_docs=2,
            run_cfg=run_cfg,
            world_size=1,
            global_rank=0,
            pad_count=0,
            weight_pad_count=0,
        )
        assert os.path.exists(f"{run_path}/per_query/q0.pt")
        assert os.path.exists(f"{run_path}/per_query/q1.pt")


def test_per_query_scores_only_real_queries_when_padded():
    """A query set padded to fill a batch must yield one score column per real
    query, not per padded row (the pads are copies of the last real query)."""
    from bergson.magic.data_stream import pad_dataset_to_batch_size

    model = _model()
    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(_equal_length_docs(3), batch_size=1, device="cpu")
    query_ds = _equal_length_docs(3, start=100)

    # run_magic pads the query set to a batch_size multiple with weight-0 rows.
    padded_ds, num_query_docs, pad_count, weight_pad_count = pad_dataset_to_batch_size(
        query_ds, 4, len(query_ds), "Query", 0
    )
    assert len(padded_ds) == 4 and pad_count == 1
    assert num_query_docs - weight_pad_count == len(query_ds)

    with tempfile.TemporaryDirectory() as run_path:
        ckpts = f"{run_path}/checkpoints"
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpts)
        run_cfg = MagicConfig(run_path=run_path, query_method="none")
        run_cfg.query.prompt_column = "input_ids"
        stream.requires_grad = True
        per_query = compute_per_query_magic_scores(
            trainer,
            ckpts,
            stream,
            fwd_state,
            model,
            padded_ds,
            num_query_docs - weight_pad_count,
            run_cfg=run_cfg,
            world_size=1,
            global_rank=0,
            pad_count=0,
            weight_pad_count=0,
        )
        assert per_query.shape == (3, len(query_ds))
        import os

        assert not os.path.exists(f"{run_path}/per_query/q3.pt")


def _per_query_run(tmp_path, attribute_tokens: bool, num_docs=5, seq_len=8, n_query=2):
    """Run worker() in per-query mode; return (scores, doc_ids).

    num_docs=5 at batch_size 4 pads by 3 rows, where weight_pad_count is 1 but
    pad_count is 3 — the gap a rank-blind trim falls into.
    """
    from bergson.config.config import DataConfig
    from bergson.magic.cli import worker

    def ds(n):
        toks = [[(d * seq_len + t) % 50 + 1 for t in range(seq_len)] for d in range(n)]
        return Dataset.from_dict(
            {
                "input_ids": toks,
                "labels": toks,
                "doc_ids": [[d] * seq_len for d in range(n)],
                "length": [seq_len] * n,
            }
        )

    run_path = tmp_path / ("tok" if attribute_tokens else "doc")
    run_cfg = MagicConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-14m",
        data=DataConfig(dataset="unused", chunk_length=seq_len),
        query=DataConfig(dataset="unused", chunk_length=seq_len),
        batch_size=4,
        attribute_tokens=attribute_tokens,
        query_method="none",
        skip_validation=True,
    )
    worker(0, 0, 1, ds(num_docs), ds(n_query), num_docs, n_query, run_cfg)

    doc_ids = run_path / "doc_ids.pt"
    return (
        torch.load(run_path / "scores.pt"),
        torch.load(doc_ids) if doc_ids.exists() else None,
    )


def test_per_query_per_token_aggregates_to_per_doc(tmp_path):
    """Per-token per-query scores are [docs, seq_len, queries], and summing
    them over each document's tokens reproduces the per-doc per-query run."""
    per_tok, doc_ids = _per_query_run(tmp_path, attribute_tokens=True)
    per_doc, _ = _per_query_run(tmp_path, attribute_tokens=False)
    num_docs, n_query = per_doc.shape

    assert per_tok.shape == (num_docs, 8, n_query), f"got {tuple(per_tok.shape)}"
    assert doc_ids is not None, "per-token run must write doc_ids.pt"
    assert doc_ids.shape == per_tok.shape[:2]

    agg = torch.zeros(num_docs, n_query, dtype=torch.float64)
    agg.scatter_add_(
        0,
        doc_ids.reshape(-1, 1).expand(-1, n_query),
        per_tok.reshape(-1, n_query).to(torch.float64),
    )

    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


def test_three_dim_scores_load_as_per_token_multi_query(tmp_path):
    """A 3-D scores.pt is unambiguous: neither classifier needs the run config."""
    from bergson.magic.cli import scores_are_per_token
    from bergson.validate import load_attribution_scores

    path = tmp_path / "scores.pt"
    torch.save(torch.randn(4, 6, 3), path)

    assert scores_are_per_token(str(path))
    _, multi_query = load_attribution_scores(str(path))
    assert multi_query
