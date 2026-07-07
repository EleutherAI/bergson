"""MAGIC integration test: forward + backward through 2 training steps."""

import os
import tempfile

import pytest
import torch
import torchopt
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

MODEL_CONFIGS = [
    "trl-internal-testing/tiny-Phi3ForCausalLM",
    "EleutherAI/pythia-14m",
]


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_two_steps(model_name, dataset):
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )

    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    train_stream = DataStream(
        dataset,
        batch_size=len(dataset),
        device=device,
    )
    assert len(train_stream) == 1

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=True,
            save_dir=ckpt_dir,
        )

        # Compute query gradients on the training batch
        with fwd_state.activate(model) as params:
            batch = train_stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }

            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads,
                opt_grads,
                torch.zeros_like(train_stream.weights),
            )

        # Backward pass through training
        train_stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            train_stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
        )

    scores = bwd_state.weight_grads.detach().cpu()
    assert scores.shape == (len(dataset),)
    assert scores.abs().sum() > 0, "Attribution scores are all zero"


def _train_and_query_loss(
    model_name,
    dataset,
    batch_size,
    *,
    per_token: bool,
    zero_subset: torch.Tensor | None = None,
    shuffle_seed: int | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[float, torch.Tensor | None]:
    """Mirror the validation loop's train + query pass with a fixed dropout subset.

    Mirrors ``run_magic`` (shuffle → pad → train) and worker()'s save logic:
    trains from a fresh init with ``stream.weights = 1``, applies
    ``stream.weights.view(-1)[zero_subset] = 0`` (the same line cli.py runs
    inside the validation loop), then averages model loss over the dataset.

    Returns ``(query_loss, trimmed_doc_ids)`` where ``trimmed_doc_ids`` is the
    post-shuffle, post-pad-trim tensor that worker() saves to ``doc_ids.pt``
    (None for per-doc runs). Tests use it to map a chosen doc set to flat
    indices into ``stream.weights.view(-1)`` — same lookup downstream
    consumers do against the saved file.
    """
    from bergson.magic.cli import attach_doc_ids_if_missing, pad_dataset_to_batch_size

    ds = attach_doc_ids_if_missing(dataset)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    num_docs = max(max(row) for row in ds["doc_ids"]) + 1

    padded_ds, num_docs_pad, pad_count, weight_pad_count = pad_dataset_to_batch_size(
        ds, batch_size, num_docs, "Test", 0
    )

    if per_token:
        T = max(len(row) for row in padded_ds["input_ids"])
        weight_shape = (len(padded_ds), T)
    else:
        weight_shape = (num_docs_pad,)

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(
        padded_ds, batch_size=batch_size, device=device, weight_shape=weight_shape
    )

    if pad_count:
        if stream.weights.ndim == 1:
            stream.weights.data[-weight_pad_count:] = 0.0
        else:
            stream.weights.data[-pad_count:] = 0.0

    if zero_subset is not None:
        stream.weights.data.view(-1)[zero_subset] = 0.0

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpt_dir)
        with fwd_state.activate(model), torch.no_grad():
            total = 0.0
            for batch in stream:
                del batch["example_weight"]
                total += model(**batch).loss.item()
        loss = total / len(stream)

    if per_token:
        trimmed = torch.tensor(padded_ds["doc_ids"])
        if pad_count:
            trimmed = trimmed[:-pad_count]
        return loss, trimmed
    return loss, None


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_validation_loop_doc_token_dropout_equiv(model_name):
    """The per-token validation loop's flat-index dropout is operationally
    equivalent to per-doc dropout: for a chosen set of docs ``D``, zeroing
    ``stream.weights.view(-1)[flat]`` in per-token mode — where ``flat`` comes
    from ``torch.isin(saved_doc_ids.flatten(), D)``, exactly the lookup a
    consumer of ``doc_ids.pt`` would do — yields the same post-training query
    loss as zeroing ``stream.weights[D]`` in per-doc mode.

    Exercises the parts of cli.py worker() that ``doc_ids.pt`` exists for:
    (a) shuffle reorders rows so the saved tensor differs from the input ds;
    (b) ``len(ds) % batch_size != 0`` forces ``pad_dataset_to_batch_size`` to
    append a synthetic-doc pad row that worker() then strips with
    ``doc_ids[:-pad_count]`` before saving; (c) one document spans rows so
    the lookup is non-trivial. If shuffle/pad-trim alignment or row-major
    flatten order ever drifts, this test breaks before any real run does.
    """
    from datasets import Dataset

    # 5 docs across 3 chunks (forces pad with batch_size=2); doc 2 spans rows.
    ds = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ],
            "labels": [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ],
            "attention_mask": [[1] * 6] * 3,
            "doc_ids": [
                [0, 0, 1, 1, 1, 2],
                [2, 2, 2, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
            ],
        }
    )
    batch_size = 2
    shuffle_seed = 7

    # First run: extract the post-shuffle, post-pad-trim doc_ids (= what
    # worker() would write to doc_ids.pt) and a baseline loss.
    loss_full, saved_doc_ids = _train_and_query_loss(
        model_name,
        ds,
        batch_size=batch_size,
        per_token=True,
        shuffle_seed=shuffle_seed,
        zero_subset=None,
    )
    assert saved_doc_ids is not None

    # Confirm shuffle actually changed the row order — otherwise the test
    # silently degenerates to "saved doc_ids == input doc_ids".
    input_doc_ids = torch.tensor(ds["doc_ids"])
    assert not torch.equal(
        saved_doc_ids, input_doc_ids
    ), "shuffle had no effect on doc_ids; test is degenerate"
    # Pad-trim actually fired (3 % 2 = 1 row of pad was stripped).
    assert saved_doc_ids.shape == input_doc_ids.shape

    docs_to_drop = torch.tensor([1, 2])
    flat_drop = (
        torch.isin(saved_doc_ids.reshape(-1), docs_to_drop).nonzero().squeeze(-1)
    )
    assert flat_drop.numel() > 0, "no tokens matched docs_to_drop"

    loss_doc, _ = _train_and_query_loss(
        model_name,
        ds,
        batch_size=batch_size,
        per_token=False,
        shuffle_seed=shuffle_seed,
        zero_subset=docs_to_drop,
    )
    loss_tok, _ = _train_and_query_loss(
        model_name,
        ds,
        batch_size=batch_size,
        per_token=True,
        shuffle_seed=shuffle_seed,
        zero_subset=flat_drop,
    )

    assert abs(loss_doc - loss_full) > 1e-6, "dropout had no effect; test is degenerate"
    torch.testing.assert_close(
        torch.tensor(loss_tok), torch.tensor(loss_doc), atol=1e-5, rtol=1e-4
    )


def _run_magic_cli(
    model_name,
    dataset,
    batch_size,
    *,
    per_token: bool,
    shuffle_seed: int | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mirror run_magic + worker end-to-end: attach doc_ids if missing,
    optionally shuffle, pad to batch_size, train, backward, and trim scores
    + doc_ids the way worker() saves them to disk.

    Returns (scores, doc_ids). doc_ids is None for per-doc runs (scores are
    already indexed by doc id and need no auxiliary lookup).
    """
    from bergson.magic.cli import attach_doc_ids_if_missing, pad_dataset_to_batch_size

    ds = attach_doc_ids_if_missing(dataset)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    num_docs = max(max(row) for row in ds["doc_ids"]) + 1

    padded_ds, num_docs_pad, pad_count, weight_pad_count = pad_dataset_to_batch_size(
        ds, batch_size, num_docs, "Test", 0
    )

    if per_token:
        T = max(len(row) for row in padded_ds["input_ids"])
        weight_shape = (len(padded_ds), T)
    else:
        weight_shape = (num_docs_pad,)

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(
        padded_ds, batch_size=batch_size, device=device, weight_shape=weight_shape
    )

    if pad_count:
        if stream.weights.ndim == 1:
            stream.weights.data[-weight_pad_count:] = 0.0
        else:
            stream.weights.data[-pad_count:] = 0.0

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(fwd_state, stream, inplace=True, save_dir=ckpt_dir)
        with fwd_state.activate(model) as params:
            batch = stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }
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
            ckpt_dir, stream, bwd_state, fwd_state, inplace=True, cleanup=True
        )

    scores = bwd_state.weight_grads.detach().cpu()
    doc_ids = torch.tensor(padded_ds["doc_ids"]) if scores.ndim == 2 else None

    if pad_count:
        if scores.ndim == 1:
            scores = scores[:-weight_pad_count]
        else:
            scores = scores[:-pad_count]
            assert doc_ids is not None
            doc_ids = doc_ids[:-pad_count]

    return scores, doc_ids


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_scores_zero_at_masked_labels(model_name):
    """MAGIC scores are exactly zero at positions whose weight has no loss path.

    Two sources of zero-by-construction in weighted_causal_lm_ce:
    - shifted labels == -100: F.cross_entropy with ignore_index=-100 makes
      tok_loss[t] == 0, so w[:, t] enters the loss multiplied by zero.
    - the last-token weight slot: example_weight is sliced as [:, :-1] before
      multiplication, so column T-1 never enters the loss.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "labels": [[1, 2, -100, 4, 5], [-100, 7, 8, -100, 10]],
            "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        }
    )
    N, T = len(ds), 5

    per_tok, _ = _run_magic_cli(model_name, ds, len(ds), per_token=True)
    assert per_tok.shape == (N, T)

    labels = torch.tensor(ds["labels"])
    zero_mask = torch.zeros(N, T, dtype=torch.bool)
    zero_mask[:, T - 1] = True  # unused last-token slot
    zero_mask[:, :-1] = labels[:, 1:] == -100  # shifted masked positions

    assert torch.all(per_tok[zero_mask] == 0), (
        f"Expected zero MAGIC scores at masked/unused positions; "
        f"got max |score| = {per_tok[zero_mask].abs().max():.3e}"
    )
    assert (
        per_tok[~zero_mask].abs().sum() > 0
    ), "All non-masked positions are zero — test is degenerate"


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_sums_to_per_doc(model_name, dataset):
    """Per-token MAGIC scores summed over tokens equal per-doc MAGIC scores.

    MAGIC computes d(query_loss)/dw through the training trajectory. With
    weighted_causal_lm_ce, the training loss is
        per-doc:   sum_{i,t} w_i     * tok_loss[i,t] / denom
        per-token: sum_{i,t} w_{i,t} * tok_loss[i,t] / denom
    Both evaluate to the same value at initialization (all weights = 1), so the
    two runs share an identical training trajectory. By linearity of the MAGIC
    backward pass, dQ/dw_i = sum_t dQ/dw_{i,t}.
    """
    N = len(dataset)
    T = len(dataset[0]["input_ids"])

    per_doc, _ = _run_magic_cli(model_name, dataset, N, per_token=False)
    per_tok, _ = _run_magic_cli(model_name, dataset, N, per_token=True)

    assert per_doc.shape == (N,)
    assert per_tok.shape == (N, T)

    torch.testing.assert_close(per_tok.sum(dim=-1), per_doc, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_sums_to_per_doc_packed(model_name):
    """Per-doc MAGIC (1D weights via doc_ids lookup) equals per-token MAGIC
    scatter-summed by doc_ids, with document packing across chunks.

    Exercises the non-trivial path used by the empirical per-token/per-doc
    comparison: chunks contain multiple documents, one document spans two
    chunks, and the per-doc weight is shared across all positions of that
    doc. Mirrors the scatter_add(doc_ids) aggregation in
    scripts/correlate_pertoken_vs_docrun.py.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            "labels": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            "attention_mask": [[1] * 6, [1] * 6],
            # Packed: 4 unique docs across 2 chunks; doc 2 spans both chunks.
            "doc_ids": [[0, 0, 1, 1, 1, 2], [2, 2, 2, 3, 3, 3]],
        }
    )
    N, T, num_docs = len(ds), 6, 4

    per_doc, _ = _run_magic_cli(model_name, ds, N, per_token=False)
    per_tok, _ = _run_magic_cli(model_name, ds, N, per_token=True)

    assert per_doc.shape == (num_docs,)
    assert per_tok.shape == (N, T)

    flat_doc_ids = torch.tensor(ds["doc_ids"]).reshape(-1)
    agg = torch.zeros(num_docs, dtype=torch.float64)
    agg.scatter_add_(0, flat_doc_ids, per_tok.reshape(-1).to(torch.float64))

    # Every doc should receive at least one nonzero token contribution.
    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_sums_to_per_doc_with_padding(model_name):
    """Per-token MAGIC scores scatter-summed by doc_ids equal per-doc MAGIC
    scores even when the chunked dataset isn't divisible by batch_size —
    exercising the pad_dataset_to_batch_size path plus worker()'s pad-zero
    writes.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ],
            "labels": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ],
            "attention_mask": [[1] * 5] * 3,
            # 3 chunks, each a distinct doc; 3 % batch_size(=2) == 1 → pad 1
            "doc_ids": [[0] * 5, [1] * 5, [2] * 5],
        }
    )
    num_real_docs = 3
    T = 5
    batch_size = 2

    per_doc, _ = _run_magic_cli(model_name, ds, batch_size, per_token=False)
    per_tok, doc_ids = _run_magic_cli(model_name, ds, batch_size, per_token=True)

    assert per_doc.shape == (num_real_docs,), f"per_doc shape {per_doc.shape}"
    assert per_tok.shape == (num_real_docs, T), f"per_tok shape {per_tok.shape}"
    assert doc_ids is not None

    agg = torch.zeros(num_real_docs, dtype=torch.float64)
    agg.scatter_add_(0, doc_ids.reshape(-1), per_tok.reshape(-1).to(torch.float64))

    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


def test_attach_doc_ids_if_missing():
    """attach_doc_ids_if_missing adds [row_idx] * max_len per row when
    doc_ids is absent, and is a no-op when it's already present.
    """
    from datasets import Dataset

    from bergson.magic.cli import attach_doc_ids_if_missing

    unpacked = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "labels": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "length": [3, 2, 4],
        }
    )
    out = attach_doc_ids_if_missing(unpacked)
    assert out["doc_ids"] == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]

    packed = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4]],
            "labels": [[1, 2], [3, 4]],
            "length": [2, 2],
            "doc_ids": [[0, 0], [1, 1]],
        }
    )
    out = attach_doc_ids_if_missing(packed)
    assert out["doc_ids"] == [[0, 0], [1, 1]]  # unchanged


def test_datastream_truncates_doc_ids_for_short_batch():
    """DataStream's 1D-weights path truncates doc_ids to the per-batch padded
    seq_len. Exercises the new `indices[:, :x.shape[1]]` clamp: doc_ids is
    width-6 but a batch with only short rows pads input_ids to width 4.
    """
    from datasets import Dataset

    max_len = 6
    ds = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15],
            ],
            "labels": [
                [1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15],
            ],
            "doc_ids": [[i] * max_len for i in range(4)],
        }
    )

    stream = DataStream(ds, batch_size=2, device="cpu", weight_shape=(4,))
    for batch in stream:
        T = batch["input_ids"].shape[1]
        assert batch["example_weight"].shape == (2, T), (
            f"example_weight shape {batch['example_weight'].shape} "
            f"should match input_ids width {T}"
        )


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_unpacked_cli_aggregation(model_name):
    """End-to-end for chunk_length=0 (unpacked): inject doc_ids, shuffle, pad,
    run per-token and per-doc. scatter_add(per_tok, doc_ids) equals per_doc,
    and both index by ORIGINAL doc id (invariant over the shuffle).
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14]],
            "labels": [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14]],
        }
    )
    num_docs = 4
    batch_size = 2
    shuffle_seed = 7

    per_tok, doc_ids = _run_magic_cli(
        model_name, ds, batch_size, per_token=True, shuffle_seed=shuffle_seed
    )
    per_doc, _ = _run_magic_cli(
        model_name, ds, batch_size, per_token=False, shuffle_seed=shuffle_seed
    )

    assert per_doc.shape == (num_docs,), f"per_doc shape {per_doc.shape}"
    assert doc_ids is not None
    assert (
        doc_ids.shape == per_tok.shape
    ), f"doc_ids shape {doc_ids.shape} != per_tok shape {per_tok.shape}"

    agg = torch.zeros(num_docs, dtype=torch.float64)
    agg.scatter_add_(0, doc_ids.reshape(-1), per_tok.reshape(-1).to(torch.float64))

    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_packed_cli_aggregation_with_shuffle(model_name):
    """End-to-end for chunk_length>0 (packed) WITH shuffle: scatter_add by
    saved doc_ids recovers per-doc scores. Guards the shuffle → doc_ids →
    score alignment that the original aggregation-script bug broke.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            "labels": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            # 4 docs packed across 3 chunks; doc 1 spans chunks 0 and 1.
            "doc_ids": [[0, 0, 1, 1], [1, 2, 2, 2], [3, 3, 3, 3]],
        }
    )
    num_docs = 4
    batch_size = 2
    shuffle_seed = 7

    per_tok, doc_ids = _run_magic_cli(
        model_name, ds, batch_size, per_token=True, shuffle_seed=shuffle_seed
    )
    per_doc, _ = _run_magic_cli(
        model_name, ds, batch_size, per_token=False, shuffle_seed=shuffle_seed
    )

    assert per_doc.shape == (num_docs,)
    assert doc_ids is not None
    assert doc_ids.shape == per_tok.shape

    agg = torch.zeros(num_docs, dtype=torch.float64)
    agg.scatter_add_(0, doc_ids.reshape(-1), per_tok.reshape(-1).to(torch.float64))

    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


def test_magic_resume(dataset):
    """Resume from a checkpoint mid-training and verify identical final state."""
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    # batch_size=1 gives us 2 batches so resume has something to skip
    train_stream = DataStream(dataset, batch_size=1, device=device)
    assert len(train_stream) == 2

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # Full training run (inplace=False to keep fwd_state intact)
        final_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
        )

        # Resume from checkpoints with the same initial state
        resumed_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
            resume=True,
        )

        for k in final_state.params:
            torch.testing.assert_close(resumed_state.params[k], final_state.params[k])


def _backward_once(
    model_name,
    ds,
    ckpt_dir,
    *,
    save_mode,
    resume,
    save_every,
    fwd_resume,
    device="cpu",
    seed=42,
):
    """One end-to-end MAGIC pass through `ckpt_dir`, mirroring worker():
    (re)build the model, train forward (optionally resuming), compute query
    grads at the end-of-forward state, then run the backward pass (optionally
    resuming). Returns the final weight-grad scores.
    """
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(ds, batch_size=1, device=device)

    fwd_state = trainer.train(
        fwd_state,
        stream,
        inplace=True,
        save_dir=ckpt_dir,
        save_mode=save_mode,
        resume=fwd_resume,
    )

    with fwd_state.activate(model) as params:
        batch = stream[0]
        del batch["example_weight"]
        loss = model(**batch).loss
        query_grads = {
            k: g.detach().clone() for k, g in grad_tree(loss, params).items()
        }
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
        ckpt_dir,
        stream,
        bwd_state,
        fwd_state,
        inplace=True,
        cleanup=True,
        resume=resume,
        save_every=save_every,
        save_mode=save_mode,
    )
    return bwd_state.weight_grads.detach().cpu().clone()


@pytest.mark.parametrize("save_mode", ["all", "sqrt"])
def test_magic_backward_resume(save_mode):
    """Crash mid-backward, resume, and verify identical final scores.

    Reproduces the production crash/restart flow: forward checkpoints are
    cleaned up as the backward pass consumes them, the run dies partway, and a
    fresh process retrains the forward pass (resuming from the preserved last
    checkpoint) before resuming the backward pass from ``backward_rank0.pt``.
    """
    from datasets import Dataset

    model_name = "trl-internal-testing/tiny-Phi3ForCausalLM"

    # 6 rows, batch_size=1 -> 6 training steps, enough to interrupt partway.
    ds = Dataset.from_dict(
        {
            "input_ids": [[i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(6)],
            "labels": [[i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(6)],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in range(6)],
        }
    )

    # Reference: uninterrupted backward in its own checkpoint dir.
    with tempfile.TemporaryDirectory() as ref_dir:
        ref_scores = _backward_once(
            model_name,
            ds,
            ref_dir,
            save_mode=save_mode,
            resume=False,
            save_every=0,
            fwd_resume=False,
        )

    # Interrupt the backward pass after two backward checkpoints are written,
    # then resume in a fresh "process" sharing the same checkpoint dir.
    with tempfile.TemporaryDirectory() as ckpt_dir:
        orig_save = Trainer.save_backward_state
        n_saves = {"count": 0}

        def crashing_save(self, *args, **kwargs):
            orig_save(self, *args, **kwargs)
            n_saves["count"] += 1
            if n_saves["count"] >= 2:
                raise RuntimeError("simulated crash mid-backward")

        Trainer.save_backward_state = crashing_save
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                _backward_once(
                    model_name,
                    ds,
                    ckpt_dir,
                    save_mode=save_mode,
                    resume=False,
                    save_every=1,
                    fwd_resume=False,
                )
        finally:
            Trainer.save_backward_state = orig_save

        assert n_saves["count"] >= 2, "crash fired before any backward checkpoint"

        # A backward checkpoint must be on disk for the resume to pick up.
        bwd_pt = os.path.join(ckpt_dir, "backward_rank0.pt")
        assert os.path.exists(bwd_pt), "backward checkpoint not written"

        resumed_scores = _backward_once(
            model_name,
            ds,
            ckpt_dir,
            save_mode=save_mode,
            resume=True,
            save_every=1,
            fwd_resume=True,
        )

    torch.testing.assert_close(resumed_scores, ref_scores, atol=1e-5, rtol=1e-4)
