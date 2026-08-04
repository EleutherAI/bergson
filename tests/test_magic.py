"""MAGIC integration test: forward + backward through 2 training steps."""

import tempfile

import pytest
import torch
import torchopt
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.grad_accum import accumulate_grads
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
    attribute_tokens: bool,
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
    from bergson.magic.cli import attach_doc_ids_if_missing
    from bergson.magic.data_stream import pad_dataset_to_batch_size

    ds = attach_doc_ids_if_missing(dataset)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    num_docs = max(max(row) for row in ds["doc_ids"]) + 1

    padded_ds, num_docs_pad, pad_count, weight_pad_count = pad_dataset_to_batch_size(
        ds, batch_size, num_docs, "Test", 0
    )

    if attribute_tokens:
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

    if attribute_tokens:
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
        attribute_tokens=True,
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
        attribute_tokens=False,
        shuffle_seed=shuffle_seed,
        zero_subset=docs_to_drop,
    )
    loss_tok, _ = _train_and_query_loss(
        model_name,
        ds,
        batch_size=batch_size,
        attribute_tokens=True,
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
    attribute_tokens: bool,
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
    from bergson.magic.cli import attach_doc_ids_if_missing
    from bergson.magic.data_stream import pad_dataset_to_batch_size

    ds = attach_doc_ids_if_missing(dataset)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)

    num_docs = max(max(row) for row in ds["doc_ids"]) + 1

    padded_ds, num_docs_pad, pad_count, weight_pad_count = pad_dataset_to_batch_size(
        ds, batch_size, num_docs, "Test", 0
    )

    if attribute_tokens:
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

    per_tok, _ = _run_magic_cli(model_name, ds, len(ds), attribute_tokens=True)
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

    per_doc, _ = _run_magic_cli(model_name, dataset, N, attribute_tokens=False)
    per_tok, _ = _run_magic_cli(model_name, dataset, N, attribute_tokens=True)

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

    per_doc, _ = _run_magic_cli(model_name, ds, N, attribute_tokens=False)
    per_tok, _ = _run_magic_cli(model_name, ds, N, attribute_tokens=True)

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

    per_doc, _ = _run_magic_cli(model_name, ds, batch_size, attribute_tokens=False)
    per_tok, doc_ids = _run_magic_cli(model_name, ds, batch_size, attribute_tokens=True)

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
        model_name, ds, batch_size, attribute_tokens=True, shuffle_seed=shuffle_seed
    )
    per_doc, _ = _run_magic_cli(
        model_name, ds, batch_size, attribute_tokens=False, shuffle_seed=shuffle_seed
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
        model_name, ds, batch_size, attribute_tokens=True, shuffle_seed=shuffle_seed
    )
    per_doc, _ = _run_magic_cli(
        model_name, ds, batch_size, attribute_tokens=False, shuffle_seed=shuffle_seed
    )

    assert per_doc.shape == (num_docs,)
    assert doc_ids is not None
    assert doc_ids.shape == per_tok.shape

    agg = torch.zeros(num_docs, dtype=torch.float64)
    agg.scatter_add_(0, doc_ids.reshape(-1), per_tok.reshape(-1).to(torch.float64))

    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4)


TINY_MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _multi_step_stream(n_steps: int, device: str = "cpu") -> DataStream:
    """A dataset of `n_steps` single-row batches, so training takes `n_steps` steps."""
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1 + 5 * i + j for j in range(5)] for i in range(n_steps)],
            "labels": [[1 + 5 * i + j for j in range(5)] for i in range(n_steps)],
            "attention_mask": [[1] * 5] * n_steps,
        }
    )
    return DataStream(ds, batch_size=1, device=device)


def _fresh_trainer(seed: int = 42, dropout: float = 0.0):
    """Build a fresh model/trainer/state, as a new process would."""
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(TINY_MODEL)
    if dropout:
        config.resid_pdrop = dropout
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.train() if dropout else model.eval()
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)
    return trainer, fwd_state, model


def _magic_scores(trainer, model, fwd_state, stream, ckpt_dir) -> torch.Tensor:
    """Query gradients on batch 0, then MAGIC backward through the trajectory."""
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
    )
    return bwd_state.weight_grads.detach().cpu()


def _saved_steps(ckpt_dir: str) -> list[int]:
    from bergson.data import sorted_checkpoints

    return [idx for idx, _ in sorted_checkpoints(ckpt_dir)]


def test_trainer_state_in_memory_checkpoint_roundtrip():
    """`state.to(...)` then `copy_` must round-trip the *whole* state.

    This is the in-RAM checkpoint path in `Trainer.backward`. Two ways it used
    to lose information: `copy_` ignored `opt_state` entirely, and `to("cpu")`
    was a no-op for a state already on the CPU, so the snapshot aliased the live
    tensors and was clobbered by the next in-place step.
    """
    trainer, state, _ = _fresh_trainer()
    stream = _multi_step_stream(2)

    snapshot = state.to("cpu").detach_()
    before = [t.clone() for t in tree_iter(snapshot.opt_state)]

    # Two in-place steps: the snapshot must survive them untouched...
    for _ in range(2):
        state = trainer.step(state, stream[state.batch_index], inplace=True)

    for saved, now in zip(before, tree_iter(snapshot.opt_state)):
        assert torch.equal(saved, now), "in-memory checkpoint aliased the live state"

    moved = [t for t in tree_iter(state.opt_state)]
    assert any(
        not torch.equal(a, b) for a, b in zip(before, moved)
    ), "optimizer state didn't change; test is degenerate"

    # ...and copying it back must restore the optimizer state, not just params.
    state.detach_()
    state.copy_(snapshot)
    for saved, now in zip(before, tree_iter(state.opt_state)):
        assert torch.equal(saved, now), "copy_ dropped part of the optimizer state"


@pytest.mark.parametrize("save_mode", ["sqrt", "log"])
def test_magic_backward_matches_across_save_modes(save_mode, monkeypatch):
    """Sparse checkpointing must not change MAGIC scores.

    With save_mode != "all" the backward pass rematerializes intermediate states
    and stashes them in RAM as `TrainerState`s, restoring them via
    `TrainerState.copy_`. `copy_` used to copy the parameters but *not* the
    optimizer state, so every traced step at or below the first in-RAM checkpoint
    was differentiated against Adam moments belonging to a *later* step.
    Regression guard: scores must be identical to the dense ("all") schedule.
    """
    import types

    from bergson.magic import trainer as trainer_mod

    n = 9

    # Force the in-RAM checkpoint branch so the test doesn't depend on how much
    # memory psutil happens to report on the machine running it.
    monkeypatch.setattr(
        trainer_mod.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=1 << 60),
    )

    # Spy on copy_ to assert the in-RAM checkpoint path was actually taken.
    copy_calls = []
    orig_copy = trainer_mod.TrainerState.copy_

    def spy_copy(self, other):
        copy_calls.append(other.batch_index)
        return orig_copy(self, other)

    monkeypatch.setattr(trainer_mod.TrainerState, "copy_", spy_copy)

    scores = {}
    for mode in ("all", save_mode):
        trainer, fwd_state, model = _fresh_trainer()
        stream = _multi_step_stream(n)
        assert len(stream) == n

        with tempfile.TemporaryDirectory() as ckpt_dir:
            fwd_state = trainer.train(
                fwd_state, stream, inplace=True, save_dir=ckpt_dir, save_mode=mode
            )
            scores[mode] = _magic_scores(trainer, model, fwd_state, stream, ckpt_dir)

    assert copy_calls, "no in-RAM checkpoints were used; test is degenerate"
    assert scores["all"].abs().sum() > 0, "scores are all zero; test is degenerate"

    torch.testing.assert_close(scores[save_mode], scores["all"], atol=1e-12, rtol=1e-6)


@pytest.mark.parametrize("save_mode", ["all", "sqrt", "log"])
def test_magic_resume_preserves_checkpoint_schedule(save_mode):
    """A resumed forward run must keep saving checkpoints on schedule.

    `next_save` was initialized to 0 before the resume branch set `start =
    state.batch_index`, so with `start > 0` the `i == next_save` condition never
    fired again and a resumed run saved nothing. Final parameters still matched a
    fresh run exactly, so the damage only showed up in the backward pass, which
    started from the last surviving checkpoint and emitted zero scores for every
    step past it.
    """
    n = 9
    crash_at = 5

    # Reference: an uninterrupted run.
    trainer, fwd_state, model = _fresh_trainer()
    stream = _multi_step_stream(n)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state, stream, inplace=True, save_dir=ckpt_dir, save_mode=save_mode
        )
        fresh_steps = _saved_steps(ckpt_dir)
        fresh_params = {k: v.detach().clone() for k, v in fwd_state.params.items()}
        fresh_scores = _magic_scores(trainer, model, fwd_state, stream, ckpt_dir)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # Interrupted run: blow up partway through training.
        def boom(i, loss):
            if i == crash_at:
                raise RuntimeError("simulated crash")

        trainer, fwd_state, model = _fresh_trainer()
        stream = _multi_step_stream(n)
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer.train(
                fwd_state,
                stream,
                inplace=True,
                save_dir=ckpt_dir,
                save_mode=save_mode,
                log_fn=boom,
            )

        # Resume in a fresh process-like context.
        trainer, fwd_state, model = _fresh_trainer()
        stream = _multi_step_stream(n)
        fwd_state = trainer.train(
            fwd_state,
            stream,
            inplace=True,
            save_dir=ckpt_dir,
            save_mode=save_mode,
            resume=True,
        )
        resumed_steps = _saved_steps(ckpt_dir)
        # Capture before the backward pass, which walks `fwd_state` back down
        # the trajectory in place.
        resumed_params = {k: v.detach().clone() for k, v in fwd_state.params.items()}
        resumed_scores = _magic_scores(trainer, model, fwd_state, stream, ckpt_dir)

    assert resumed_steps == fresh_steps, (
        f"resumed run saved checkpoints {resumed_steps}, "
        f"uninterrupted run saved {fresh_steps}"
    )
    for k, v in fresh_params.items():
        torch.testing.assert_close(resumed_params[k], v)

    assert fresh_scores.abs().sum() > 0, "scores are all zero; test is degenerate"
    torch.testing.assert_close(resumed_scores, fresh_scores, atol=1e-12, rtol=1e-6)


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


def test_prepare_trainer_respects_train_mode(tmp_path):
    """train_mode drives the model's train/eval mode; default is eval."""
    from bergson.magic.config import MagicConfig
    from bergson.magic.trainer import prepare_trainer

    def build(train_mode):
        cfg = MagicConfig(
            run_path=str(tmp_path),
            model="EleutherAI/pythia-14m",
            train_mode=train_mode,
        )
        _, _, model = prepare_trainer(cfg, rank=0, schedule=lambda step: 1e-4)
        return model.training

    assert build(False) is False
    assert build(True) is True


def test_magic_backward_matches_across_save_modes_with_dropout(monkeypatch):
    """Sparse checkpointing must not change MAGIC scores when dropout is active.

    Each save mode rematerializes a different number of steps before the traced
    re-do, so if the replay drew fresh dropout masks instead of restoring each
    step's saved RNG, the schedules would disagree.
    """
    import types

    from bergson.magic import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=1 << 60),
    )

    scores = {}
    for mode in ("all", "log"):
        trainer, fwd_state, model = _fresh_trainer(dropout=0.5)
        assert any(
            isinstance(m, torch.nn.Dropout) and m.training and m.p > 0
            for m in model.modules()
        ), "dropout is not active; test is degenerate"
        stream = _multi_step_stream(9)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            fwd_state = trainer.train(
                fwd_state, stream, inplace=True, save_dir=ckpt_dir, save_mode=mode
            )
            scores[mode] = _magic_scores(trainer, model, fwd_state, stream, ckpt_dir)

    assert scores["all"].abs().sum() > 0, "scores are all zero; test is degenerate"
    torch.testing.assert_close(scores["log"], scores["all"], atol=1e-12, rtol=1e-6)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_metagrad_step_matches_single_shot_under_dropout(dataset, device):
    """The micro-batched VJP must reproduce the single-shot one, dropout included.

    ``metagrad_step`` runs the model once per micro-batch in stage 0 and again
    in stage B, against a forward pass that ran it once per micro-batch too.
    All three must draw the same dropout masks, which only holds if the replay
    rewinds the RNG the way ``Trainer.step`` does. With dropout off every draw
    is a no-op and the bug is invisible, so this test turns it up high.

    Runs on CUDA too: dropout there draws from the CUDA generator, so rewinding
    only the CPU one passes the CPU case while breaking the GPU one.
    """
    torch.manual_seed(42)

    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    config.hidden_dropout = 0.5
    config.attention_dropout = 0.5
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    model.train()  # dropout is a no-op in eval mode

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    stream = DataStream(dataset, batch_size=len(dataset), device=device)
    stream.requires_grad = True
    # Two independent views of the same batch: each carries its own
    # ``example_weight`` graph back to ``stream.weights``, so the first path's
    # VJP doesn't free the buffers the second one needs.
    batch, batch_mb = stream[0], stream[0]

    fwd_state.detach_()
    fwd_state.requires_grad = True

    # Fixed incoming cotangents, shared by both paths.
    torch.manual_seed(7)
    param_cot = {k: torch.randn_like(v) for k, v in fwd_state.params.items()}
    opt_cot = [
        torch.zeros_like(t)
        for t in tree_iter(fwd_state.opt_state)
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    ]
    bwd_state = BackwardState(param_cot, opt_cot, torch.zeros_like(stream.weights))

    # Reference: one traced step holding every micro-graph at once, then a
    # single VJP -- what metagrad_step decomposes without the memory cost.
    flat_i = fwd_state.differentiable_tensors()
    state_f = trainer.step(fwd_state, batch, trace=True, grad_accum_steps=2)
    ref = torch.autograd.grad(
        state_f.differentiable_tensors(),
        flat_i + [stream.weights],
        grad_outputs=list(param_cot.values()) + opt_cot,
        allow_unused=True,
    )
    ref_params = dict(zip(param_cot.keys(), ref))
    ref_weights = ref[-1]
    assert ref_weights is not None and ref_weights.abs().sum() > 0

    out = trainer.metagrad_step(
        fwd_state, batch_mb, bwd_state, stream.weights, grad_accum_steps=2
    )

    for k, expected in ref_params.items():
        if expected is None:
            continue
        torch.testing.assert_close(
            out.param_grads[k],
            expected,
            msg=lambda m, k=k: f"param cotangent mismatch for {k}\n{m}",
        )
    torch.testing.assert_close(out.weight_grads, ref_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_step_replay_is_deterministic_under_cuda_dropout(dataset):
    """Re-running a step from the same state must reproduce it bit for bit.

    The whole backward pass rests on this: it replays the forward trajectory
    from checkpoints, so a step re-run later — with arbitrary RNG consumption
    in between — has to land on the same next state. ``TrainerState`` records
    both generators for exactly this reason, and CUDA dropout draws from the
    CUDA one, so rewinding only the CPU generator silently breaks the replay
    on GPU while looking fine on CPU.
    """
    device = "cuda"
    torch.manual_seed(42)

    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    config.hidden_dropout = 0.5
    config.attention_dropout = 0.5
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    model.train()  # dropout is a no-op in eval mode

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    stream = DataStream(dataset, batch_size=len(dataset), device=device)
    batch = stream[0]

    first = trainer.step(fwd_state, batch, inplace=False)

    # Advance both generators, standing in for everything the backward pass
    # does between the original step and its replay.
    torch.rand(1024)
    torch.rand(1024, device=device)

    replay = trainer.step(fwd_state, batch, inplace=False)

    for k, p in first.params.items():
        torch.testing.assert_close(
            replay.params[k],
            p,
            msg=lambda m, k=k: f"replayed step diverged for {k}\n{m}",
        )


@pytest.mark.parametrize(
    "valid_lens",
    [
        [31, 4, 27, 2, 31, 9, 31, 5],  # uneven valid-token counts
        [31, 4, 27, 2, 0, 0, 0, 0],  # second micro-batch entirely empty
    ],
)
def test_accumulate_grads_matches_full_batch(valid_lens):
    """Micro-batch accumulation must reproduce the full-batch gradient.

    weighted_causal_lm_ce normalizes by the batch's valid-token count
    (``shift_loss_mask.sum()``), so each micro-gradient must be rescaled by
    its share of it — a bug there is invisible to tests that compare the two
    accumulation code paths against each other. Uneven masks make a wrong
    rescale show up as O(1) gradient error; the all-empty case checks that
    zero-denominator micro-batches are skipped instead of poisoning the sum
    with 0/0 NaNs.
    """
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    model = AutoModelForCausalLM.from_config(config).double().eval()
    model.loss_function = weighted_causal_lm_ce
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}

    B, T = len(valid_lens), 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = input_ids.clone()
    for i, n in enumerate(valid_lens):
        labels[i, n + 1 :] = -100
    shift_loss_mask = torch.zeros(B, T, dtype=torch.bool)
    shift_loss_mask[:, :-1] = labels[:, 1:] != -100
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "shift_loss_mask": shift_loss_mask,
        "example_weight": torch.ones(B, dtype=torch.float64),
    }

    g_full = grad_tree(model(**batch).loss, params, create_graph=False)
    g_accum, _ = accumulate_grads(model, params, batch, 2, create_graph=False)

    num = sum((g_accum[k] - g_full[k]).pow(2).sum() for k in g_full).sqrt()
    den = sum(g_full[k].pow(2).sum() for k in g_full).sqrt()
    # The loss casts logits to fp32 internally, so fp32-level associativity
    # noise is the floor even for a float64 model.
    assert num / den < 1e-6, f"accumulated gradient off by {num / den:.3e}"


def test_next_save_index_interval():
    from bergson.magic.trainer import next_save_index

    assert next_save_index(0, 1746, "interval", save_interval=291) == 291
    assert next_save_index(291, 1746, "interval", save_interval=291) == 582
    with pytest.raises(ValueError, match="save_interval"):
        next_save_index(0, 100, "interval")


def test_per_token_backward_compatibility():
    from bergson.magic.config import MagicConfig

    cfg = MagicConfig.from_dict(
        {"run_path": "x", "per_token": True}, drop_extra_fields=False
    )
    assert cfg.attribute_tokens


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_magic_grad_accum_weight_grads_match(model_name, dataset, dtype):
    """The ga>1 (microbatch) metagradient must match the ga=1 (traced) path.

    Regression guard for ``microbatch_step_vjp``'s weight-gradient path, which
    had no coverage. Over a single step the fp-associativity between the two
    summation orders is negligible, so the paths must agree to tight tolerance;
    an algorithmic difference would show here.
    """

    def scores(ga: int) -> torch.Tensor:
        torch.manual_seed(42)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager"
        ).to(dtype)
        model.eval()
        model.loss_function = weighted_causal_lm_ce
        model.requires_grad_(True)
        optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-8)
        trainer, fwd_state = Trainer.initialize(model, optimizer)
        stream = DataStream(dataset, batch_size=len(dataset), device="cpu")
        with tempfile.TemporaryDirectory() as ckpt_dir:
            fwd_state = trainer.train(
                fwd_state,
                stream,
                inplace=True,
                save_dir=ckpt_dir,
                grad_accum_steps=ga,
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
                grad_accum_steps=ga,
            )
        return bwd_state.weight_grads.detach().float().cpu()

    s1 = scores(1)
    s2 = scores(2)
    assert s1.abs().sum() > 0, "metagradient is all zero; test is degenerate"
    torch.testing.assert_close(s2, s1, atol=1e-6, rtol=1e-4)
