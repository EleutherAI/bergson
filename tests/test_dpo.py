"""DPO gradients are per pair: they match autograd of each pair's DPO loss."""

import json
import math
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, Olmo3Config

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import (
    DataConfig,
    DistributedConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    QuerySetConfig,
    ScoreConfig,
)
from bergson.data import (
    allocate_dpo_batches,
    column_offsets,
    expand_dpo_dataset,
    load_gradients,
    load_scores,
    pad_and_tensor,
)
from bergson.dpo import (
    combine_pair_gradients,
    combine_pair_outer_products,
    dpo_ggn_surrogate,
    dpo_losses,
    dpo_margin,
)
from bergson.gradients import GradientProcessor, OuterProductGradients
from bergson.hessians.pipeline import hessian_pipeline
from bergson.score.output_influence import output_token_influence, query_directions
from bergson.utils.recompute import checkpoint_attentions, checkpoint_mlps

BETA = 5.0


def test_dpo_losses_and_hessian_surrogate():
    torch.manual_seed(0)
    theta = torch.randn(3, dtype=torch.float64, requires_grad=True)
    policy = torch.stack([theta.sum(), theta[0] * 2, theta[1] - theta[2], theta[2]])
    reference = torch.tensor([0.1, -0.3, 0.2, 0.4], dtype=torch.float64)

    margin = dpo_margin(policy, reference, BETA)
    expected = -F.logsigmoid(
        BETA * ((policy[:2] - reference[:2]) - (policy[2:] - reference[2:]))
    )
    torch.testing.assert_close(dpo_losses(policy, reference, BETA), expected)

    # Each pair's surrogate gradient is sqrt(sigmoid'(margin)) times its margin
    # gradient, so its outer product is the pair's Gauss-Newton Hessian
    for pair in range(2):
        (g_surrogate,) = torch.autograd.grad(
            dpo_ggn_surrogate(policy, reference, BETA)[pair], theta, retain_graph=True
        )
        (g_margin,) = torch.autograd.grad(margin[pair], theta, retain_graph=True)
        weight = torch.sigmoid(margin[pair]) * torch.sigmoid(-margin[pair])
        torch.testing.assert_close(g_surrogate, weight.sqrt().detach() * g_margin)


def test_expand_dpo_dataset_and_allocate_pairs():
    pairs = Dataset.from_dict(
        {
            "chosen_input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "chosen_labels": [[-100, 2, 3], [-100, 5], [-100, -100, 8, 9]],
            "chosen_length": [3, 2, 4],
            "rejected_input_ids": [[1, 2], [4, 5, 6, 7], [6, 7, 1]],
            "rejected_labels": [[-100, 2], [-100, 5, 6, 7], [-100, -100, 1]],
            "rejected_length": [2, 4, 3],
            "reference_chosen_logps": [-1.0, -2.0, -3.0],
            "reference_rejected_logps": [-4.0, -5.0, -6.0],
            "source": ["a", "b", "c"],
        }
    )
    rows = expand_dpo_dataset(pairs)

    assert len(rows) == 6
    assert rows["input_ids"][:3] == pairs["chosen_input_ids"]
    assert rows["input_ids"][3:] == pairs["rejected_input_ids"]
    assert rows["reference_logps"] == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    assert rows["pair_index"] == [0, 1, 2, 0, 1, 2]
    assert rows["source"] == ["a", "b", "c"] * 2

    batches = allocate_dpo_batches(rows["length"], 64)
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    for batch in batches:
        half = len(batch) // 2
        assert [i + 3 for i in batch[:half]] == batch[half:]


@pytest.mark.parametrize("bias", [False, True])
def test_combine_pair_outer_products_matches_materialized(bias: bool):
    torch.manual_seed(0)
    grads = OuterProductGradients(
        g=torch.randn(4, 5, 3),
        a=torch.randn(4, 5, 6),
        bias=torch.randn(4, 3) if bias else None,
        bias_col=torch.eye(6)[-1] if bias else None,
    )
    torch.testing.assert_close(
        combine_pair_outer_products(grads).materialize(),
        combine_pair_gradients(grads.materialize()),
    )


@dataclass(kw_only=True)
class _KeepGradients(GradientCollector):
    """Keeps each batch's pair gradients before they are cleared."""

    kept: dict = field(default_factory=dict)

    def process_batch(self, indices, **kwargs):
        pairs = list(indices[: len(indices) // 2])
        for name, grad in self.mod_grads.items():
            for row, pair in enumerate(pairs):
                self.kept.setdefault(pair, {})[name] = grad[row].clone()
        super().process_batch(indices, **kwargs)


def _tiny_model(attn_implementation: str = "sdpa"):
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GPTNeoXForCausalLM")
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float64, attn_implementation=attn_implementation
    )


def _pair_rows():
    generator = torch.Generator().manual_seed(0)

    def side(lengths, prompt):
        ids = [
            torch.randint(1, 100, (n,), generator=generator).tolist() for n in lengths
        ]
        labels = [[-100] * prompt + seq[prompt:] for seq in ids]
        return ids, labels

    chosen_ids, chosen_labels = side([7, 5, 9], 2)
    rejected_ids, rejected_labels = side([6, 8, 4], 2)
    return expand_dpo_dataset(
        Dataset.from_dict(
            {
                "chosen_input_ids": chosen_ids,
                "chosen_labels": chosen_labels,
                "chosen_length": [len(x) for x in chosen_ids],
                "rejected_input_ids": rejected_ids,
                "rejected_labels": rejected_labels,
                "rejected_length": [len(x) for x in rejected_ids],
                "reference_chosen_logps": [-4.0, -3.5, -5.0],
                "reference_rejected_logps": [-4.2, -3.0, -4.5],
            }
        )
    )


def _sequence_log_prob(model, ids, labels):
    logits = model(torch.tensor([ids])).logits[0, :-1]
    targets = torch.tensor(labels[1:])
    mask = targets != -100
    token_lps = logits.log_softmax(-1).gather(-1, targets.clamp(min=0)[:, None])[:, 0]
    return (token_lps * mask).sum() / mask.sum()


@pytest.mark.parametrize("pairs_per_batch", [1, 3])
def test_dpo_pair_gradients_match_autograd(tmp_path, pairs_per_batch: int):
    model = _tiny_model()
    rows = _pair_rows()
    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        loss_fn="dpo",
        dpo_beta=BETA,
        loss_reduction="mean",
    )
    cfg.partial_run_path.mkdir(parents=True)
    collector = _KeepGradients(
        model=model.base_model,
        data=rows,
        cfg=cfg,
        processor=GradientProcessor(),
        skip_index=True,
    )
    budget = 1000 if pairs_per_batch > 1 else 18
    batches = allocate_dpo_batches(rows["length"], budget)
    assert max(len(b) for b in batches) == 2 * pairs_per_batch
    CollectorComputer(
        model=model, data=rows, collector=collector, batches=batches, cfg=cfg
    ).run_with_collector_hooks()

    n = len(rows) // 2
    for pair in range(n):
        log_probs = torch.stack(
            [
                _sequence_log_prob(model, rows["input_ids"][i], rows["labels"][i])
                for i in (pair, pair + n)
            ]
        )
        references = torch.tensor(
            [rows["reference_logps"][pair], rows["reference_logps"][pair + n]],
            dtype=torch.float64,
        )
        loss = dpo_losses(log_probs, references, BETA).sum()
        model.zero_grad()
        loss.backward()
        for name, grad in collector.kept[pair].items():
            expected = model.base_model.get_submodule(name).weight.grad
            torch.testing.assert_close(grad, expected.flatten(), rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("unpadded", [False, True])
def test_dpo_output_token_influence_sums_to_pair_gradient(unpadded: bool):
    # The fused attention kernels don't implement forward-mode derivatives
    model = _tiny_model("eager").eval()
    rows = _pair_rows()
    n = len(rows) // 2
    cfg = IndexConfig(
        run_path="unused", loss_fn="dpo", dpo_beta=BETA, loss_reduction="mean"
    )
    target_info = GradientCollector.discover_targets(model.base_model, None, False)
    generator = torch.Generator().manual_seed(1)
    blocks = {
        name: torch.randn(math.prod(shape), 2, generator=generator, dtype=torch.float64)
        for name, (_, shape, _) in target_info.items()
    }
    directions = query_directions(model, blocks, target_info)

    batch = rows[list(range(len(rows)))]
    x, y, _, _ = pad_and_tensor(
        batch["input_ids"], labels=batch["labels"], sync_max_len=False
    )
    rates, losses = output_token_influence(
        model,
        x,
        y,
        directions,
        cfg,
        reference_logps=batch["reference_logps"],
        lengths=batch["length"] if unpadded else None,
    )
    row_sums = rates.sum(dim=1)  # [2 n, queries]
    pair_scores = row_sums[:n] + row_sums[n:]

    for pair in range(n):
        log_probs = torch.stack(
            [
                _sequence_log_prob(model, rows["input_ids"][i], rows["labels"][i])
                for i in (pair, pair + n)
            ]
        )
        references = torch.tensor(
            [rows["reference_logps"][pair], rows["reference_logps"][pair + n]],
            dtype=torch.float64,
        )
        loss = dpo_losses(log_probs, references, BETA).sum()
        model.zero_grad()
        loss.backward()
        expected = sum(
            blocks[name].T @ model.base_model.get_submodule(name).weight.grad.flatten()
            for name in blocks
        )
        torch.testing.assert_close(
            pair_scores[pair].double(), expected, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            losses[pair].double(), loss.detach(), rtol=1e-5, atol=1e-7
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dpo_hessian_pipeline_scores_pairs(tmp_path):
    """Fit EK-FAC on DPO pairs, transform an NLL query and score the pairs: each
    pair's score is the transformed query's dot product with its DPO gradient."""
    generator = torch.Generator().manual_seed(0)

    def rows(lengths, prompt=3):
        ids = [
            torch.randint(10, 1000, (n,), generator=generator).tolist() for n in lengths
        ]
        return ids, [[-100] * prompt + seq[prompt:] for seq in ids]

    chosen_ids, chosen_labels = rows([12, 9, 15, 11, 8, 14])
    rejected_ids, rejected_labels = rows([10, 13, 9, 12, 15, 7])
    pairs = Dataset.from_dict(
        {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_length": [len(x) for x in chosen_ids],
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_length": [len(x) for x in rejected_ids],
            "reference_chosen_logps": [-6.0, -5.5, -7.0, -6.2, -5.8, -6.6],
            "reference_rejected_logps": [-6.1, -5.0, -6.5, -6.0, -6.4, -6.3],
        }
    )
    pairs.save_to_disk(str(tmp_path / "pairs"))
    query_ids, query_labels = rows([9, 11, 10])
    Dataset.from_dict({"input_ids": query_ids, "labels": query_labels}).save_to_disk(
        str(tmp_path / "query")
    )

    run = tmp_path / "run"
    hessian_pipeline(
        IndexConfig(
            run_path=str(run),
            model="EleutherAI/pythia-14m",
            data=DataConfig(dataset=str(tmp_path / "pairs")),
            token_batch_size=512,
            precision="fp32",
            filter_modules="embed_out",
            loss_fn="dpo",
            dpo_beta=BETA,
            loss_reduction="mean",
            # The fused attention kernels' fp32 gradients for this model are
            # too far from autograd's for the comparison below
            force_math_sdp=True,
            # Six pairs can't be spread over every GPU
            distributed=DistributedConfig(nproc_per_node=1),
        ),
        HessianConfig(method="kfac", ev_correction=True),
        ScoreConfig(batch_size=64),
        PreprocessConfig(),
        HessianPipelineConfig(
            query=QuerySetConfig(dataset=str(tmp_path / "query"), aggregation="mean"),
            query_loss_fn="ce",
            query_loss_reduction="mean",
        ),
    )

    scores = load_scores(run / "scores")
    assert len(scores) == len(pairs)
    with open(run / "kfac_query" / "info.json") as f:
        grad_sizes = json.load(f)["grad_sizes"]
    query = torch.from_numpy(load_gradients(run / "kfac_query")[0].astype("float64"))

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m", torch_dtype=torch.float64
    )
    expanded = expand_dpo_dataset(pairs)
    n = len(pairs)
    for pair in range(n):
        log_probs = torch.stack(
            [
                _sequence_log_prob(
                    model, expanded["input_ids"][i], expanded["labels"][i]
                )
                for i in (pair, pair + n)
            ]
        )
        references = torch.tensor(
            [expanded["reference_logps"][pair], expanded["reference_logps"][pair + n]],
            dtype=torch.float64,
        )
        model.zero_grad()
        dpo_losses(log_probs, references, BETA).sum().backward()
        products = torch.cat(
            [
                query[start:end]
                * model.base_model.get_submodule(name).weight.grad.flatten()
                for name, (start, end) in column_offsets(grad_sizes).items()
            ]
        )
        # The fit on six pairs leaves the transformed query poorly conditioned,
        # so measure the fp32 scoring error against the size of the summed terms
        error = abs(float(scores.mmap["score_0"][pair]) - float(products.sum()))
        assert error <= 1e-4 * float(products.abs().sum()) + 1e-8


def _tiny_olmo3():
    config = Olmo3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float64, attn_implementation="sdpa"
    )


def _olmo3_pair_gradients(tmp_path, recompute: bool, offload: bool):
    model = _tiny_olmo3()
    if recompute:
        checkpoint_mlps(model)
        checkpoint_attentions(model)
    rows = _pair_rows()
    cfg = IndexConfig(
        run_path=str(tmp_path / f"run-{recompute}-{offload}"),
        loss_fn="dpo",
        dpo_beta=BETA,
        loss_reduction="mean",
        offload_activations=offload,
    )
    cfg.partial_run_path.mkdir(parents=True)
    collector = _KeepGradients(
        model=model.base_model,
        data=rows,
        cfg=cfg,
        processor=GradientProcessor(),
        skip_index=True,
    )
    CollectorComputer(
        model=model,
        data=rows,
        collector=collector,
        batches=allocate_dpo_batches(rows["length"], 1000),
        cfg=cfg,
    ).run_with_collector_hooks()
    return collector.kept


@pytest.mark.parametrize("offload", [False, True])
def test_recomputed_sublayers_give_the_same_pair_gradients(tmp_path, offload: bool):
    expected = _olmo3_pair_gradients(tmp_path, recompute=False, offload=False)
    actual = _olmo3_pair_gradients(tmp_path, recompute=True, offload=offload)
    assert actual.keys() == expected.keys()
    for pair, grads in expected.items():
        assert actual[pair].keys() == grads.keys()
        for name, grad in grads.items():
            torch.testing.assert_close(actual[pair][name], grad, rtol=1e-10, atol=1e-12)
