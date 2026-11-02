"""DPO gradients are per pair: they match autograd of each pair's DPO loss."""

import math
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.data import allocate_dpo_batches, expand_dpo_dataset, pad_and_tensor
from bergson.dpo import (
    combine_pair_gradients,
    combine_pair_outer_products,
    dpo_ggn_surrogate,
    dpo_losses,
    dpo_margin,
)
from bergson.gradients import GradientProcessor, OuterProductGradients
from bergson.score.output_influence import output_token_influence, query_directions

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
