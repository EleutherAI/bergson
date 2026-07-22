"""Normalizers and head-coverage validation for attention-head splitting."""

import pytest
import torch
from datasets import Dataset
from torch import nn

from bergson import GradientProcessor
from bergson.collector.collector import HookCollectorBase
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import AttentionConfig, IndexConfig
from bergson.gradients import AdafactorNormalizer, AdamNormalizer

O, I, N, S = 8, 6, 2, 3
HEADS, HEAD_SIZE = 2, 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(I, O, bias=False)

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, x):
        return self.proj(x)


def _collect(normalizers, attention_cfgs, tmp_path):
    torch.manual_seed(0)
    model = _Model()
    processor = GradientProcessor(normalizers=normalizers or {})
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)
    collector = InMemoryCollector(
        model=model,
        data=Dataset.from_dict({"input_ids": [[1, 2, 3]] * N}),
        cfg=cfg,
        processor=processor,
        attention_cfgs=attention_cfgs,
    )
    with collector:
        model.zero_grad()
        model(torch.randn(N, S, I)).sum().backward()
        return {k: v.clone() for k, v in collector.mod_grads.items()}


def test_split_head_name():
    assert HookCollectorBase.split_head_name("a.b.head_3") == ("a.b", 3)
    assert HookCollectorBase.split_head_name("a.b") is None
    assert HookCollectorBase.split_head_name("a.head_x") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_head_normalizer_is_applied(tmp_path):
    """A split module must still be normalized, using its parent's stats."""
    cfgs = {"proj": AttentionConfig(num_heads=HEADS, head_size=HEAD_SIZE, head_dim=2)}
    avg_sq = torch.full((O, I), 4.0)

    raw = _collect(None, cfgs, tmp_path)
    normed = _collect({"proj": AdamNormalizer(avg_sq)}, cfgs, tmp_path)

    assert raw and set(raw) == set(normed)
    for key in raw:
        # normalize_weight divides by sqrt(4) = 2
        torch.testing.assert_close(normed[key], raw[key] / 2, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_head_normalizer_slices_the_output_dim(tmp_path):
    """Each head gets the slice of the output-dim factors it covers."""
    cfgs = {"proj": AttentionConfig(num_heads=HEADS, head_size=HEAD_SIZE, head_dim=2)}
    # Distinct value per output row, so a wrong slice is visible.
    avg_sq = (torch.arange(O, dtype=torch.float32) + 1).pow(2)[:, None].expand(O, I)

    raw = _collect(None, cfgs, tmp_path)
    normed = _collect({"proj": AdamNormalizer(avg_sq.contiguous())}, cfgs, tmp_path)

    for head in range(HEADS):
        key = f"proj.head_{head}"
        lo = head * HEAD_SIZE
        scale = (torch.arange(HEAD_SIZE, dtype=torch.float32) + 1 + lo)[None, :, None]
        expected = raw[key].reshape(N, HEAD_SIZE, I) / scale
        torch.testing.assert_close(
            normed[key].reshape(N, HEAD_SIZE, I), expected, rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adafactor_head_normalizer_slices_row_only(tmp_path):
    """Adafactor's row factor is per-output-row; col is shared across heads."""
    cfgs = {"proj": AttentionConfig(num_heads=HEADS, head_size=HEAD_SIZE, head_dim=2)}
    norm = AdafactorNormalizer(torch.arange(1, O + 1).float(), torch.ones(I))

    collector_norms = _collect({"proj": norm}, cfgs, tmp_path)
    assert set(collector_norms) == {f"proj.head_{h}" for h in range(HEADS)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_head_coverage_must_match_output_features(tmp_path):
    """num_heads * head_size that under-covers the module must not pass."""
    cfgs = {"proj": AttentionConfig(num_heads=2, head_size=3, head_dim=2)}  # 6 != 8
    with pytest.raises(AssertionError, match="does not cover"):
        _collect(None, cfgs, tmp_path)
