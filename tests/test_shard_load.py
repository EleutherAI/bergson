"""Unit tests for the streaming shard loader's pure logic."""

import torch

from bergson.magic.shard_load import checkpoint_key, chunk_bounds, lora_init


def test_chunk_bounds_matches_torch_chunk():
    """DTensor's Shard(0) follows torch.chunk, so the bounds must agree.

    Covers even splits, uneven splits, and more ranks than rows -- the last
    happens for a LoRA A matrix (r=32) sharded over 128 ranks.
    """
    for n in (1, 2, 7, 8, 32, 35, 100):
        for world in (1, 2, 3, 4, 8, 128):
            want = [c.shape[0] for c in torch.arange(n).chunk(world)]
            want += [0] * (world - len(want))
            got = []
            for rank in range(world):
                lo, hi = chunk_bounds(n, world, rank)
                assert 0 <= lo <= hi <= n, (n, world, rank, lo, hi)
                got.append(hi - lo)
            assert got == want, f"n={n} world={world}: {got} != {want}"
            assert sum(got) == n


def test_chunk_bounds_are_contiguous_and_cover_everything():
    n, world = 35, 4
    covered = []
    for rank in range(world):
        lo, hi = chunk_bounds(n, world, rank)
        covered.extend(range(lo, hi))
    assert covered == list(range(n))


def test_checkpoint_key_strips_peft_wrapping():
    assert (
        checkpoint_key(
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        checkpoint_key(
            "base_model.model.gpt_neox.layers.3.attention.dense.base_layer.bias"
        )
        == "gpt_neox.layers.3.attention.dense.bias"
    )
    # Untargeted parameters only lose the prefix.
    assert (
        checkpoint_key("base_model.model.model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_lora_init_is_identical_across_calls():
    """Every rank builds the adapter itself, so the values must not vary."""
    shape = torch.Size((32, 128))
    path = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    a = lora_init(path, shape, torch.float32)
    b = lora_init(path, shape, torch.float32)
    assert torch.equal(a, b)
    # A different parameter gets different values.
    other = lora_init(path.replace("layers.0", "layers.1"), shape, torch.float32)
    assert not torch.equal(a, other)


def test_lora_b_is_zero_and_a_is_bounded():
    shape = torch.Size((32, 128))
    b = lora_init("x.lora_B.default.weight", shape, torch.float32)
    assert torch.count_nonzero(b) == 0

    a = lora_init("x.lora_A.default.weight", shape, torch.float32)
    bound = 1.0 / 128**0.5
    assert a.abs().max() <= bound
    assert a.abs().max() > bound / 2  # actually spread over the range
