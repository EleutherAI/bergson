"""Unit tests for the streaming shard loader's pure logic."""

import pytest
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


def test_slices_reconstruct_the_checkpoint():
    """Concatenating every rank's slice must give back the checkpoint tensor.

    chunk_bounds being right is not enough: the reader also has to map the
    parameter path to the right key and read the right rows. If it did not,
    training would still run and produce plausible-looking scores from a
    scrambled model.
    """
    import glob

    pytest.importorskip("safetensors")
    from safetensors import safe_open

    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download("EleutherAI/pythia-160m", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("pythia-160m is not in the local hub cache")

    from bergson.magic.shard_load import ShardReader

    files = glob.glob(f"{root}/*.safetensors")
    assert files, "no safetensors in the cached snapshot"

    truth = {}
    with safe_open(files[0], framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118
            tensor = handle.get_tensor(key)
            # Legacy scalar buffers are not parameters and never get sharded.
            if tensor.dim() > 0:
                truth[key] = tensor
            if len(truth) >= 12:
                break

    empties = 0
    for world in (1, 3, 4, 128):
        readers = [
            ShardReader("EleutherAI/pythia-160m", world, r) for r in range(world)
        ]
        try:
            for key, full in truth.items():
                pieces = []
                for rank in range(world):
                    lo, hi = chunk_bounds(full.shape[0], world, rank)
                    got = readers[rank].local("base_model.model." + key, full)
                    assert got.shape[0] == hi - lo
                    empties += got.shape[0] == 0
                    pieces.append(got)
                assert torch.equal(torch.cat(pieces, dim=0), full), (key, world)
        finally:
            for reader in readers:
                reader.close()
    assert empties > 0, "the world=128 case should produce empty shards"
