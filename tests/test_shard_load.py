import torch
from safetensors.torch import save_file

from bergson.magic.shard_load import ShardReader, chunk_bounds, lora_init


def test_chunk_bounds_matches_torch_chunk():
    for n in (1, 7, 32, 35):
        for world in (1, 3, 4, 128):
            want = [c.shape[0] for c in torch.arange(n).chunk(world)]
            want += [0] * (world - len(want))
            got = [
                hi - lo for lo, hi in (chunk_bounds(n, world, r) for r in range(world))
            ]
            assert got == want


def test_lora_init_is_deterministic():
    shape = torch.Size((32, 128))
    path = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    a = lora_init(path, shape, torch.float32)
    assert torch.equal(a, lora_init(path, shape, torch.float32))
    assert a.abs().max() <= 1.0 / 128**0.5
    b = lora_init(path.replace("lora_A", "lora_B"), shape, torch.float32)
    assert torch.count_nonzero(b) == 0


def test_slices_reconstruct_the_checkpoint(tmp_path):
    truth = {
        "model.layers.0.q_proj.weight": torch.randn(35, 8),
        "model.embed_tokens.weight": torch.randn(3, 8),
    }
    save_file(truth, tmp_path / "model.safetensors")
    prefix = "base_model.model."
    paths = {
        prefix
        + "model.layers.0.q_proj.base_layer.weight": ("model.layers.0.q_proj.weight"),
        prefix + "model.embed_tokens.weight": "model.embed_tokens.weight",
    }
    for world in (1, 4):
        readers = [ShardReader(tmp_path, world, r) for r in range(world)]
        for path, key in paths.items():
            pieces = [r.local(path, truth[key]) for r in readers]
            assert torch.equal(torch.cat(pieces), truth[key])
        for reader in readers:
            reader.close()
