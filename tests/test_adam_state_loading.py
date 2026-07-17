import tempfile

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from bergson.gradients import AdafactorNormalizer, AdamNormalizer
from bergson.utils.load_from_optimizer import (
    OptimizerStateFormat,
    get_optimizer_state_format,
    get_unfactored_second_moment,
    load_from_optimizer,
    optimizer_param_index_to_name,
    save_second_moments_as_optimizer_pt,
)
from bergson.utils.worker_utils import extract_peft_target_modules


def _create_model():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)


class _TiedModel(nn.Module):
    """Reproduces the GPT-2 weight tie (head.weight is emb.weight) plus a
    couple of untied 2D layers, to exercise the name-based normalizer mapping."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.blk_a = nn.Linear(4, 6, bias=True)
        self.blk_b = nn.Linear(6, 8, bias=False)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight


def test_save_second_moments_roundtrip_is_name_correct():
    """A torchopt AdamW state exports to optimizer.pt and each normalizer lands
    on the correct module -- even with a tied weight and torchopt's sorted-key
    nu ordering (the module-name mismatch this guards against)."""
    torchopt = pytest.importorskip("torchopt")

    model = _TiedModel()
    # dedup drops the tied duplicate: full param list is longer than named_parameters()
    assert len(list(model.named_parameters(remove_duplicate=False))) == 5
    assert len(list(model.named_parameters())) == 4

    params = {
        k: v
        for k, v in model.named_parameters(remove_duplicate=False)
        if v.requires_grad
    }
    opt = torchopt.adamw(1e-3, betas=(0.95, 0.975), eps_root=1e-8, weight_decay=0.01)
    state = opt.init(params)

    # Mark each param's second moment with a unique constant keyed by name, in
    # torchopt's sorted-key order, so we can verify the mapping end-to-end.
    adam = next(s for s in state if hasattr(s, "nu"))
    marker = {}
    for i, name in enumerate(sorted(params)):
        adam.nu[i] = torch.full_like(adam.nu[i], float(1000 + i))
        marker[name] = float(1000 + i)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/optimizer.pt"
        n = save_second_moments_as_optimizer_pt(model, state, path)
        assert n == 4  # emb(=head), blk_a, blk_b weights

        normalizers = load_from_optimizer(model, path)

    # tie resolves to the surviving deduplicated name, not the alias
    assert "emb" in normalizers and "head" not in normalizers
    for module_name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        assert float(norm.weight_avg_sq.flatten()[0]) == marker[module_name + ".weight"]


def _create_fake_optimizer_state(model, lr=1e-3):
    """Create a fake optimizer state dict matching the model's parameters."""
    state = {}
    param_groups = [{"lr": lr, "params": []}]

    for idx, (name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(100),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }

    return {"state": state, "param_groups": param_groups}


def _train_checkpoint(optim_name: str) -> tuple:
    """Train a tiny model for a few steps and return (checkpoint_path, model)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dummy_data = Dataset.from_dict({"text": ["hello world"] * 20, "label": [0, 1] * 10})
    dummy_data = dummy_data.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=32
        ),
        batched=True,
    )

    tmpdir = tempfile.mkdtemp()
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=3,
        save_steps=3,
        per_device_train_batch_size=4,
        optim=optim_name,
        learning_rate=1e-4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dummy_data)
    trainer.train()

    import os

    ckpt = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")][0]
    return os.path.join(tmpdir, ckpt), model


# ---------------------------------------------------------------------------
# Unit tests with fake state
# ---------------------------------------------------------------------------


def test_load_from_optimizer_file(tmp_path):
    """Load normalizers from a bare optimizer.pt file."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

    assert len(normalizers) > 0
    for name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        assert norm.weight_avg_sq.ndim == 2


def test_load_from_checkpoint_dir(tmp_path):
    """Load normalizers from a checkpoint directory containing optimizer.pt."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    torch.save(opt_state, checkpoint_dir / "optimizer.pt")

    normalizers = load_from_optimizer(model, str(checkpoint_dir))
    assert len(normalizers) > 0


def test_target_modules_filter(tmp_path):
    """Only layers in target_modules are loaded.

    Module names are relative to ``model.base_model`` -- the collector attaches
    hooks to ``model.base_model`` and looks up normalizers by that relative name
    (e.g. ``layers.0.self_attn.qkv_proj``, not ``model.layers.0...``), so
    ``load_from_optimizer`` keys and filters by the same relative names.
    """
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    base = getattr(model, "base_model", model)
    all_linear = {
        name for name, module in base.named_modules() if isinstance(module, nn.Linear)
    }
    subset = set(sorted(all_linear)[:2])

    normalizers = load_from_optimizer(model, str(opt_path), target_modules=subset)
    assert set(normalizers.keys()) == subset


def test_load_from_gpt2_conv1d_base_relative_keys(tmp_path):
    """GPT-2 stores attn/mlp weights as ``Conv1D`` (layout ``[in, out]``) under a
    ``transformer.`` prefix. load_from_optimizer must (a) key normalizers by the
    ``model.base_model``-relative name the collector looks up (``h.0.attn.c_attn``,
    not ``transformer.h.0...``) and (b) orient the Conv1D second moment into the
    collector's ``[out, in]`` layout.
    """
    from transformers.pytorch_utils import Conv1D

    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    base = model.base_model
    conv_names = sorted(n for n, m in base.named_modules() if isinstance(m, Conv1D))
    assert conv_names, "expected Conv1D layers in GPT-2"
    subset = set(conv_names[:2])

    normalizers = load_from_optimizer(model, str(opt_path), target_modules=subset)

    # Keys are base-relative (``transformer.`` stripped), matching the collector.
    assert set(normalizers.keys()) == subset
    for name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        module = base.get_submodule(name)
        out_f = module.nf  # Conv1D output features
        in_f = module.weight.shape[0]  # Conv1D param is [in, out]
        # Optimizer moment is stored [in, out]; must be transposed to [out, in].
        assert norm.weight_avg_sq.shape == (out_f, in_f)


def test_missing_optimizer_file(tmp_path):
    """Error when directory has no optimizer.pt."""
    model = _create_model()

    with pytest.raises(FileNotFoundError):
        load_from_optimizer(model, str(tmp_path))


# ---------------------------------------------------------------------------
# load_optimizer (local + Hub) tests
# ---------------------------------------------------------------------------


def test_load_optimizer_local_file(tmp_path):
    from bergson.utils.load_from_optimizer import load_optimizer

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(state, opt_path)

    loaded = load_optimizer(str(opt_path))
    assert "state" in loaded and "param_groups" in loaded


def test_load_optimizer_local_dir(tmp_path):
    from bergson.utils.load_from_optimizer import load_optimizer

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    torch.save(state, tmp_path / "optimizer.pt")

    loaded = load_optimizer(str(tmp_path))
    assert "state" in loaded


def test_load_optimizer_hub_dispatch(tmp_path, monkeypatch):
    """hf:// URIs should dispatch to hf_hub_download with parsed args."""
    from bergson.utils import load_from_optimizer as mod

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    cached = tmp_path / "optimizer.pt"
    torch.save(state, cached)

    calls = []

    def fake_download(repo_id, filename, revision=None, repo_type=None, **_):
        calls.append((repo_id, filename, revision, repo_type))
        return str(cached)

    monkeypatch.setattr(mod, "hf_hub_download", fake_download)

    cases = [
        ("hf://org/repo", ("org/repo", "optimizer.pt", None, "model")),
        ("hf://org/repo@rev", ("org/repo", "optimizer.pt", "rev", "model")),
        (
            "hf://org/repo/checkpoint-1",
            ("org/repo", "checkpoint-1/optimizer.pt", None, "model"),
        ),
        ("hf://org/repo/custom.pt", ("org/repo", "custom.pt", None, "model")),
        (
            "hf://org/repo@v2/sub/dir/optimizer.pth",
            ("org/repo", "sub/dir/optimizer.pth", "v2", "model"),
        ),
        (
            "hf://datasets/org/repo/optimizer.pt",
            ("org/repo", "optimizer.pt", None, "dataset"),
        ),
    ]
    for spec, expected in cases:
        calls.clear()
        mod.load_optimizer(spec)
        assert calls == [expected], f"{spec} -> {calls}"


def test_load_optimizer_invalid_spec():
    from bergson.utils.load_from_optimizer import load_optimizer

    with pytest.raises(FileNotFoundError):
        load_optimizer("not/a/local/path")


# ---------------------------------------------------------------------------
# get_optimizer_state_format / get_unfactored_second_moment unit tests
# ---------------------------------------------------------------------------


def test_get_optimizer_state_format_adam():
    state = {"exp_avg_sq": torch.zeros(2, 3), "step": torch.tensor(1)}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.UNFACTORED


def test_get_optimizer_state_format_adafactor():
    state = {"exp_avg_sq_row": torch.zeros(2), "exp_avg_sq_col": torch.zeros(3)}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.FACTORED


def test_get_optimizer_state_format_bnb_8bit_adam():
    state = {"__bnb_optimizer_quant_state__": {"state2": torch.zeros(8)}}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.UNFACTORED


def test_get_optimizer_state_format_empty_or_unknown_returns_none():
    # Empty (param registered but never stepped) and unknown formats both
    # return None so the main loop can skip without crashing.
    assert get_optimizer_state_format({}) is None
    assert get_optimizer_state_format({"step": torch.tensor(1)}) is None
    assert get_optimizer_state_format({"square_avg": torch.zeros(2)}) is None


def test_get_optimizer_state_format_non_dict_returns_none():
    # The isinstance guard means None / non-dicts return None instead of
    # raising "argument of type 'NoneType' is not iterable" on the `in` check.
    assert get_optimizer_state_format(None) is None
    assert get_optimizer_state_format("not a dict") is None


def test_get_unfactored_second_moment_adam_and_bnb():
    sq = torch.rand(2, 3)
    assert torch.equal(get_unfactored_second_moment({"exp_avg_sq": sq}), sq)

    bnb_sq = torch.rand(8)
    state = {"__bnb_optimizer_quant_state__": {"state2": bnb_sq}}
    assert torch.equal(get_unfactored_second_moment(state), bnb_sq)


# ---------------------------------------------------------------------------
# Bad / empty target_modules paths
# ---------------------------------------------------------------------------


def test_target_modules_no_overlap_raises(tmp_path):
    """If target_modules names don't match any param, no normalizers loaded."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(
            model, str(opt_path), target_modules={"definitely.not.a.real.module"}
        )


def test_target_modules_empty_set_raises(tmp_path):
    """An empty target_modules set rejects everything → assertion."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(model, str(opt_path), target_modules=set())


# ---------------------------------------------------------------------------
# Unrecognized / mixed-format states
# ---------------------------------------------------------------------------


def test_unrecognized_state_skipped_others_loaded(tmp_path):
    """A state entry with no recognised keys is skipped; the rest still load."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    # Replace the first param's state with an unknown-format dict.
    first_idx = next(iter(opt_state["state"]))
    opt_state["state"][first_idx] = {"step": torch.tensor(1)}

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

    # Some normalizers loaded — the unrecognised one was simply skipped.
    assert len(normalizers) > 0


# ---------------------------------------------------------------------------
# include_bias path
# ---------------------------------------------------------------------------


def test_include_bias_loads_bias_normalizer(tmp_path):
    """Bias second moments are attached when include_bias=True."""
    # Build a minimal model with a bias and craft optimizer state for both
    # weight and bias of the same Linear.
    model = nn.Sequential(nn.Linear(3, 4, bias=True))
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    for idx, (_name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    opt_state = {"state": state, "param_groups": param_groups}
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path), include_bias=True)  # type: ignore[arg-type]
    assert len(normalizers) == 1
    norm = next(iter(normalizers.values()))
    assert isinstance(norm, AdamNormalizer)
    assert norm.bias_avg_sq is not None
    assert norm.bias_avg_sq.shape == (4,)


def test_include_bias_false_leaves_bias_unset(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4, bias=True))
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    for idx, (_name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    opt_state = {"state": state, "param_groups": param_groups}
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path), include_bias=False)  # type: ignore[arg-type]
    assert len(normalizers) == 1
    norm = next(iter(normalizers.values()))
    assert isinstance(norm, AdamNormalizer)
    assert norm.bias_avg_sq is None


# ---------------------------------------------------------------------------
# PEFT path: adapter-suffixed target_modules must match
# ---------------------------------------------------------------------------


def _create_peft_model() -> PeftModel:
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["qkv_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    assert isinstance(model, PeftModel)
    return model


def _fake_optimizer_state_for_peft(peft_model):
    """Build optimizer state keyed by index, matching get_peft_model_state_dict
    order (which is what load_from_optimizer uses for PEFT models)."""
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    psd = get_peft_model_state_dict(peft_model)
    for idx, (_name, param) in enumerate(psd.items()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    return {"state": state, "param_groups": param_groups}


def test_load_from_peft_model_with_adapter_suffix(tmp_path):
    """Regression: PEFT module names from extract_peft_target_modules include
    the adapter suffix (``.default``); load_from_optimizer must produce
    matching keys, otherwise the target_modules filter rejects everything."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    target_modules = extract_peft_target_modules(model)
    assert any(name.endswith(".default") for name in target_modules)

    normalizers = load_from_optimizer(
        model, str(opt_path), target_modules=target_modules
    )

    # Every adapter-suffixed module should have a normalizer.
    assert set(normalizers.keys()) == target_modules
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)
        assert norm.weight_avg_sq.ndim == 2


def test_load_from_peft_model_without_target_modules(tmp_path):
    """target_modules=None on a PEFT model still loads every LoRA weight."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))
    # Every LoRA weight produces one normalizer; names carry the adapter
    # suffix because adapter_suffix is appended unconditionally for PEFT.
    assert len(normalizers) > 0
    for name in normalizers:
        assert name.endswith(".default")


def test_load_from_peft_strip_adapter_target_modules_misses(tmp_path):
    """If a caller passes target_modules WITHOUT the adapter suffix (the bug
    we just fixed), nothing matches and the assertion fires."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    target_modules = extract_peft_target_modules(model)
    stripped = {name.removesuffix(".default") for name in target_modules}

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(model, str(opt_path), target_modules=stripped)


# ---------------------------------------------------------------------------
# Integration tests with real training checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adam_checkpoint():
    """Load from a real AdamW training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adamw_torch")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint, looking entries up via
    # the group-aware index mapping (HF Trainer writes two param groups).
    index_to_name = optimizer_param_index_to_name(opt_state, model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    name_to_idx = {n: i for i, n in index_to_name.items()}
    for name, idx in name_to_idx.items():
        entry = opt_state["state"].get(idx)
        if (
            entry is not None
            and "exp_avg_sq" in entry
            and entry["exp_avg_sq"].ndim == 2
        ):
            assert tuple(entry["exp_avg_sq"].shape) == shapes[name]
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["exp_avg_sq"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdamNormalizer)
        torch.testing.assert_close(norm.weight_avg_sq.cpu(), raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adafactor_checkpoint():
    """Load from a real Adafactor training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adafactor")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdafactorNormalizer)

    # Verify loaded values match the raw checkpoint, looking entries up via
    # the group-aware index mapping (HF Trainer writes two param groups).
    index_to_name = optimizer_param_index_to_name(opt_state, model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    name_to_idx = {n: i for i, n in index_to_name.items()}
    for name, idx in name_to_idx.items():
        entry = opt_state["state"].get(idx)
        if (
            entry is not None
            and "exp_avg_sq" in entry
            and entry["exp_avg_sq"].ndim == 2
        ):
            assert tuple(entry["exp_avg_sq"].shape) == shapes[name]
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw_row = opt_state["state"][idx]["exp_avg_sq_row"]
        raw_col = opt_state["state"][idx]["exp_avg_sq_col"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdafactorNormalizer)
        torch.testing.assert_close(norm.row.cpu(), raw_row)
        torch.testing.assert_close(norm.col.cpu(), raw_col)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_8bit_adam_checkpoint():
    """Load from a real 8-bit Adam (bitsandbytes) training checkpoint."""
    ckpt_path, model = _train_checkpoint("adamw_bnb_8bit")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint, looking entries up via
    # the group-aware index mapping (HF Trainer writes two param groups).
    index_to_name = optimizer_param_index_to_name(opt_state, model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    name_to_idx = {n: i for i, n in index_to_name.items()}
    for name, idx in name_to_idx.items():
        entry = opt_state["state"].get(idx)
        if (
            entry is not None
            and "exp_avg_sq" in entry
            and entry["exp_avg_sq"].ndim == 2
        ):
            assert tuple(entry["exp_avg_sq"].shape) == shapes[name]
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["__bnb_optimizer_quant_state__"]["state2"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdamNormalizer)
        torch.testing.assert_close(norm.weight_avg_sq.cpu(), raw)


def test_optimizer_index_mapping_hf_decay_groups():
    """The two-group (HF Trainer decay/no-decay) index reconstruction maps
    every state entry back to the right param — verified by tensor identity
    against a real torch AdamW built the way HF Trainer builds it."""
    model = _create_model()
    decay_names = {
        n for n in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in n
    }
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in named if n in decay_names], "weight_decay": 0.01},
            {
                "params": [p for n, p in named if n not in decay_names],
                "weight_decay": 0.0,
            },
        ],
        lr=1e-3,
    )
    # One step so every param has state.
    for _, p in named:
        p.grad = torch.rand_like(p)
    optimizer.step()

    opt_state = optimizer.state_dict()
    mapping = optimizer_param_index_to_name(opt_state, model)

    params_by_name = dict(named)
    assert len(mapping) == len(named)
    for idx, name in mapping.items():
        expected = optimizer.state[params_by_name[name]]["exp_avg_sq"]
        torch.testing.assert_close(opt_state["state"][idx]["exp_avg_sq"], expected)


def test_square_conv1d_moment_is_transposed(tmp_path):
    """A SQUARE HF Conv1D weight is stored ``[in, out]`` but its shape cannot
    reveal that — orientation must come from the module class (LayerAdapter),
    or the moment is silently used transposed."""
    from transformers.pytorch_utils import Conv1D

    class SquareConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Conv1D(4, 4)  # weight [in=4, out=4]
            self.fc = nn.Linear(4, 6)

    model = SquareConvModel()
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))
    names = [n for n, _ in model.named_parameters()]
    proj_moment = opt_state["state"][names.index("proj.weight")]["exp_avg_sq"]
    fc_moment = opt_state["state"][names.index("fc.weight")]["exp_avg_sq"]

    # Conv1D: stored [in, out] -> normalizer holds the transpose; Linear: as-is.
    torch.testing.assert_close(normalizers["proj"].weight_avg_sq, proj_moment.T)
    torch.testing.assert_close(normalizers["fc"].weight_avg_sq, fc_moment)


def test_load_peft_hf_two_group_checkpoint(tmp_path):
    """PEFT + HF Trainer checkpoints: the optimizer's decay/no-decay groups
    reorder state indices relative to the serialized PEFT param list, so the
    mapping must be reconstructed group-aware. bias="lora_only" interleaves
    trainable (no-decay) biases with the (decay) LoRA weights, which breaks
    the old positional convention."""
    # tiny-gpt2's Conv1D layers have biases, so bias="lora_only" yields
    # trainable (no-decay) bias params.
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
            bias="lora_only",
            task_type="CAUSAL_LM",
        ),
    )

    # Build the optimizer exactly as HF Trainer does: [decay, no_decay].
    decay_names = {
        n for n in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in n
    }
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert any(n not in decay_names for n, _ in named), "need no-decay params"
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in named if n in decay_names], "weight_decay": 0.01},
            {
                "params": [p for n, p in named if n not in decay_names],
                "weight_decay": 0.0,
            },
        ],
        lr=1e-3,
    )
    for _, p in named:
        p.grad = torch.rand_like(p)
    optimizer.step()

    opt_path = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))
    assert normalizers, "expected LoRA normalizers"
    for module_name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        weight = model.get_submodule(f"base_model.{module_name}").weight
        expected = optimizer.state[weight]["exp_avg_sq"]
        torch.testing.assert_close(norm.weight_avg_sq, expected)
