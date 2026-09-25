"""SOURCE reading its hyperparameters off a bergson training run.

The rule these pin: derivation is a *fallback*. Anything set explicitly wins, so
checkpoints from another trainer keep working exactly as before.
"""

import json

import pytest
import torch
import yaml

from bergson.approx_unrolling.approx_unrolling_math import (
    _checkpoint_step,
    compute_lr_times_steps_per_segment,
)
from bergson.approx_unrolling.train_cfg_io import (
    derive_momentum,
    load_training_config,
    resolve,
)
from bergson.config.config import ApproxUnrollingConfig, TrainingConfig
from bergson.config.config_io import save_run_config
from bergson.magic.trainer import LR_HISTORY_FILENAME, write_lr_history


def _run_dir(tmp_path, **overrides):
    """A bergson run directory with a config.yaml.

    Written by ``save_run_config`` itself so the loader is pinned against the
    real on-disk shape rather than a hand-rolled approximation of it."""
    save_run_config(TrainingConfig(run_path=str(tmp_path), **overrides), tmp_path)
    return tmp_path


# ── step parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("checkpoint-120", 120),  # HF Trainer
        ("step_7.ckpt", 7),  # bergson trainer, native
        ("step_7", 7),  # bergson trainer, exported
        ("42", 42),  # bare step dir
    ],
)
def test_checkpoint_step_accepts_both_conventions(name, expected):
    assert _checkpoint_step(f"/runs/x/{name}") == expected


# ── momentum derivation ─────────────────────────────────────────────────────


def test_derive_momentum_sgd_default_is_not_zero():
    """The default adam_beta1 is 0.95, so assuming 0.0 is a 20x lr*steps error."""
    cfg = TrainingConfig(run_path="/tmp/x", optimizer="sgd")
    assert derive_momentum(cfg) == pytest.approx(0.95)
    assert 1.0 / (1.0 - derive_momentum(cfg)) == pytest.approx(20.0)


def test_derive_momentum_adamw_is_zero():
    """AdamW's own preconditioner accounts for its first moment."""
    cfg = TrainingConfig(run_path="/tmp/x", optimizer="adamw", adam_beta1=0.95)
    assert derive_momentum(cfg) == 0.0


def test_derive_momentum_muon_warns_and_defaults(caplog):
    cfg = TrainingConfig(run_path="/tmp/x", optimizer="muon")
    assert derive_momentum(cfg) == 0.0


# ── resolution precedence ───────────────────────────────────────────────────


def test_resolve_is_noop_without_trainer_run():
    """Configs for other trainers pass through untouched."""
    cfg = ApproxUnrollingConfig(checkpoints=["a", "b"], model_path="gpt2")
    out = resolve(cfg)
    assert out.checkpoints == ["a", "b"]
    assert out.model_path == "gpt2"
    assert out.momentum == 0.0  # sentinel normalized, nothing derived


def test_resolve_fills_momentum_and_model_from_run(tmp_path):
    run = _run_dir(
        tmp_path, optimizer="sgd", adam_beta1=0.9, model="EleutherAI/pythia-14m"
    )
    ckpt = run / "exported" / "checkpoint-0"
    ckpt.mkdir(parents=True)
    cfg = ApproxUnrollingConfig(checkpoints=[str(ckpt)])

    out = resolve(cfg)

    assert out.momentum == 0.9
    assert out.model_path == "EleutherAI/pythia-14m"


def test_explicit_zero_momentum_is_respected(tmp_path):
    """0.0 is a real value, not 'unset' -- it must survive derivation."""
    run = _run_dir(tmp_path, optimizer="sgd", adam_beta1=0.9)
    ckpt = run / "checkpoint-0"
    ckpt.mkdir()
    cfg = ApproxUnrollingConfig(checkpoints=[str(ckpt)], momentum=0.0)
    assert resolve(cfg).momentum == 0.0


def test_explicit_model_path_wins(tmp_path):
    run = _run_dir(tmp_path, model="EleutherAI/pythia-14m")
    ckpt = run / "checkpoint-0"
    ckpt.mkdir()
    cfg = ApproxUnrollingConfig(checkpoints=[str(ckpt)], model_path="gpt2")
    assert resolve(cfg).model_path == "gpt2"


def test_missing_config_yaml_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="bergson run directory"):
        load_training_config(tmp_path)


def test_load_reads_the_saved_steps_document(tmp_path):
    """save_run_config wraps the step list in {steps, metadata}."""
    _run_dir(tmp_path, optimizer="adamw", adam_beta2=0.98)

    doc = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert set(doc) == {"steps", "metadata"}
    assert load_training_config(tmp_path).adam_beta2 == pytest.approx(0.98)


def test_load_still_reads_a_bare_step_list(tmp_path):
    """Runs saved before the {steps, metadata} wrapper stay readable."""
    cfg = TrainingConfig(run_path=str(tmp_path), optimizer="sgd", adam_beta1=0.8)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump([{"magic": cfg.to_dict()}]))

    assert load_training_config(tmp_path).adam_beta1 == pytest.approx(0.8)


# ── LR history ──────────────────────────────────────────────────────────────


def test_write_lr_history_matches_hf_log_history_shape(tmp_path):
    """Written in HF's shape so SOURCE's existing reader picks it up unchanged."""
    path = write_lr_history(tmp_path, lambda step: 1e-4 * (step + 1), 3)

    assert path.name == LR_HISTORY_FILENAME
    entries = json.loads(path.read_text())
    assert entries == [
        {"step": 0, "learning_rate": pytest.approx(1e-4)},
        {"step": 1, "learning_rate": pytest.approx(2e-4)},
        {"step": 2, "learning_rate": pytest.approx(3e-4)},
    ]


def test_lr_times_steps_reads_bergson_history(tmp_path):
    """End to end: a written history drives lr*K without any HF artifacts."""
    export = tmp_path / "exported"
    export.mkdir()
    for step in (2, 4):
        (export / f"checkpoint-{step}").mkdir()
    write_lr_history(export, lambda step: 1e-3, 5)

    cfg = ApproxUnrollingConfig(
        checkpoints=[str(export / "checkpoint-2"), str(export / "checkpoint-4")],
        segments=2,
        momentum=0.0,
    )

    # Segment 1 covers steps 1..2, segment 2 covers 3..4 -> two steps each.
    assert compute_lr_times_steps_per_segment(cfg) == [
        pytest.approx(2e-3),
        pytest.approx(2e-3),
    ]


def test_unset_momentum_defaults_to_no_scaling():
    """Without a trainer_run, behaviour is exactly as before this change."""
    cfg = ApproxUnrollingConfig(
        checkpoints=["a"], segments=1, lr_list=[1e-3], step_size_list=[10]
    )
    assert compute_lr_times_steps_per_segment(cfg) == [pytest.approx(1e-2)]


# ── export end to end ───────────────────────────────────────────────────────


def test_lr_history_read_from_the_run_not_the_export(tmp_path):
    """One logical location: the trainer writes it beside its own checkpoints
    and the reader finds it there, so nothing has to be copied on export."""
    run = _run_dir(tmp_path)
    write_lr_history(run / "checkpoints", lambda step: 1e-3, 5)

    export = tmp_path / "exported"
    export.mkdir()
    for step in (2, 4):
        (export / f"checkpoint-{step}").mkdir()
    assert not (export / LR_HISTORY_FILENAME).exists()

    cfg = ApproxUnrollingConfig(
        checkpoints=[str(export / "checkpoint-2"), str(export / "checkpoint-4")],
        segments=2,
        momentum=0.0,
    )
    assert compute_lr_times_steps_per_segment(cfg) == [
        pytest.approx(2e-3),
        pytest.approx(2e-3),
    ]


def test_dcp_tolerates_optimizer_state_inside_the_checkpoint(tmp_path):
    """The layout rests on this: an optimizer.pt inside step_<i>.ckpt/ must not
    disturb DCP's own load or a resumed run, and must survive a re-save."""
    import torchopt
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from bergson.magic.data_stream import DataStream
    from bergson.magic.trainer import Trainer
    from bergson.utils.trainer_export import OPTIMIZER_STATE_FILE

    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")

    def fresh():
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(
            config, dtype=torch.float32, attn_implementation="eager"
        )
        m.requires_grad_(True)
        return m

    n = 4
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * n, "labels": [[1, 2, 3, 4]] * n}
    )
    stream = DataStream(ds, batch_size=1, device="cpu")
    opt = torchopt.sgd(lambda step: 1e-4, momentum=0.95)

    save_dir = tmp_path / "checkpoints"
    trainer, state = Trainer.initialize(fresh(), opt)
    final = trainer.train(
        state, stream, inplace=True, save_dir=str(save_dir), save_mode="all"
    )

    ckpt = save_dir / "step_2.ckpt"
    torch.save(
        {"state": {0: {"exp_avg_sq": torch.ones(2, 2)}}}, ckpt / OPTIMIZER_STATE_FILE
    )

    model = fresh()
    _, loaded = Trainer.initialize(model, opt)
    loaded.load(str(ckpt))

    trainer2, state2 = Trainer.initialize(fresh(), opt)
    resumed = trainer2.train(
        state2,
        stream,
        inplace=True,
        save_dir=str(save_dir),
        save_mode="all",
        resume=True,
    )
    for k in final.params:
        torch.testing.assert_close(resumed.params[k], final.params[k])

    blob = torch.load(ckpt / OPTIMIZER_STATE_FILE, weights_only=False)
    assert blob["state"][0]["exp_avg_sq"].shape == (2, 2)


def test_trainer_writes_optimizer_state_inside_each_checkpoint(tmp_path):
    """optimizer_cfg puts each step's second moments in that step's own dir."""
    import torchopt
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from bergson.magic.data_stream import DataStream
    from bergson.magic.trainer import Trainer
    from bergson.utils.load_from_optimizer import load_optimizer
    from bergson.utils.trainer_export import (
        sorted_dcp_checkpoints,
    )

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float32, attn_implementation="eager"
    )
    model.requires_grad_(True)

    n = 3
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * n, "labels": [[1, 2, 3, 4]] * n}
    )
    stream = DataStream(ds, batch_size=1, device="cpu")
    trainer, state = Trainer.initialize(model, torchopt.adamw(lambda step: 1e-4))

    save_dir = tmp_path / "checkpoints"
    trainer.train(
        state,
        stream,
        inplace=True,
        save_dir=str(save_dir),
        save_mode="all",
        optimizer_cfg=dict(betas=(0.9, 0.999), eps=1e-8, eps_root=0.0),
    )

    # No loose siblings: the state lives inside the checkpoint it belongs to.
    assert not list(save_dir.glob("step_*.optimizer.pt"))

    for step, ckpt in sorted_dcp_checkpoints(save_dir):
        blob = load_optimizer(str(ckpt))
        assert blob["state"], f"step {step} has no second moments"
        entry = next(iter(blob["state"].values()))
        recorded = entry["step"]
        assert int(recorded.item() if torch.is_tensor(recorded) else recorded) == step
        assert blob["param_groups"][0]["betas"] == (0.9, 0.999)


def test_trainer_run_inferred_from_run_root_checkpoints(tmp_path):
    run = _run_dir(tmp_path, optimizer="sgd", adam_beta1=0.8)
    ckpt = run / "checkpoint-1"
    ckpt.mkdir()

    assert resolve(ApproxUnrollingConfig(checkpoints=[str(ckpt)])).momentum == 0.8


@pytest.mark.parametrize(
    "value,expected",
    [(True, "last"), (False, "none"), ("all", "all")],
)
def test_save_optimizer_state_accepts_old_booleans(value, expected):
    """`true` used to mean "write the final state"; keep that meaning."""
    cfg = TrainingConfig(run_path="/tmp/x", save_optimizer_state=value)
    assert cfg.save_optimizer_state == expected


def test_dcp_checkpoints_resolve_to_exported_dirs(tmp_path, monkeypatch):
    """Raw step_<i>.ckpt paths map to exported/checkpoint-<i>, reusing an
    existing export and exporting the rest in one call per run."""
    import bergson.utils.trainer_export as te

    for step in (10, 20):
        (tmp_path / "checkpoints" / f"step_{step}.ckpt").mkdir(parents=True)
    (tmp_path / "exported" / "checkpoint-10").mkdir(parents=True)

    calls = []
    monkeypatch.setattr(
        te, "export_checkpoints", lambda run, steps=None, **kw: calls.append(steps)
    )
    cfg = resolve(
        ApproxUnrollingConfig(
            checkpoints=[
                str(tmp_path / "checkpoints" / f"step_{s}.ckpt") for s in (10, 20)
            ]
        )
    )
    assert cfg.checkpoints == [
        str(tmp_path / "exported" / f"checkpoint-{s}") for s in (10, 20)
    ]
    assert calls == [[20]]


def test_export_checkpoints_end_to_end(tmp_path):
    """The function itself: reads the run's config.yaml, builds the model via
    prepare_trainer, and writes loadable checkpoint-<i>/ dirs with their
    optimizer state -- the wiring the round-trip test does not cover."""
    import torchopt
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from bergson.magic.data_stream import DataStream
    from bergson.magic.trainer import Trainer
    from bergson.utils.load_from_optimizer import load_optimizer
    from bergson.utils.trainer_export import (
        EXPORT_DIRNAME,
        OPTIMIZER_STATE_FILE,
        export_checkpoints,
    )

    model_name = "EleutherAI/pythia-14m"
    run = _run_dir(tmp_path, model=model_name, optimizer="adamw")

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float32, attn_implementation="eager"
    )
    model.requires_grad_(True)

    n = 3
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * n, "labels": [[1, 2, 3, 4]] * n}
    )
    stream = DataStream(ds, batch_size=1, device="cpu")
    trainer, state = Trainer.initialize(model, torchopt.adamw(lambda step: 1e-4))
    trainer.train(
        state,
        stream,
        inplace=True,
        save_dir=str(run / "checkpoints"),
        save_mode="all",
        optimizer_cfg=dict(betas=(0.9, 0.999), eps=1e-8, eps_root=0.0),
    )

    exported = export_checkpoints(run, steps=[0, 2])

    assert [p.name for p in exported] == ["checkpoint-0", "checkpoint-2"]
    assert exported[0].parent == run / EXPORT_DIRNAME

    for dst in exported:
        AutoModelForCausalLM.from_pretrained(str(dst))
        assert load_optimizer(str(dst))["state"], f"{dst} lost its optimizer state"
        assert (dst / OPTIMIZER_STATE_FILE).is_file()

    # And the exported dirs are what resolve() then picks up.
    out = resolve(ApproxUnrollingConfig(checkpoints=[str(p) for p in exported]))
    assert out.model_path == model_name
    assert out.momentum == 0.0  # adamw


def test_resolve_ignores_a_non_training_config(tmp_path):
    """A checkpoint dir's sibling config.yaml may belong to an attribution run.

    ``infer_trainer_run`` only checks that a config.yaml exists, so ``resolve``
    has to tolerate one it cannot read as a TrainingConfig.
    """
    run = tmp_path / "attribution_run"
    (run / "models" / "step_1").mkdir(parents=True)
    (run / "config.yaml").write_text("steps:\n- approxunrolling:\n    index_cfg: {}\n")

    cfg = ApproxUnrollingConfig(checkpoints=[str(run / "models" / "step_1")])
    resolved = resolve(cfg)

    assert resolved.momentum == 0.0
    assert resolved.model_path is None
