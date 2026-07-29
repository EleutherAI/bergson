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
from bergson.approx_unrolling.trainer_run import (
    LR_HISTORY_FILENAME,
    derive_momentum,
    load_training_config,
    resolve,
    write_lr_history,
)
from bergson.config.config import ApproxUnrollingConfig, TrainingConfig


def _run_dir(tmp_path, **overrides):
    """A bergson run directory with a config.yaml, as save_run_config writes it."""
    cfg = TrainingConfig(run_path=str(tmp_path), **overrides)
    payload = {"magic": cfg.to_dict()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump([payload]))
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


def test_checkpoint_step_still_rejects_junk():
    with pytest.raises(ValueError, match="Cannot infer a training step"):
        _checkpoint_step("/runs/x/final-model")


# ── momentum derivation ─────────────────────────────────────────────────────


def test_derive_momentum_sgd_uses_adam_beta1():
    """bergson's SGD passes adam_beta1 as torchopt.sgd's momentum."""
    cfg = TrainingConfig(run_path="/tmp/x", optimizer="sgd", adam_beta1=0.9)
    assert derive_momentum(cfg) == 0.9


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
    cfg = ApproxUnrollingConfig(trainer_run=str(run), checkpoints=["a", "b"])

    out = resolve(cfg)

    assert out.momentum == 0.9
    assert out.model_path == "EleutherAI/pythia-14m"


def test_explicit_momentum_wins_over_run(tmp_path):
    """A user training elsewhere must be able to override what the run says."""
    run = _run_dir(tmp_path, optimizer="sgd", adam_beta1=0.9)
    cfg = ApproxUnrollingConfig(trainer_run=str(run), checkpoints=["a"], momentum=0.5)
    assert resolve(cfg).momentum == 0.5


def test_explicit_zero_momentum_is_respected(tmp_path):
    """0.0 is a real value, not 'unset' -- it must survive derivation."""
    run = _run_dir(tmp_path, optimizer="sgd", adam_beta1=0.9)
    cfg = ApproxUnrollingConfig(trainer_run=str(run), checkpoints=["a"], momentum=0.0)
    assert resolve(cfg).momentum == 0.0


def test_explicit_model_path_wins(tmp_path):
    run = _run_dir(tmp_path, model="EleutherAI/pythia-14m")
    cfg = ApproxUnrollingConfig(
        trainer_run=str(run), checkpoints=["a"], model_path="gpt2"
    )
    assert resolve(cfg).model_path == "gpt2"


def test_explicit_checkpoints_win(tmp_path):
    run = _run_dir(tmp_path)
    (run / "checkpoint-0").mkdir()
    cfg = ApproxUnrollingConfig(trainer_run=str(run), checkpoints=["mine"])
    assert resolve(cfg).checkpoints == ["mine"]


def test_resolve_discovers_exported_checkpoints_in_order(tmp_path):
    run = _run_dir(tmp_path)
    for step in (0, 10, 2):
        (run / f"checkpoint-{step}").mkdir()

    out = resolve(ApproxUnrollingConfig(trainer_run=str(run)))

    assert [p.split("-")[-1] for p in out.checkpoints] == ["0", "2", "10"]


def test_unexported_run_names_the_export_step(tmp_path):
    """Native DCP checkpoints can't be loaded by from_pretrained; say so."""
    run = _run_dir(tmp_path)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "step_0.ckpt").mkdir()

    with pytest.raises(FileNotFoundError, match="export_checkpoints"):
        resolve(ApproxUnrollingConfig(trainer_run=str(run)))


def test_missing_config_yaml_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="bergson run directory"):
        load_training_config(tmp_path)


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


def test_momentum_scales_lr_times_steps(tmp_path):
    """The 1/(1-beta) terminal-velocity factor is what the SGD fix is about."""
    cfg = ApproxUnrollingConfig(
        checkpoints=["a", "b"],
        segments=2,
        lr_list=[1e-3, 1e-3],
        step_size_list=[10, 10],
        momentum=0.0,
    )
    baseline = compute_lr_times_steps_per_segment(cfg)

    cfg.momentum = 0.95
    scaled = compute_lr_times_steps_per_segment(cfg)

    assert scaled == [pytest.approx(20 * b) for b in baseline]


def test_momentum_out_of_range_is_rejected():
    cfg = ApproxUnrollingConfig(
        checkpoints=["a"], segments=1, lr_list=[1e-3], step_size_list=[1], momentum=1.0
    )
    with pytest.raises(ValueError, match="momentum must be in"):
        compute_lr_times_steps_per_segment(cfg)


def test_unset_momentum_defaults_to_no_scaling():
    """Without a trainer_run, behaviour is exactly as before this change."""
    cfg = ApproxUnrollingConfig(
        checkpoints=["a"], segments=1, lr_list=[1e-3], step_size_list=[10]
    )
    assert compute_lr_times_steps_per_segment(cfg) == [pytest.approx(1e-2)]


# ── export end to end ───────────────────────────────────────────────────────


def test_export_round_trips_checkpoint_weights(tmp_path):
    """A DCP checkpoint must survive export as a from_pretrained-loadable model.

    SOURCE loads every checkpoint with from_pretrained, so an export that lost
    or mangled weights would silently attribute the wrong trajectory.
    """
    import torchopt
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from bergson.magic.data_stream import DataStream
    from bergson.magic.trainer import Trainer
    from bergson.utils.trainer_export import sorted_dcp_checkpoints

    torch.manual_seed(0)
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

    trainer, state = Trainer.initialize(fresh(), opt)
    save_dir = tmp_path / "checkpoints"
    trainer.train(state, stream, inplace=True, save_dir=str(save_dir), save_mode="all")

    found = sorted_dcp_checkpoints(save_dir)
    assert [s for s, _ in found] == list(range(n))

    # Reload the last checkpoint and export it, as export_checkpoints does.
    model = fresh()
    _, loaded = Trainer.initialize(model, opt)
    loaded.load(str(found[-1][1]))

    out = tmp_path / "checkpoint-3"
    with loaded.activate(model), torch.no_grad():
        model.save_pretrained(str(out), safe_serialization=True)
        reference = {k: v.detach().clone() for k, v in model.named_parameters()}

    reloaded = AutoModelForCausalLM.from_pretrained(str(out))
    got = dict(reloaded.named_parameters())
    for name, ref in reference.items():
        torch.testing.assert_close(got[name], ref, atol=0, rtol=0)


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
        trainer_run=str(run),
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
