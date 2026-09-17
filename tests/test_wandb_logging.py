import pytest

from bergson.utils.logging import wandb_log_fn

wandb = pytest.importorskip("wandb")


def test_offline_init_without_api_key(tmp_path, monkeypatch):
    """Without an API key the log_fn logs to disk via an offline run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    # Point wandb away from any real ~/.netrc credentials.
    monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    assert wandb.run is None

    with pytest.warns(UserWarning, match="offline"):
        log_fn = wandb_log_fn("bergson-test")
    try:
        assert wandb.run is not None
        assert wandb.run.offline
        log_fn(0, 1.0)
        log_fn(1, 0.5)
    finally:
        wandb.finish()

    offline_runs = list(tmp_path.glob("wandb/offline-run-*"))
    assert offline_runs, "expected an offline run directory on disk"


def test_disabled_mode_is_noop(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    log_fn = wandb_log_fn("bergson-test")
    log_fn(0, 1.0)
    assert wandb.run is None
