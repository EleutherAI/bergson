import os
import warnings
from collections.abc import Callable

os.environ.setdefault("WANDB__SERVICE_WAIT", "30")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")


def _api_key_available(wandb) -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return bool(wandb.api.api_key)
    except Exception:
        return False


def wandb_log_fn(
    project: str, config: dict | None = None, **init_kwargs
) -> Callable[[int, float], None]:
    """Create a log_fn callback that logs loss to Weights & Biases.

    Usage with Trainer.train()::

        log_fn = wandb_log_fn("my-project", config={"lr": 1e-4})
        trainer.train(state, data, log_fn=log_fn)

    When no API key is available and ``WANDB_MODE`` is unset, the run is
    logged to a local wandb directory.

    Logging degrades to a no-op (without importing wandb) when
    ``WANDB_MODE=disabled`` is set, or when wandb is not installed. This keeps
    a MAGIC run that has ``wandb_project`` set from crashing post-training just
    because wandb is missing or explicitly disabled.
    """

    def _noop(step: int, loss: float) -> None: ...

    if os.environ.get("WANDB_MODE", "").strip().lower() == "disabled":
        return _noop

    try:
        import wandb  # type: ignore[reportMissingImports]
    except ImportError:
        warnings.warn(
            "wandb is not installed; continuing without wandb logging. "
            "Install wandb or set WANDB_MODE=disabled to silence this warning.",
            stacklevel=2,
        )
        return _noop

    if not wandb.run:
        if (
            "mode" not in init_kwargs
            and not os.environ.get("WANDB_MODE")
            and not _api_key_available(wandb)
        ):
            warnings.warn(
                "No W&B API key found; logging offline to the local wandb "
                "directory. Upload later with `wandb sync`.",
                stacklevel=2,
            )
            init_kwargs["mode"] = "offline"

        try:
            wandb.init(project=project, config=config, **init_kwargs)
        except Exception as e:
            # An online init can fail without network or credentials; offline
            # writes to disk and needs neither.
            if init_kwargs.get("mode") != "offline":
                init_kwargs["mode"] = "offline"
                try:
                    wandb.init(project=project, config=config, **init_kwargs)
                    warnings.warn(
                        f"wandb.init failed ({type(e).__name__}: {e}); "
                        "logging offline to the local wandb directory instead. "
                        "Upload later with `wandb sync`."
                    )
                except Exception as e2:
                    warnings.warn(
                        f"wandb.init failed offline as well "
                        f"({type(e2).__name__}: {e2}); "
                        "continuing without wandb logging."
                    )
                    return _noop
            else:
                warnings.warn(
                    f"wandb.init failed ({type(e).__name__}: {e}); "
                    "continuing without wandb logging."
                )
                return _noop

    def log_fn(step: int, loss: float):
        wandb.log({"train/loss": loss}, step=step)

    return log_fn
