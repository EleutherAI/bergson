import os
import warnings
from collections.abc import Callable

os.environ.setdefault("WANDB__SERVICE_WAIT", "30")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")


def wandb_log_fn(
    project: str, config: dict | None = None, **init_kwargs
) -> Callable[[int, float], None]:
    """Create a log_fn callback that logs loss to Weights & Biases.

    Usage with Trainer.train()::

        log_fn = wandb_log_fn("my-project", config={"lr": 1e-4})
        trainer.train(state, data, log_fn=log_fn)
    """
    import wandb  # type: ignore[reportMissingImports]

    def _noop(step: int, loss: float) -> None: ...

    if not wandb.run:
        try:
            wandb.init(project=project, config=config, **init_kwargs)
        except Exception as e:
            warnings.warn(
                f"wandb.init failed ({type(e).__name__}: {e}); "
                "continuing without wandb logging."
            )
            return _noop

    def log_fn(step: int, loss: float):
        wandb.log({"train/loss": loss}, step=step)

    return log_fn
