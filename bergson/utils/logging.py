from collections.abc import Callable


def wandb_log_fn(
    project: str, config: dict | None = None, **init_kwargs
) -> Callable[[int, float], None]:
    """Create a log_fn callback that logs loss to Weights & Biases.

    Usage with Trainer.train()::

        log_fn = wandb_log_fn("my-project", config={"lr": 1e-4})
        trainer.train(state, data, log_fn=log_fn)
    """
    import wandb  # type: ignore[reportMissingImports]

    if not wandb.run:
        wandb.init(project=project, config=config, **init_kwargs)

    def log_fn(step: int, loss: float):
        wandb.log({"train/loss": loss}, step=step)

    return log_fn
