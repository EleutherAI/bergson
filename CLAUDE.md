https://developers.google.com/style/timeless-documentation

Always test your changes. Ensure your scripts or CLI commands run without issues for 3 minutes+ (at minimum). If you find an error unrelated to your task, at minimum communicate the exact error when you have completed your task and offer to investigate and fix it.

Compare comments to human-written examples in the codebase and match style: imperative summary line, second paragraph only for a mechanism you can't derive.

Don't coin terms or use jargon. Don't write code or comments containing specialized terms not already in the codebase without permission.

## Project Structure and Conventions

Consider writing a new CLI tool if you add a standalone, complex feature used in more than one place.

Put imports at the top of the file unless you have a very good reason to do otherwise.

# Development

Never run an unbounded background wait. Any run_in_background waiter/poll loop must have a hard deadline that makes it exit and return control, and a process-liveness check (kill -0/pgrep) as a terminal condition. Prefer the Monitor tool for watches.

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

Ensure logs, scripts, data, and other files not ready for production are added to directories present in the .gitignore.

Don't remove large datasets from the HF cache without asking.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

### Tests

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

Accelerate tests with `pytest -n 8 --dist loadgroup` or similar.

### Environment Setup

You can pull secrets from .env.
