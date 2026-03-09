Always test your changes by running the appropriate script or CLI command. Never complete a task without testing your changes until the script or CLI command runs without issues for 3 minutes+ (at minimum). If you find an error unrelated to your task, at minimum quote the exact error back to me when you have completed your task and offer to investigate and fix it.

## Project Structure and Conventions

Keep __main__.py clean - it should primarily provide documentation and routing for the available CLI commands and their configs.

Consider writing a new library file if you add a standalone, complex feature used in more than one place.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

Never save logs, scripts, and other random development into the root of a project. Create an appropriate directory such as runs/ or scripts/ and add it to the .gitignore.

torch.cuda.empty_cache() doesn't do what you hope it will do - don't use it.

Put imports at the top of the file unless you have a good reason to do otherwise.

# Development

You can call CLI commands without prefixing `python -m`, like `bergson build`.

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

Run bash commands in the dedicated tmux pane named "claude" if it is available.

Don't keep default run path values inside low level code - if a module calls another module, the higher level module should always pass through inject a base path.

Don't save data to a directory that is not in the gitignore - especially the data/ directory.

Don't remove large datasets from the HF cache without asking.

When a bug is reported, start by writing a test that reproduces the bug. Then fix the bug and prove it with a passing test.

### Tests

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

### Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate`.

You can pull secrets from .env.
