Always test your changes by running the appropriate script or CLI command. Never complete a task without testing your changes until the script or CLI command runs without issues. If it's a long-running script let it run for at least a few iterations of the main loop. If you find an error unrelated to your task, at minimum quote the exact error back to me when you have completed your task and offer to investigate and fix it.

## Project Structure and Conventions

Keep __main__.py clean - it should primarily provide documentation and routing for the available CLI commands and their configs.

Consider writing a new library file if you add a standalone, complex feature used in more than one place.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

Never save logs, scripts, and other random development into the root of a project. Create an appropriate directory such as runs/ or scripts/ and add it to the .gitignore.

torch.cuda.empty_cache() doesn't do what you hope it will do - don't use it.

Put imports at the top of the file unless you have a good reason to do otherwise.

# Development

Never use try/except blocks - fail fast, fail explicitly.

Never use "fallbacks".

Do not write lines longer than 88 characters.

Don't use ALL CAPS unless it's proper English (e.g. an acronym).

Don't keep default run path values inside low level code - if a module calls another module, the higher level module should always pass through inject a base path.

Don't save data to a directory that is not in the .gitignore - especially the data/ directory.

Don't remove large datasets from the HF cache without asking.

You can call CLI commands without prefixing `python -m`, like `bergson build`.

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

Run bash commands in the dedicated tmux pane named "claude" if it is available.

Don't betray lineage. An example of betraying lineage is duplicating a file, making changes in the duplicate, then calling it "foo_fixed" rather than "foo". The correct action here would be to commit the file to git and then modify it directly. Another example of betraying lineage is adding a RoundButton to a module containing a Button but not updating the original Button to be called RectangleButton. This lack of action betrays that the rectangular button came first.

If you think some data files (e.g. CSVs) have been invalidated but you're not 100% sure, you can add them to a .gitignore'd archive directory along with an equivalentally named markdown file explaining the context.

File names always use snake case - in_memory, not inmemory.

When writing files to disk python scripts should choose their own filenames but be provided with their file paths.

### Documentation

Do not mark documentation for code that has been removed as deprecated - simply remove the documentation.

No context leakage: do not write code or comments that link features to the specific experiment for which the feature was developed, unless it's only useful for that particular experiment. Be as generic as is correctly possible and not more.

### Tests

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

### Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate`.

You can pull secrets from .env.
