"""Regression test: verify all CLI subcommands can construct their argument parser."""

import subprocess

import pytest

from .cli_command import bergson_cmd, bergson_env

SUBCOMMANDS = [
    "build",
    "ekfac",
    "hessian",
    "magic",
    "query",
    "reduce",
    "score",
    "trackstar",
    "test_model_configuration",
]


@pytest.fixture(scope="module")
def help_results() -> dict[str, subprocess.CompletedProcess]:
    """Run every subcommand's --help concurrently.

    Each invocation spends several seconds importing torch, so running them
    sequentially costs ~1 minute; concurrently they take as long as the slowest.
    """
    procs = {
        cmd: subprocess.Popen(
            bergson_cmd(cmd, "--help"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=bergson_env(),
        )
        for cmd in SUBCOMMANDS
    }
    results = {}
    for cmd, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=120)
        results[cmd] = subprocess.CompletedProcess(
            proc.args, proc.returncode, stdout, stderr
        )
    return results


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_cli_help(cmd, help_results):
    """Each subcommand should produce --help output without crashing."""
    result = help_results[cmd]
    assert result.returncode == 0, f"bergson {cmd} --help failed:\n{result.stderr}"
