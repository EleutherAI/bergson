"""Regression test: verify all CLI subcommands can construct their argument parser."""

import subprocess

import pytest

SUBCOMMANDS = [
    "build",
    "ekfac",
    "hessian",
    "magic",
    "preconditioners",
    "query",
    "reduce",
    "score",
    "trackstar",
    "test_model_configuration",
]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_cli_help(cmd):
    """Each subcommand should produce --help output without crashing."""
    result = subprocess.run(
        ["bergson", cmd, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"bergson {cmd} --help failed:\n{result.stderr}"
