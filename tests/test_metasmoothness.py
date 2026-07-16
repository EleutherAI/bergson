"""Unit tests for the metasmoothness command.

These tests exercise CLI parsing and the scoring function only — they never
call `.execute()`, so they need no GPU and no model downloads.
"""

import torch
from simple_parsing import ArgumentParser, ConflictResolution

from bergson.__main__ import Main
from bergson.magic.metasmoothness import metasmoothness_score


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    return parser


def test_cli_parser_constructs_with_metasmoothness():
    """A single-character field name (``h``) makes simple_parsing derive a ``-h``
    short flag that collides with argparse's ``-h/--help``, which raises while the
    parser is still being built and takes down *every* subcommand, not just this
    one. Guard the whole-parser construction path."""
    build_parser()


def test_fd_step_and_direction_seed_parse():
    args = build_parser().parse_args(
        ["metasmoothness", "run/path", "--fd_step", "0.25", "--direction_seed", "7"]
    )
    assert args.prog.command.fd_step == 0.25
    assert args.prog.command.direction_seed == 7


def test_score_is_one_for_perfectly_linear_response():
    """Equal consecutive steps => both finite differences share a sign everywhere."""
    theta0 = torch.zeros(8)
    delta = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
    assert metasmoothness_score(theta0, theta0 + delta, theta0 + 2 * delta) == 1.0


def test_score_is_negative_when_response_reverses():
    """Second step undoes the first => signs disagree on every moved coordinate."""
    theta0 = torch.zeros(4)
    theta_h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    theta_2h = torch.tensor([0.5, 1.0, 1.5, 2.0])
    assert metasmoothness_score(theta0, theta_h, theta_2h) == -1.0


def test_score_is_one_when_nothing_moves():
    theta = torch.zeros(4)
    assert metasmoothness_score(theta, theta, theta) == 1.0
