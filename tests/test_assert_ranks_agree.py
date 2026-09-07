"""Ranks resuming at different points must fail rather than deadlock later.

Runs on CPU via gloo so it doesn't need a multi-GPU node.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.distributed import assert_ranks_agree


def _worker(rank: int, world_size: int, port: int, values, results) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        try:
            assert_ranks_agree(values[rank], torch.device("cpu"), "Progress")
            results[rank] = "ok"
        except RuntimeError as e:
            results[rank] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(values: list[int]) -> dict[int, str]:
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _worker,
        args=(len(values), port, values, results),
        nprocs=len(values),
        join=True,
    )
    return dict(results)


def test_matching_values_pass():
    assert _run([7, 7]) == {0: "ok", 1: "ok"}


def test_mismatched_values_raise_on_every_rank():
    outcomes = _run([7, 8])
    for rank in (0, 1):
        assert "disagrees across ranks (7..8)" in outcomes[rank], outcomes[rank]


def test_single_process_is_a_no_op():
    assert_ranks_agree(3, torch.device("cpu"), "Progress")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
