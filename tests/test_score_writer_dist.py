"""One-shot score savers must not issue collectives.

``save_sequence_scores`` / ``save_token_scores`` are called from a single rank
(e.g. rank 0 saving aggregate MAGIC scores). If they issued a barrier, that
stray collective would pair with an unrelated collective on the other ranks and
deadlock the process group at teardown.
"""

import socket
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.score.score_writer import save_sequence_scores, save_token_scores


def _rank0_only_save_worker(rank, world_size, port, tmpdir):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        if rank == 0:
            scores = np.arange(8, dtype=np.float32)[:, None]
            save_sequence_scores(Path(tmpdir) / "seq_scores", scores)

            offsets = np.array([0, 3, 8], dtype=np.int64)
            save_token_scores(
                Path(tmpdir) / "tok_scores",
                np.arange(8, dtype=np.float32)[:, None],
                offsets,
            )

        # Every rank participates. A stray barrier inside the rank-0-only saves
        # above would pair with this one, desync the group, and hang one rank
        # past the gloo timeout.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_one_shot_savers_are_not_collective(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    world_size = 2
    mp.spawn(
        _rank0_only_save_worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    seq = np.memmap(
        tmp_path / "seq_scores" / "scores.bin",
        dtype=np.dtype(
            {
                "names": ["score_0", "written_0"],
                "formats": ["<f4", "?"],
                "offsets": [0, 4],
                "itemsize": 8,
            }
        ),
        mode="r",
    )
    assert np.allclose(seq["score_0"], np.arange(8, dtype=np.float32))
    assert seq["written_0"].all()
