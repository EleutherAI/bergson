"""Multi-node bergson build simulated on a single node.

Spawns two ``python -m bergson build`` processes that pretend to be separate
nodes. Each owns one physical GPU via outer CUDA_VISIBLE_DEVICES, and they
rendezvous over loopback. Exercises the multi-node code path through
``launch_distributed_run``: cross-node NCCL init, ``start_rank`` math
(``rank = node_rank * nproc_per_node + local_rank``), and the rank-0-only
partial-to-final move at the end.

If multi-node rendezvous regresses (e.g. CUDA_VISIBLE_DEVICES pinning
breaks the per-node NCCL handshake, or only one node's data ever reaches disk),
this test either times out or fails the gradient-count assertion.
"""

import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch

from bergson.data import load_gradients

from .cli_command import bergson_cmd, bergson_env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Multi-node simulation needs >= 2 GPUs",
)
def test_multinode_build_two_nodes_one_gpu_each(tmp_path: Path):
    run_path = tmp_path / "mn_run"
    n_examples = 32
    port = free_port()

    common_args = bergson_cmd(
        "build",
        str(run_path),
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--dataset",
        "NeelNanda/pile-10k",
        "--split",
        f"train[:{n_examples}]",
        "--truncation",
        "--projection_dim",
        "8",
        "--token_batch_size",
        "256",
        "--nnode",
        "2",
        "--nproc_per_node",
        "1",
        "--overwrite",
    )

    base_env = {
        **bergson_env(),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(port),
    }

    # Capture each node to its own file rather than a pipe. Both nodes must run
    # concurrently to meet at the collective, but communicate() drains one pipe
    # at a time: the other node fills its 64 KiB buffer (these emit ~80 KiB of
    # progress bars), blocks on write, never reaches the rendezvous, and both
    # nodes deadlock until the timeout.
    procs = []
    log_paths = []
    handles = []
    for node_rank, gpu in [(1, "1"), (0, "0")]:
        env = {**base_env, "CUDA_VISIBLE_DEVICES": gpu}
        log_path = tmp_path / f"node_{node_rank}.log"
        handle = log_path.open("w")
        log_paths.append(log_path)
        handles.append(handle)
        procs.append(
            subprocess.Popen(
                common_args + ["--node_rank", str(node_rank)],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    def read_logs() -> list[str]:
        return [p.read_text(errors="replace") if p.exists() else "" for p in log_paths]

    try:
        deadline = time.monotonic() + 600
        for proc in procs:
            proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        for proc in procs:
            proc.wait()
        pytest.fail(
            "Multi-node simulation timed out — likely a cross-node "
            "rendezvous hang. Outputs so far:\n"
            + "\n---\n".join(o or "<no output>" for o in read_logs())
        )
    finally:
        for handle in handles:
            handle.close()

    outputs = read_logs()

    for i, proc in enumerate(procs):
        assert proc.returncode == 0, (
            f"Node {1 - i if i == 0 else 0} exited {proc.returncode}:\n" f"{outputs[i]}"
        )

    assert run_path.exists(), (
        "Final run_path missing — rank 0's shutil.move from partial path "
        "did not run, so the multi-node rank computation is wrong."
    )

    grads = load_gradients(run_path)
    assert len(grads) == n_examples, (
        f"Expected {n_examples} gradients across both nodes, got {len(grads)}. "
        "One node's shard may not have been written."
    )
