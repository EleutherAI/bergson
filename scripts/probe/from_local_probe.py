"""Can each rank build its shard directly, with no collective?

Checks DTensor.from_local against distribute_tensor on EVERY rank, including
uneven splits, and that simple_fsdp's parametrization still runs a forward.
Every assertion is reported per rank: the previous probe only printed rank 0,
which held the source data and so could not fail.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def chunk_bounds(n: int, world: int, rank: int) -> tuple[int, int]:
    """Bounds of ``rank``'s piece under torch.chunk semantics on dim 0."""
    per = (n + world - 1) // world
    start = min(rank * per, n)
    return start, min(start + per, n)


def say(msg):
    print(f"[probe r{dist.get_rank()}] {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world,))

    # even, uneven, and fewer rows than ranks (LoRA r=32 over 128 ranks)
    for n in (8 * world, 8 * world + 3, max(1, world // 2)):
        full = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)

        # Reference: the existing path, which needs the full tensor everywhere.
        with mesh:
            ref = distribute_tensor(full.clone().cuda(), placements=(Shard(0),))
        ref_local = ref.to_local()

        # Candidate: this rank reads only its own slice.
        lo, hi = chunk_bounds(n, world, rank)
        mine = full[lo:hi].clone().cuda()
        with mesh:
            got = DTensor.from_local(
                mine, placements=(Shard(0),), shape=full.shape, stride=full.stride()
            )
        got_local = got.to_local()

        same_local = got_local.shape == ref_local.shape and torch.equal(
            got_local, ref_local
        )
        rebuilt = got.full_tensor()
        same_full = torch.equal(rebuilt.cpu(), full)
        say(
            f"n={n} slice=({lo},{hi}) local={tuple(got_local.shape)} "
            f"ref={tuple(ref_local.shape)} local_match={same_local} "
            f"full_match={same_full}"
        )
        assert same_local, f"rank {rank}: local shard differs from distribute_tensor"
        assert same_full, f"rank {rank}: full_tensor does not rebuild the original"

    # Does it survive simple_fsdp's parametrization and a forward?
    from torch.nn.utils.parametrize import register_parametrization

    from bergson.magic.fsdp import ReplicateComputation

    # Same weights on every rank, or the shards assemble into a tensor that
    # matches nobody's local reference.
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 4, bias=False).cuda()
    weight = lin.weight.detach().clone()
    lo, hi = chunk_bounds(4, world, rank)
    with mesh:
        shard = DTensor.from_local(
            weight[lo:hi].contiguous(),
            placements=(Shard(0),),
            shape=weight.shape,
            stride=weight.stride(),
        )
    lin.weight = torch.nn.Parameter(shard)
    register_parametrization(lin, "weight", ReplicateComputation(), unsafe=True)
    x = torch.ones(2, 4, device="cuda")
    out = lin(x)
    want = x @ weight.T
    say(f"forward ok diff={float((out - want).abs().max()):.3e}")
    assert torch.allclose(out, want, atol=1e-5), f"rank {rank}: forward mismatch"

    dist.barrier()
    say("ALL CHECKS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
