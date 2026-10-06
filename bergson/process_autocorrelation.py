import torch
import torch.distributed as dist

from bergson.gradients import GradientProcessor


def _eigh(prec: torch.Tensor, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(prec.to(dtype=torch.float64, device=device))
    return (
        eigvals.to(dtype=dtype).contiguous().cpu(),
        eigvecs.to(dtype=dtype).contiguous().cpu(),
    )


def process_autocorrelation_matrices(
    processor: GradientProcessor,
    hessians: dict[str, torch.Tensor],
    num_rows: int,
    grad_sizes: dict[str, int],
    rank: int,
):
    """
    Aggregate autocorrelation matrices across ranks and compute their eigen
    decomposition distributed across all ranks.

    ``num_rows`` is the number of gradient rows summed into the Gram — the
    document count for per-sequence gradients, or the gradient-carrying token
    count when ``attribute_tokens`` produced per-token rows — so the
    normalized matrix is the second moment over whichever unit the rows
    represent.

    A rank may hold a partial Gram over its own rows for every module, or the
    full Gram for the modules it owns. Each module's Gram is summed onto one
    of the ranks that hold it, which eigendecomposes the total.
    """
    device = next(iter(hessians.values())).device
    dtype = next(iter(hessians.values())).dtype

    if rank == 0:
        print("Saving hessians...")

    for name, prec in hessians.items():
        hessians[name] = (prec / num_rows).cpu()

    if not dist.is_initialized():
        if rank == 0:
            print("Computing hessian eigen decompositions...")
        processor.hessians = hessians
        processor.hessians_eigen = {
            name: _eigh(prec, device, dtype) for name, prec in hessians.items()
        }
        return

    if rank == 0:
        print("Reducing hessians and computing eigen decompositions...")

    cpu_group: dist.ProcessGroup = dist.new_group(backend="gloo")  # type: ignore[assignment]
    world_size = dist.get_world_size()
    held: list[list[str]] = [[] for _ in range(world_size)]
    dist.all_gather_object(held, sorted(hessians), group=cpu_group)

    totals, hessians_eigen = {}, {}
    for i, (name, grad_size) in enumerate(grad_sizes.items()):
        holders = [r for r in range(world_size) if name in held[r]] or [0]
        owner = holders[i % len(holders)]
        if name in hessians:
            local_prec = hessians.pop(name)
        else:
            local_prec = torch.zeros([grad_size, grad_size], dtype=dtype, device="cpu")

        dist.reduce(local_prec, dst=owner, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == owner:
            totals[name] = local_prec
            hessians_eigen[name] = _eigh(local_prec, device, dtype)

    if rank == 0:
        print("Gathering hessians and eigen decompositions...")

    for name, grad_size in grad_sizes.items():
        prec_size = torch.Size([grad_size, grad_size])
        prec = totals.pop(name, None)
        if name in hessians_eigen:
            eigval, eigvec = hessians_eigen[name]
        else:
            prec = torch.zeros(prec_size, dtype=dtype)
            eigval = torch.zeros(prec_size[0], dtype=dtype)
            eigvec = torch.zeros(prec_size, dtype=dtype)

        for t in (prec, eigval, eigvec):
            dist.reduce(t, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            hessians[name] = prec
            hessians_eigen[name] = (eigval, eigvec)

    if rank == 0:
        processor.hessians = hessians
        processor.hessians_eigen = hessians_eigen
