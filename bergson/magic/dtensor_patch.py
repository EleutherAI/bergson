"""Runtime monkey-patch for twice-differentiable DTensor redistribution.

Implements pytorch/pytorch#160509 at runtime, avoiding the need to modify
torch source files on disk. Call `apply_dtensor_patch()` before any DTensor
operations that require double backward (e.g. MAGIC attribution with FSDP).

Safe to call multiple times (idempotent).
"""

import torch

_PATCHED = False


def apply_dtensor_patch():
    """Patch DTensor redistribution to support double backward.

    Monkey-patches `Redistribute.backward` and `_ToTorchTensor.backward`
    in the installed torch package so that FSDP redistribution is
    twice-differentiable.
    """
    global _PATCHED
    if _PATCHED:
        return

    _patch_redistribute()
    _patch_to_torch_tensor()
    _PATCHED = True


def _patch_redistribute():
    import torch.distributed.tensor._api as dtensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor._redistribute import (
        Redistribute,
        redistribute_local_tensor,
    )
    from torch.distributed.tensor.placement_types import Replicate

    def _redistribute_backward(
        grad_output,
        previous_spec,
        original_dtype: torch.dtype | None = None,
        backward_dtype: torch.dtype | None = None,
        async_op: bool = False,
    ):
        if (
            backward_dtype is not None
            and backward_dtype != grad_output._local_tensor.dtype
        ):
            local_tensor = grad_output._local_tensor.to(dtype=backward_dtype)
            current_spec = DTensorSpec(
                mesh=grad_output._spec.device_mesh,
                placements=grad_output._spec.placements,
                tensor_meta=TensorMeta(
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                    dtype=backward_dtype,
                ),
            )
            previous_spec = DTensorSpec(
                mesh=previous_spec.device_mesh,
                placements=previous_spec.placements,
                tensor_meta=current_spec.tensor_meta,
            )
        else:
            local_tensor = grad_output._local_tensor
            current_spec = grad_output._spec

        normalized_placements = []
        for current, target in zip(current_spec.placements, previous_spec.placements):
            if (current.is_shard() or current.is_replicate()) and target.is_partial():
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(target)

        previous_spec = DTensorSpec(
            previous_spec.device_mesh,
            placements=tuple(normalized_placements),
            tensor_meta=previous_spec.tensor_meta,
        )

        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
        )

        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=output.dtype,
            ),
        )
        return output, spec

    class NestedRedistribute(torch.autograd.Function):
        """Makes DTensor redistribution twice-differentiable.

        Called during Redistribute.backward (first backward pass).
        NestedRedistribute.backward handles the second backward pass.
        """

        @staticmethod
        def forward(
            ctx,
            grad_output,
            previous_spec,
            async_op=False,
            backward_dtype=None,
            original_dtype=None,
        ):
            ctx.async_op = async_op
            ctx.backward_dtype = backward_dtype or original_dtype
            ctx.original_dtype = grad_output._local_tensor.dtype

            output, spec = _redistribute_backward(
                grad_output,
                previous_spec,
                ctx.original_dtype,
                backward_dtype,
                async_op,
            )

            ctx.current_spec = spec

            return dtensor.DTensor(
                output,
                spec,
                requires_grad=grad_output.requires_grad,
            )

        @staticmethod
        def backward(ctx, *grad_outputs):  # type: ignore[override]
            grad2_output = grad_outputs[0]
            output_dtensor = NestedRedistribute.apply(
                grad2_output,
                ctx.current_spec,
                ctx.async_op,
                ctx.backward_dtype,
                ctx.original_dtype,
            )

            return (output_dtensor, None, None, None, None)

    @staticmethod
    def _new_redistribute_backward(ctx, grad_output):
        previous_spec = ctx.current_spec
        output_dtensor = NestedRedistribute.apply(
            grad_output,
            previous_spec,
            ctx.async_op,
            ctx.backward_dtype,
            ctx.original_dtype,
        )
        return (output_dtensor, None, None, None, None, None)

    Redistribute.backward = _new_redistribute_backward  # type: ignore[reportAttributeAccessIssue]


def _patch_to_torch_tensor():
    import torch.distributed.tensor._api as dtensor_api
    from torch.distributed.tensor._api import _ToTorchTensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor._utils import compute_global_tensor_info

    @staticmethod
    def _new_backward(ctx, grad_output):
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        _, tensor_stride = compute_global_tensor_info(
            grad_output, mesh, dtensor_spec.placements
        )
        tensor_stride = tuple(tensor_stride)
        grad_placements = grad_placements or dtensor_spec.placements
        grad_spec = DTensorSpec(
            mesh,
            grad_placements,
            tensor_meta=TensorMeta(
                shape=dtensor_meta.shape,
                stride=tensor_stride,
                dtype=dtensor_meta.dtype,
            ),
        )

        return (
            dtensor_api.DTensor.from_local(
                grad_output,
                grad_spec.device_mesh,
                grad_spec.placements,
            ),
            None,
        )

    _ToTorchTensor.backward = _new_backward
