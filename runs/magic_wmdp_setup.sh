#!/bin/bash
# Setup script for MAGIC WMDP attribution experiment.
#
# FSDP double-backward requires a PyTorch patch that makes DTensor
# redistribution twice-differentiable (pytorch/pytorch#160509).
# This script applies the patch to the installed torch package.
#
# Prerequisites:
#   - bergson3 venv with torch 2.10.0+cu126 and bergson (editable, magic branch)
#   - Run from /home/a6a/lucia.a6a/bergson3
#
# Usage:
#   source .venv/bin/activate
#   bash runs/magic_wmdp_setup.sh
#   sbatch runs/magic_wmdp.sbatch

set -euo pipefail

TORCH_DIR="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
REDIST="$TORCH_DIR/distributed/tensor/_redistribute.py"
API="$TORCH_DIR/distributed/tensor/_api.py"

echo "Torch dir: $TORCH_DIR"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"

# ── 1. Patch _redistribute.py: add twice-differentiable redistribution ──
if grep -q "NestedRedistribute" "$REDIST"; then
    echo "[SKIP] _redistribute.py already patched (NestedRedistribute found)"
else
    echo "[PATCH] Applying twice-differentiable DTensor redistribution (pytorch/pytorch#160509)..."

    # We insert _redistribute_backward() helper and NestedRedistribute class,
    # and modify Redistribute.backward to delegate to NestedRedistribute.
    python - "$REDIST" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# 1a. Insert _redistribute_backward helper before class Redistribute
helper = '''

def _redistribute_backward(
    grad_output: "dtensor.DTensor",
    previous_spec: DTensorSpec,
    original_dtype: torch.dtype | None = None,
    backward_dtype: torch.dtype | None = None,
    async_op: bool = False,
):
    """
    Common function for redistributing a distributed tensor during backward
    and twice-backward backpropagation steps.
    """
    if backward_dtype != grad_output._local_tensor.dtype:
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

    normalized_placements: list[Placement] = []
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
        is_backward=True,
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


'''

marker = "class Redistribute(torch.autograd.Function):"
assert marker in src, f"Could not find '{marker}' in {path}"
src = src.replace(marker, helper + marker)

# 1b. Replace Redistribute.backward with delegation to NestedRedistribute
old_bwd_start = "    @staticmethod\n    def backward(ctx, grad_output: \"dtensor.DTensor\"):  # type: ignore[override]"
assert old_bwd_start in src, "Could not find Redistribute.backward"

# Find the end of the backward method (the return block ends the class)
bwd_idx = src.index(old_bwd_start)
# Find the closing of Redistribute class — look for the last return in backward
# We need to replace from the @staticmethod to the end of the file (backward is last method)
old_tail = src[bwd_idx:]

new_tail = '''    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        output_dtensor = NestedRedistribute.apply(
            grad_output,
            previous_spec,
            ctx.async_op,
            ctx.backward_dtype,
            ctx.original_dtype,
        )
        return (
            output_dtensor,
            None,
            None,
            None,
            None,
            None,
        )


class NestedRedistribute(torch.autograd.Function):
    """
    Makes DTensor redistribution twice-differentiable.
    Called during Redistribute.backward (first backward pass).
    NestedRedistribute.backward handles the second backward pass.
    Triple backward is not yet supported.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output: "dtensor.DTensor",
        previous_spec,
        async_op: bool = False,
        backward_dtype: torch.dtype | None = None,
        original_dtype: torch.dtype | None = None,
    ):
        ctx.original_dtype = original_dtype
        ctx.async_op = async_op
        ctx.backward_dtype = backward_dtype or ctx.original_dtype
        ctx.original_dtype = grad_output._local_tensor.dtype

        output, spec = _redistribute_backward(
            grad_output, previous_spec, ctx.original_dtype, backward_dtype, async_op
        )

        ctx.current_spec = spec

        return dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad2_output: "dtensor.DTensor"):
        previous_spec = ctx.current_spec
        async_op = ctx.async_op
        backward_dtype = ctx.backward_dtype or ctx.original_dtype

        output_dtensor = NestedRedistribute.apply(
            grad2_output,
            previous_spec,
            async_op,
            backward_dtype,
            ctx.original_dtype,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None,
        )
'''

src = src[:bwd_idx] + new_tail

with open(path, "w") as f:
    f.write(src)
print(f"  Patched {path}")
PYEOF
fi

# ── 2. Patch _api.py: use DTensor.from_local in _ToTorchTensor.backward ──
if grep -q "DTensor.from_local" "$API" && grep -q "grad_spec.device_mesh" "$API"; then
    echo "[SKIP] _api.py already patched"
else
    echo "[PATCH] Updating _api.py _ToTorchTensor.backward to use DTensor.from_local..."
    # Replace DTensor(grad_output, grad_spec, ...) with DTensor.from_local(grad_output, mesh, placements)
    python - "$API" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = '''            DTensor(
                # pyrefly: ignore [bad-argument-count]
                grad_output,
                grad_spec,
                # pyrefly: ignore [unexpected-keyword]
                requires_grad=grad_output.requires_grad,
            )'''

new = '''            DTensor.from_local(
                # pyrefly: ignore [bad-argument-count]
                grad_output,
                grad_spec.device_mesh,
                grad_spec.placements,
            )'''

assert old in src, f"Could not find DTensor(..., grad_spec, ...) pattern in {path}"
src = src.replace(old, new, 1)

with open(path, "w") as f:
    f.write(src)
print(f"  Patched {path}")
PYEOF
fi

# ── 3. Clear .pyc caches ──
find "$TORCH_DIR/distributed/tensor/__pycache__" \
    \( -name "_redistribute*" -o -name "_api*" \) -delete 2>/dev/null || true
echo "[OK] Cleared .pyc caches"

# ── 4. Verify patch loads ──
python -c "from torch.distributed.tensor._redistribute import NestedRedistribute, _redistribute_backward; print('[OK] Patch verified: NestedRedistribute importable')"

echo ""
echo "Setup complete. Submit with: sbatch runs/magic_wmdp.sbatch"
