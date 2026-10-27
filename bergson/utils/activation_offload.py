"""CPU saved tensors that retain the padded layout required by efficient SDPA."""
import torch
import os
import json
from pathlib import Path
from collections import Counter
from contextlib import contextmanager


@contextmanager
def save_on_cpu_preserve_attention_bias():
    directory = os.environ.get('CAMILLA_OFFLOAD_STATS_DIR')
    stats = Counter()
    by_shape = Counter()
    def report():
        if directory:
            path = Path(directory);path.mkdir(parents=True, exist_ok=True)
            target = path / f'{os.getpid()}.json'
            temporary = target.with_suffix('.partial')
            temporary.write_text(json.dumps({'pid':os.getpid(), **stats,
                'largest_cumulative_saved_shapes':by_shape.most_common(16)},indent=2)+'\n')
            temporary.replace(target)

    def pack(tensor):
        # Efficient SDPA checks bias strides again during backward. A normal
        # device copy compacts sliced, padded bias and destroys its alignment.
        padded_bias = (tensor.ndim == 4 and tensor.shape[-2] == tensor.shape[-1]
                       and tensor.stride(-1) == 1 and tensor.stride(-2) > tensor.shape[-1])
        broadcast = any(stride == 0 and size > 1 for size, stride in zip(tensor.shape, tensor.stride()))
        bias_stride = tensor.stride() if padded_bias or broadcast else None
        shape = tensor.shape
        source = tensor
        if bias_stride is not None:
            # SDPA may save a broadcast bias with zero strides across heads.
            # Store one copy, then restore the broadcast view without writing
            # into overlapping storage.
            for axis, stride in enumerate(bias_stride):
                if stride == 0 and source.shape[axis] > 1:
                    source = source.narrow(axis, 0, 1)
        if directory:
            size=source.numel()*source.element_size()
            stats['saved_tensor_calls']+=1;stats['cumulative_saved_bytes']+=size
            by_shape[str(tuple(shape))]+=size
            if stats['saved_tensor_calls'] % 100 == 0:report()
        return tensor.device, source.to('cpu'), bias_stride, shape

    def unpack(saved):
        device, cpu, strides, shape = saved
        if strides is not None:
            restored = torch.empty_strided(cpu.shape, strides, device=device, dtype=cpu.dtype)
            restored.copy_(cpu)
            return restored.as_strided(shape, strides)
        return cpu.to(device)

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield
    finally:
        report()
