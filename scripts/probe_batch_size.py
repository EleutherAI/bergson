#!/usr/bin/env python3
"""
Probe maximum token_batch_size for OLMo-3-7B by running bergson build
with --max_tokens to limit to a few iterations. Tests increasing batch
sizes until OOM.
"""

import subprocess
import sys
import time


def try_batch_size(batch_size: int, precision: str) -> bool:
    """Run a tiny build with the given batch size. Returns True if it succeeds."""
    run_path = f"runs/_probe_{precision}_{batch_size}"
    cmd = [
        "bergson",
        "build",
        run_path,
        "--model",
        "allenai/Olmo-3-7B-Instruct",
        "--normalizer",
        "none",
        "--precision",
        precision,
        "--dataset",
        "EleutherAI/SmolLM2-135M-10B",
        "--split",
        "train[:1000]",
        "--truncation",
        "--projection_dim",
        "16",
        "--token_batch_size",
        str(batch_size),
        "--nproc_per_node",
        "4",
        "--max_tokens",
        "10000",
        "--overwrite",
    ]
    print(f"\n{'='*60}")
    print(f"Testing token_batch_size={batch_size} precision={precision}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    oom = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "OutOfMemory" in line or "CUDA out of memory" in line or "out of memory" in line.lower():
            oom = True
    proc.wait()
    elapsed = time.monotonic() - start

    if proc.returncode == 0 and not oom:
        print(f"\n  SUCCESS: batch_size={batch_size} in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED: batch_size={batch_size} (rc={proc.returncode}, oom={oom}) in {elapsed:.1f}s")
        return False


def main():
    precision = sys.argv[1] if len(sys.argv) > 1 else "bf16"
    print(f"Probing max batch size for precision={precision}")

    # Test increasing batch sizes
    batch_sizes = [8192, 16384, 32768, 49152, 65536, 98304, 131072]
    max_working = 0

    for bs in batch_sizes:
        ok = try_batch_size(bs, precision)
        if ok:
            max_working = bs
        else:
            # Try midpoint between last working and this failed size
            if max_working > 0:
                mid = (max_working + bs) // 2
                # Round to nearest 1024
                mid = (mid // 1024) * 1024
                if mid != max_working and mid != bs:
                    print(f"\n  Trying midpoint: {mid}")
                    if try_batch_size(mid, precision):
                        max_working = mid
            break

    print(f"\n{'='*60}")
    print(f"RESULT: Max working token_batch_size for {precision} = {max_working}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
