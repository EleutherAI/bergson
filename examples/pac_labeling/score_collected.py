"""Score training-time gradients collected by ``train_collect.py`` against
query gradients built with the same projection.

Usage:
    python examples/pac_labeling/score_collected.py \
        --gradients <run>/gradients/train --query <run>/query --out <dir>

Each epoch directory holds one projected, optimizer-normalized gradient per
document, the update that document contributed at the step it was trained
on. Its dot product with a query's projected gradient at the final model is
that document's first-order effect on the query loss through that update.
Writes one loss-signed score directory per epoch (``epoch_*``) and their sum
(``all``), plus ``update_norm.npy`` (per-document L2 norm of the summed
update) and ``epoch_cosine.npy`` (per-document cosine between the epochs'
updates).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bergson.data import load_gradients
from bergson.score.score_writer import save_sequence_scores


def load(path: Path) -> tuple[np.memmap, dict[str, int]]:
    info = json.loads((path / "info.json").read_text())
    return load_gradients(path), info["grad_sizes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gradients", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    q, q_sizes = load(args.query)
    q = np.asarray(q, dtype=np.float32)
    epochs = sorted(p for p in args.gradients.iterdir() if p.name.startswith("epoch_"))
    total = None
    norms_sq = []
    per_epoch = []
    for ep in epochs:
        g, sizes = load(ep)
        assert list(sizes) == list(q_sizes), (list(sizes)[:3], list(q_sizes)[:3])
        scores = np.empty((g.shape[0], q.shape[0]), dtype=np.float32)
        sq = np.empty(g.shape[0], dtype=np.float64)
        for start in range(0, g.shape[0], args.chunk):
            block = np.asarray(g[start : start + args.chunk], dtype=np.float32)
            scores[start : start + args.chunk] = block @ q.T
            sq[start : start + args.chunk] = (block.astype(np.float64) ** 2).sum(1)
        per_epoch.append((ep.name, g))
        norms_sq.append(sq)
        save_sequence_scores(args.out / ep.name, -scores)
        total = scores if total is None else total + scores
        print(f"{ep.name}: {g.shape}, |score| mean {np.abs(scores).mean():.3g}")
    assert total is not None
    save_sequence_scores(args.out / "all", -total)

    if len(per_epoch) == 2:
        (_, a), (_, b) = per_epoch
        cos = np.empty(a.shape[0], dtype=np.float32)
        norm = np.empty(a.shape[0], dtype=np.float32)
        for start in range(0, a.shape[0], args.chunk):
            x = np.asarray(a[start : start + args.chunk], dtype=np.float32)
            y = np.asarray(b[start : start + args.chunk], dtype=np.float32)
            dot = (x * y).sum(1)
            nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
            cos[start : start + args.chunk] = dot / np.maximum(nx * ny, 1e-12)
            norm[start : start + args.chunk] = np.linalg.norm(x + y, axis=1)
        np.save(args.out / "epoch_cosine.npy", cos)
        np.save(args.out / "update_norm.npy", norm)
    else:
        np.save(args.out / "update_norm.npy", np.sqrt(sum(norms_sq)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
