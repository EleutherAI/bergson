"""Assemble per-token stores from forward-mode passes and check their chunk sums.

    python examples/compare_wikitext/tokens/combine_tokens.py \
        {source <SOURCE run> <passes>, rescale <pass> <doc scores>,
         trak <TRAK run> <passes>} <out>

``source``: SOURCE's score is minus the sum over segments of the mean over each
segment's checkpoints of ``g(checkpoint) . query_grad_segment``; the passes
``<passes>/source_s<l>c<c>`` combine the same way. ``rescale``: under unit
normalization a chunk's score is its rows' sum times one factor per chunk, fitted by
least squares over the queries against the doc-level scores. ``trak``: each
member's rows over its ``trak_scale``, averaged over members, times the members'
mean ``1 - p`` weights, as bergson averages TRAK members. Each mode prints how well
the chunk sums match the doc-level scores.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from bergson.data import load_scores
from bergson.score.score_writer import save_token_scores


def tokens(path):
    store = load_scores(Path(path))
    return np.asarray(store[:], dtype=np.float64), store.offsets


def doc_scores(path):
    return np.asarray(load_scores(Path(path))[:], dtype=np.float64)


def chunk_sums(rows, offsets):
    return np.add.reduceat(rows, offsets[:-1], axis=0)


def compare(sums, doc, label):
    rel = np.linalg.norm(sums - doc) / np.linalg.norm(doc)
    corr = np.mean(
        [np.corrcoef(sums[:, q], doc[:, q])[0, 1] for q in range(doc.shape[1])]
    )
    print(
        f"{label}: chunk sums vs doc scores, relative error {rel:.1e}, corr {corr:.6f}"
    )


def write(out, rows, offsets, config_from, higher_is_better):
    out = Path(out)
    save_token_scores(out, rows.astype(np.float32), offsets)
    cfg = yaml.safe_load(open(Path(config_from) / "config.yaml"))
    for step in cfg["steps"]:
        for body in step.values():
            body["score_cfg"]["higher_is_better"] = higher_is_better
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    shutil.copy(Path(config_from) / "processor_config.yaml", out)


def main():
    mode = sys.argv[1]
    if mode == "source":
        run, passes, out = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
        total = 0
        for l in range(3):
            seg = 0
            for c in range(2):
                rows, off = tokens(passes / f"source_s{l}c{c}")
                ckpt = doc_scores(run / f"segment_{l}" / f"scores_ckpt_{c}")
                compare(chunk_sums(rows, off), ckpt, f"segment {l} checkpoint {c}")
                seg = seg + rows
            total = total + seg / 2
        total = -total  # loss-signed, as the pipeline's aggregate
        compare(chunk_sums(total, off), doc_scores(run / "scores"), "SOURCE")
        write(out, total, off, passes / "source_s0c0", False)
    elif mode == "rescale":
        rows, off = tokens(sys.argv[2])
        doc = doc_scores(sys.argv[3])
        s = chunk_sums(rows, off)
        factor = (s * doc).sum(1) / (s * s).sum(1)
        rows = rows * np.repeat(factor, np.diff(off))[:, None]
        compare(chunk_sums(rows, off), doc, "rescaled")
        write(sys.argv[4], rows, off, sys.argv[2], True)
    elif mode == "trak":
        run, passes, out = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
        members = sorted(run.glob("checkpoint_*"))
        total, weights = 0, []
        for m in members:
            rows, off = tokens(passes / f"trak_{m.name}")
            compare(chunk_sums(rows, off), doc_scores(m / "scores"), m.name)
            total = total + rows / float(np.load(m / "scores" / "trak_scale.npy"))
            weights.append(np.load(m / "scores" / "trak_weights.npy"))
        w = np.repeat(np.mean(weights, axis=0), np.diff(off))[:, None]
        write(
            out, total / len(members) * w, off, passes / f"trak_{members[0].name}", True
        )
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
