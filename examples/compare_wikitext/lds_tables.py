"""Print the per-example README tables from lds_<method>.json (and qld_<method>.json
when present) files.

    python examples/compare_wikitext/lds_tables.py runs/compare_wikitext

LDS json is written by examples/compare_wikitext/lds_from_bank.py, QLD json by
examples/compare_wikitext/qld_from_filters.py. Rows are sorted by
proponent QLD when any is present (rows without one last), else by mean rho.
"""

import json
import sys
from pathlib import Path

import numpy as np

LABELS = {
    "magic": "MAGIC (per-query)",
    "magic_seed43": "MAGIC (cross-seed)",
    "ekfac": "EK-FAC",
    "kfac": "KFAC",
    "trak_ens": "TRAK (8-model ensemble)",
    "source": "SOURCE",
    "source_adam": "SOURCE (Adam)",
    "trackstar_p16": "TrackStar (no optimizer correction, projection 16)",
    "trackstar_adam_p16": "TrackStar (Adam, projection 16)",
    "trackstar_p32": "TrackStar (no optimizer correction, projection 32)",
    "trackstar_adam_p32": "TrackStar (Adam, projection 32)",
    "trackstar_p64": "TrackStar (no optimizer correction, projection 64)",
    "trackstar_adam_p64": "TrackStar (Adam, projection 64)",
    "bif": "BIF",
    "activation": "Activation similarity",
    "gradient_cosine": "Gradient cosine similarity",
    "gradient_cosine_projected": "Projected gradient cosine similarity",
    "shampoo": "Shampoo",
    "bm25": "BM25",
    "semantic": "Jina v5 semantic search",
    "qwen3": (
        "[Qwen3-Embedding-8B](https://huggingface.co/spaces/mteb/leaderboard)"
        " semantic search"
    ),
}


def rows(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("lds_*.json")):
        key = p.stem[4:]
        d = json.loads(p.read_text())
        per_q = d.get("per_query", [])
        lo, hi = d["ci95"]
        if "median" in d:
            med, mn, mx, sig, nq = (
                d["median"],
                d["min"],
                d["max"],
                d["n_sig"],
                d["n_queries"],
            )
        else:  # bif_lds.py output
            med, mn, mx, nq = (
                float(np.median(per_q)),
                min(per_q),
                max(per_q),
                len(per_q),
            )
            sig = "—"
        q = run_dir / f"qld_{key}.json"
        qld = json.loads(q.read_text()) if q.exists() else None
        out.append((d["lds"], LABELS.get(key, key), lo, hi, med, mn, mx, sig, nq, qld))
    if any(r[-1] for r in out):
        return sorted(
            out,
            key=lambda r: (r[-1]["qld"] if r[-1] else float("-inf"), r[0]),
            reverse=True,
        )
    return sorted(out, key=lambda r: r[0], reverse=True)


for arg in sys.argv[1:]:
    run_dir = Path(arg)
    has_qld = any(r[-1] for r in rows(run_dir))
    print(f"\n### {run_dir.name}\n")
    head = "| method |"
    if has_qld:
        head += " proponent QLD | 95% CI |"
    head += " mean ρ | 95% CI | median ρ | min | max | queries p<.05 |"
    print(head)
    print("|---" * (9 if has_qld else 7) + "|")
    for lds, name, lo, hi, med, mn, mx, sig, nq, qld in rows(run_dir):
        sig_s = sig if isinstance(sig, str) else f"{sig}/{nq}"
        line = f"| {name} |"
        if has_qld:
            line += (
                f" {qld['qld']:.4f} | [{qld['ci95'][0]:.4f}, {qld['ci95'][1]:.4f}] |"
                if qld
                else " — | — |"
            )
        line += (
            f" {lds:.3f} | [{lo:.3f}, {hi:.3f}] | {med:.3f} | {mn:.3f} | {mx:.3f}"
            f" | {sig_s} |"
        )
        print(line)
