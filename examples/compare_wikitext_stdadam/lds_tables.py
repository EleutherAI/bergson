"""Print the per-example README tables from lds_<method>.json (and qld_<method>.json
when present) files.

    python examples/compare_wikitext_stdadam/lds_tables.py runs/compare_wikitext_stdadam

LDS json is written by examples/compare_wikitext_stdadam/lds_from_bank.py (BIF: metasmoothness bif_lds.py),
QLD json by examples/compare_wikitext_stdadam/qld_from_filters.py. Rows are sorted by
proponent QLD when any is present (rows without one last), else by mean rho.
"""

import json
import sys
from pathlib import Path

import numpy as np

LABELS = {
    "magic": "MAGIC (per-query)",
    "ekfac": "EK-FAC",
    "trak": "TRAK (per-module kernel, proj 32)",
    "trak_joint": "TRAK (joint kernel, proj 16)",
    "trak_joint_p8": "TRAK (joint kernel, proj 8)",
    "trak_joint_p16_d1": "TRAK (joint kernel, proj 16, damping 1.0)",
    "source": "SOURCE",
    "source_adam": "SOURCE-Adam",
    "trackstar": "TrackStar (projection 64)",
    "trackstar_adam": "TrackStar (Adam, projection 64)",
    "bif": "BIF",
    "activation": "Activation similarity",
    "gradient": "Gradient cosine similarity",
}


def rows(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("lds_*.json")):
        key = p.stem[4:]
        d = json.loads(p.read_text())
        per_q = d.get("per_query", [])
        lo, hi = d["ci95"]
        if "median" in d:
            med, mn, mx, sig, nq = d["median"], d["min"], d["max"], d["n_sig"], d["n_queries"]
        else:  # bif_lds.py output
            med, mn, mx, nq = float(np.median(per_q)), min(per_q), max(per_q), len(per_q)
            sig = "—"
        q = run_dir / f"qld_{key}.json"
        qld = json.loads(q.read_text()) if q.exists() else None
        out.append((d["lds"], LABELS.get(key, key), lo, hi, med, mn, mx, sig, nq, qld))
    if any(r[-1] for r in out):
        return sorted(out, key=lambda r: (r[-1]["qld"] if r[-1] else float("-inf"), r[0]), reverse=True)
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
                f" {qld['qld']:.4f} | [{qld['ci95'][0]:.4f}, {qld['ci95'][1]:.4f}] |" if qld else " — | — |"
            )
        line += f" {lds:.3f} | [{lo:.3f}, {hi:.3f}] | {med:.3f} | {mn:.3f} | {mx:.3f} | {sig_s} |"
        print(line)
