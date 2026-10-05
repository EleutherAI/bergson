"""Combine theta shards from metasmoothness trainings run with ``BERGSON_MS_K``.

    python -m bergson.magic.combine_ms_thetas RUN_PATH
    python -m bergson.magic.combine_ms_thetas RUN_K0 RUN_K1 RUN_K2 --ranks 4

Under ``BERGSON_TP`` pass ``--ranks <tp width>`` to read one data-parallel
replica. ``--by-param`` adds a per-matrix and per-layer-block breakdown.
"""

import argparse
import collections
import glob
import json
import os
import re

import torch


def shard_paths(run_path: str, k: int, ranks: int | None) -> list[str]:
    """Shard paths for training ``k`` in rank order, refusing a partial set."""
    shard_dir = os.path.join(run_path, "theta_shards")
    found = {}
    for path in glob.glob(os.path.join(shard_dir, f"theta_k{k}_rank*.pt")):
        match = re.search(r"_rank(\d+)\.pt$", path)
        assert match, path
        found[int(match.group(1))] = path
    if not found:
        raise FileNotFoundError(f"no shards for k={k} in {shard_dir}")
    world = torch.load(found[min(found)], map_location="cpu")["world_size"]
    want = range(ranks if ranks is not None else world)
    missing = [r for r in want if r not in found]
    if missing:
        raise ValueError(
            f"k={k}: missing shards for ranks {missing[:8]} of {len(want)}; "
            "the combine needs every rank's shard"
        )
    return [found[r] for r in want]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_paths", nargs="+", help="one run_path, or one per k")
    parser.add_argument("--fd-step", type=float, default=0.1)
    parser.add_argument("--direction-seed", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--by-param", action="store_true")
    parser.add_argument("--out", default=None, help="default: first run_path")
    args = parser.parse_args()

    if len(args.run_paths) not in (1, 3):
        parser.error("pass one run_path, or three in k order")
    run_paths = args.run_paths * 3 if len(args.run_paths) == 1 else args.run_paths
    trips = [shard_paths(p, k, args.ranks) for k, p in enumerate(run_paths)]
    if len({len(t) for t in trips}) != 1:
        raise ValueError(f"differing shard counts per k: {[len(t) for t in trips]}")

    # group -> [weighted sum, |d|_1]
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for p0, ph, p2h in zip(*trips):
        s0 = torch.load(p0, map_location="cpu")
        t0 = s0["theta"].double()
        th = torch.load(ph, map_location="cpu")["theta"].double()
        t2h = torch.load(p2h, map_location="cpu")["theta"].double()
        if not (t0.numel() == th.numel() == t2h.numel()):
            raise ValueError(
                f"shard length mismatch: {t0.numel()}, {th.numel()}, {t2h.numel()}"
            )
        d = (t2h - t0).abs()
        w = d * torch.sign(th - t0) * torch.sign(t2h - th)
        acc[("all",)][0] += float(w.sum())
        acc[("all",)][1] += float(d.sum())
        if args.by_param:
            off = 0
            for name, n in s0["layout"]:
                keys = [("matrix", "A" if "lora_A" in name else "B")]
                layer = re.search(r"layers?\.(\d+)\.", name)
                if layer:
                    lo = int(layer.group(1)) // 8 * 8
                    keys.append(("layers", f"{lo}-{lo + 7}"))
                for key in keys:
                    acc[key][0] += float(w[off : off + n].sum())
                    acc[key][1] += float(d[off : off + n].sum())
                off += n
            assert off == t0.numel()
        del t0, th, t2h, d, w

    weighted, total = acc[("all",)]
    score = 1.0 if total == 0 else weighted / total
    result = {
        "score": score,
        "fd_step": args.fd_step,
        "direction_seed": args.direction_seed,
        "total_movement_l1": total,
    }
    if args.by_param:
        result["by_param"] = {
            " ".join(key): {"score": w / max(d, 1e-30), "movement_share": d / total}
            for key, (w, d) in sorted(acc.items())
            if key != ("all",)
        }
        for key, val in result["by_param"].items():
            print(f"{key:16s} ms={val['score']:+.4f} share={val['movement_share']:.3f}")
    out = os.path.join(args.out or args.run_paths[0], "metasmoothness.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[metasmoothness] score = {score:.4f} (h={args.fd_step})")
    print(f"[metasmoothness] saved to {out}")


if __name__ == "__main__":
    main()
