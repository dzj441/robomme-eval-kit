"""Merge sharded RoboMME results and compare against the paper's Table 15."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

SUITE = {
    "BinFill": "Counting", "PickXtimes": "Counting",
    "SwingXtimes": "Counting", "StopCube": "Counting",
    "VideoUnmask": "Permanence", "ButtonUnmask": "Permanence",
    "VideoUnmaskSwap": "Permanence", "ButtonUnmaskSwap": "Permanence",
    "PickHighlight": "Reference", "VideoRepick": "Reference",
    "VideoPlaceButton": "Reference", "VideoPlaceOrder": "Reference",
    "MoveCube": "Imitation", "InsertPeg": "Imitation",
    "PatternLock": "Imitation", "RouteStick": "Imitation",
}
# paper Table 15, FrameSamp+Modul (mean +- std over 3 ckpts x 3 seeds)
PAPER = {
    "BinFill": (39.56, 5.27), "PickXtimes": (87.33, 2.45),
    "SwingXtimes": (92.00, 2.24), "StopCube": (42.00, 11.36),
    "VideoUnmask": (32.67, 3.16), "ButtonUnmask": (25.11, 3.18),
    "VideoUnmaskSwap": (24.44, 6.06), "ButtonUnmaskSwap": (18.22, 3.80),
    "PickHighlight": (22.89, 3.89), "VideoRepick": (30.44, 5.81),
    "VideoPlaceButton": (60.00, 4.00), "VideoPlaceOrder": (32.00, 3.87),
    "MoveCube": (77.78, 3.80), "InsertPeg": (7.56, 3.57),
    "PatternLock": (53.56, 4.56), "RouteStick": (66.67, 4.12),
}
PAPER_AVG = 44.51

# Collect every (task, episode) across shards before scoring. Under
# episode-level sharding each shard holds only its own slice, marking the rest
# "skip", so the union is what defines a task's rate.
# setup_save_directory nests results under <policy>/ckpt<id>/seed<seed>/
episodes: dict[str, dict[str, object]] = {}
_shards = sorted(OUT.glob("shard*/**/progress.json"),
                 key=lambda p: int(str(p).split("shard")[1].split("/")[0]))
_n = len(_shards)
for prog in _shards:
    shard = int(str(prog).split("shard")[1].split("/")[0])
    for task, eps in json.loads(prog.read_text()).items():
        for ep, v in eps.items():
            # Under episode-level sharding a shard's own slice is ep % n == shard;
            # everything else is a pre-seeded placeholder. With one shard per task
            # (task-level sharding) n == 1, so every entry is owned.
            if v != "skip" and (_n == 1 or int(ep) % _n == shard):
                episodes.setdefault(task, {})[ep] = v

ours: dict[str, float] = {}
errors: dict[str, int] = {}
counts: dict[str, int] = {}
for task, eps in episodes.items():
    vals = list(eps.values())
    if not vals:
        continue
    ours[task] = 100.0 * sum(1 for v in vals if v is True) / len(vals)
    errors[task] = sum(1 for v in vals if v == "error")
    counts[task] = len(vals)

print("=" * 82)
print("RoboMME — MME-VLA perceptual-framesamp-modul, ckpt 79999, seed 7")
print("=" * 82)
print(f"{'task':<20} {'suite':<12} {'ours%':>7} {'paper':>16} {'sigma':>8} {'err':>5} {'n':>5}")
print("-" * 82)
for task in SUITE:
    if task not in ours:
        print(f"{task:<20} {SUITE[task]:<12} {'MISSING':>7}")
        continue
    o = ours[task]
    if task in PAPER:
        mu, sd = PAPER[task]
        z = (o - mu) / sd if sd else 0.0
        paper_s, z_s = f"{mu:.2f} +- {sd:.2f}", f"{z:+.2f}"
    else:
        paper_s, z_s = "-", "-"
    print(f"{task:<20} {SUITE[task]:<12} {o:7.1f} {paper_s:>16} {z_s:>8} {errors.get(task,0):5d} {counts.get(task,0):5d}")

print("-" * 82)
if ours:
    avg = sum(ours.values()) / len(ours)
    print(f"{'AVG':<20} {'':12} {avg:7.1f}   paper {PAPER_AVG}  (over {len(ours)}/16 tasks)")
    for suite in ("Counting", "Permanence", "Reference", "Imitation"):
        sub = [v for k, v in ours.items() if SUITE[k] == suite]
        if sub:
            print(f"  {suite:<14} n={len(sub)}  {sum(sub)/len(sub):5.1f}%")
    tot_err = sum(errors.values())
    if tot_err:
        print(f"\n  WARNING: {tot_err} episodes recorded status 'error'")
    (OUT / "aggregate.json").write_text(
        json.dumps({"per_task": ours, "avg": avg, "errors": errors}, indent=2) + "\n"
    )
