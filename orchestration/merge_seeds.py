"""Aggregate several single-seed RoboMME runs into a baseline with error bars.

The paper reports 9-run means (3 checkpoints x 3 seeds). Only checkpoint 79999
was released, so the best we can do is seeds; this reports our mean +- std over
seeds and compares against the paper via a two-sample z on the difference of
means, which is the honest test given both sides carry spread.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/datadrive1/dzj/RoboMME/eval_out")
RUNS = sorted(ROOT.glob("ultra3_seed*")) or sorted(ROOT.glob("fsm_seed*"))

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

per_seed: dict[str, dict[str, float]] = {}
for run in RUNS:
    seed = run.name.split("seed")[-1]
    tasks: dict[str, float] = {}
    shards = sorted(run.glob("shard*/**/progress.json"),
                    key=lambda p: int(str(p).split("shard")[1].split("/")[0]))
    n = len(shards)
    owned: dict[str, dict[str, object]] = {}
    for prog in shards:
        sh = int(str(prog).split("shard")[1].split("/")[0])
        for task, eps in json.loads(prog.read_text()).items():
            for ep, v in eps.items():
                # episode-level sharding: a shard owns ep % n == shard
                if n == 1 or int(ep) % n == sh:
                    owned.setdefault(task, {})[ep] = v
    for task, eps in owned.items():
        vals = list(eps.values())
        if vals:
            tasks[task] = 100.0 * sum(1 for v in vals if v is True) / len(vals)
    # Only fold in seeds that finished all 16 tasks at 50 episodes; a run still
    # in flight would otherwise drag the mean toward its partial rates.
    complete = len(tasks) == 16 and all(len(v) == 50 for v in owned.values())
    if tasks and complete:
        per_seed[seed] = tasks
    elif tasks:
        n_eps = sum(len(v) for v in owned.values())
        print(f"  (skipping seed {seed}: incomplete, {n_eps}/800 episodes)")

if not per_seed:
    print(f"no completed runs under {ROOT}")
    raise SystemExit

seeds = sorted(per_seed, key=int)
print("=" * 96)
print(f"RoboMME — FrameSamp+Modul, ckpt 79999, seeds {', '.join(seeds)}")
print("=" * 96)
head = "".join(f"{'s'+s:>8}" for s in seeds)
print(f"{'task':<19}{head} {'mean':>7} {'sd':>6} | {'paper':>16} {'z':>7}")
print("-" * 96)

means: dict[str, float] = {}
for task in SUITE:
    vals = [per_seed[s][task] for s in seeds if task in per_seed[s]]
    if not vals:
        print(f"{task:<19}{'  pending':>{8*len(seeds)}}")
        continue
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    means[task] = mu
    pm, ps = PAPER[task]
    # difference of two means, each with its own spread
    denom = math.sqrt(ps**2 + (sd**2 if sd else 0.0)) or 1e-9
    z = (mu - pm) / denom
    cells = "".join(f"{per_seed[s].get(task, float('nan')):8.1f}" for s in seeds)
    flag = "  <<<" if abs(z) >= 2 else ""
    print(f"{task:<19}{cells} {mu:7.1f} {sd:6.1f} | {pm:7.2f} +-{ps:5.2f} {z:+7.2f}{flag}")

print("-" * 96)
if means:
    avg = statistics.mean(means.values())
    per_seed_avg = [statistics.mean(per_seed[s].values()) for s in seeds
                    if len(per_seed[s]) == len(SUITE)]
    sd_avg = statistics.stdev(per_seed_avg) if len(per_seed_avg) > 1 else 0.0
    print(f"{'AVG':<19}{''.join(f'{a:8.1f}' for a in per_seed_avg)} "
          f"{avg:7.1f} {sd_avg:6.1f} | {PAPER_AVG:7.2f}")
    for suite in ("Counting", "Permanence", "Reference", "Imitation"):
        sub = [v for k, v in means.items() if SUITE[k] == suite]
        if sub:
            print(f"  {suite:<14} n={len(sub)}  {statistics.mean(sub):5.1f}%")
    (ROOT / "baseline_3seed.json").write_text(
        json.dumps({"per_seed": per_seed, "mean": means, "avg": avg}, indent=2) + "\n")
    print(f"\nwrote {ROOT/'baseline_3seed.json'}")
