"""Compare the memory interventions against the seed-7 baseline, per task.

Everything here is the same checkpoint, the same seed and the same 800 episodes,
so a task's difference between two columns is attributable to the intervention
alone -- no retraining, no checkpoint variance. The baseline's own seed-to-seed
spread (sd 0.9 on AVG over three seeds) is what a difference has to clear to mean
anything, and per task at n=50 the binomial standard error is about 7 points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/datadrive1/dzj/RoboMME/eval_out")
BASE = ROOT / "ultra3_seed7"
MODES = ["off", "repeatcur", "timeshuf", "budget8", "budget128", "frames128", "primacy"]
# Paper Table 3: pi0.5 conditioned on the current observation only.
PI05_AVG = 17.93

SUITE = {
    "BinFill": "Count", "PickXtimes": "Count", "SwingXtimes": "Count", "StopCube": "Count",
    "VideoUnmask": "Perm", "ButtonUnmask": "Perm",
    "VideoUnmaskSwap": "Perm", "ButtonUnmaskSwap": "Perm",
    "PickHighlight": "Ref", "VideoRepick": "Ref",
    "VideoPlaceButton": "Ref", "VideoPlaceOrder": "Ref",
    "MoveCube": "Imit", "InsertPeg": "Imit", "PatternLock": "Imit", "RouteStick": "Imit",
}


def read_run(run: Path) -> dict[str, tuple[int, int]]:
    """Per-task (successes, episodes), counting only the episodes each shard owns."""
    shards = sorted(run.glob("shard*/**/progress.json"),
                    key=lambda p: int(str(p).split("shard")[1].split("/")[0]))
    n = len(shards)
    owned: dict[str, dict[str, object]] = {}
    for prog in shards:
        sh = int(str(prog).split("shard")[1].split("/")[0])
        for task, eps in json.loads(prog.read_text()).items():
            for ep, v in eps.items():
                if n == 1 or int(ep) % n == sh:
                    owned.setdefault(task, {})[ep] = v
    return {t: (sum(1 for v in e.values() if v is True), len(e)) for t, e in owned.items()}


def main() -> None:
    base = read_run(BASE)
    runs = {m: read_run(ROOT / f"interv_{m}") for m in MODES
            if (ROOT / f"interv_{m}").exists()}
    runs = {m: r for m, r in runs.items() if r}
    if not runs:
        print("no intervention runs found yet")
        return

    cols = list(runs)
    print("=" * (34 + 16 * len(cols)))
    print("RoboMME test-time memory interventions — FrameSamp+Modul ckpt 79999, seed 7")
    print("=" * (34 + 16 * len(cols)))
    head = "".join(f"{m:>16}" for m in cols)
    print(f"{'task':<19}{'suite':>6}{'base':>7}{head}")
    print("-" * (34 + 16 * len(cols)))

    totals = {m: [] for m in cols}
    base_tot = []
    for task in SUITE:
        if task not in base:
            continue
        bs, bn = base[task]
        b = 100.0 * bs / bn
        base_tot.append(b)
        cells = ""
        for m in cols:
            if task in runs[m] and runs[m][task][1]:
                s, n = runs[m][task]
                v = 100.0 * s / n
                totals[m].append(v)
                cells += f"{v:10.1f}{v - b:+6.1f}"
            else:
                cells += f"{'-':>16}"
        print(f"{task:<19}{SUITE[task]:>6}{b:7.1f}{cells}")

    print("-" * (34 + 16 * len(cols)))
    cells = ""
    for m in cols:
        if len(totals[m]) == len(base_tot):
            avg = sum(totals[m]) / len(totals[m])
            cells += f"{avg:10.1f}{avg - sum(base_tot)/len(base_tot):+6.1f}"
        else:
            cells += f"{'(partial)':>16}"
    print(f"{'AVG':<19}{'':>6}{sum(base_tot)/len(base_tot):7.1f}{cells}")

    for suite in ("Count", "Perm", "Ref", "Imit"):
        idx = [i for i, t in enumerate(t for t in SUITE if t in base) if SUITE[list(SUITE)[i]] == suite]
        keys = [t for t in SUITE if t in base and SUITE[t] == suite]
        bsub = sum(100.0 * base[t][0] / base[t][1] for t in keys) / len(keys)
        cells = ""
        for m in cols:
            got = [t for t in keys if t in runs[m] and runs[m][t][1]]
            if len(got) == len(keys):
                v = sum(100.0 * runs[m][t][0] / runs[m][t][1] for t in keys) / len(keys)
                cells += f"{v:10.1f}{v - bsub:+6.1f}"
            else:
                cells += f"{'-':>16}"
        print(f"  {suite:<17}{'':>6}{bsub:7.1f}{cells}")

    print(f"\nn=50/task; binomial SE ~{100*math.sqrt(0.25/50):.1f}pp per task, "
          f"~{100*math.sqrt(0.25/800):.1f}pp on AVG")
    print(f"reference: paper's pi0.5 without any memory scores {PI05_AVG} AVG "
          f"(a separately trained model, not this checkpoint with memory removed)")

    # How a run fails matters as much as how often: a drop that turns wrong
    # decisions into exhausted step budgets is a different failure than one that
    # leaves the policy decisive but wrong.
    print(f"\n{'run':<12}{'success':>9}{'fail':>8}{'timeout':>9}{'timeout%':>10}")
    for name, run in [("base", BASE)] + [(m, ROOT / f"interv_{m}") for m in cols]:
        vids = list(run.glob("shard*/**/videos/*.mp4"))
        if not vids:
            continue
        n_s = sum(1 for p in vids if "_success_" in p.name)
        n_t = sum(1 for p in vids if "_timeout_" in p.name)
        n_f = len(vids) - n_s - n_t
        print(f"{name:<12}{n_s:>9}{n_f:>8}{n_t:>9}{100*n_t/max(1,len(vids)-n_s):>9.1f}%")
    (ROOT / "interventions.json").write_text(json.dumps(
        {"base": {t: base[t] for t in base},
         **{m: {t: runs[m][t] for t in runs[m]} for m in cols}}, indent=2) + "\n")
    print(f"wrote {ROOT/'interventions.json'}")


if __name__ == "__main__":
    main()
