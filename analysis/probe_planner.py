"""Probe the motion-planner path before committing to a long sweep.

DemonstrationWrapper.reset() unconditionally builds a mplib motion planner and,
for tasks whose task_list carries demonstration subtasks, actually plans. mplib
0.1.1 is known to segfault there (upstream PR #14, unmerged), and a segfault is
a process-level crash that no Python handler can catch — it would kill a shard
mid-sweep.

This exercises exactly that path: reset() only, no policy rollout. Each task runs
in its own subprocess so a crash is attributed to a task instead of taking the
probe down with it.
"""

from __future__ import annotations

import subprocess
import sys
import time

TASKS_WITH_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
TASKS_NO_DEMO = ["BinFill", "StopCube", "PickXtimes", "SwingXtimes",
                 "ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight"]

CHILD = r'''
import sys, time
sys.path.insert(0, "/home/qid/dzj/RoboMME_policy/examples/robomme")
from env_runner import EnvRunner
task = sys.argv[1]; n = int(sys.argv[2])
r = EnvRunner(task, "/tmp/probe_videos", max_steps=50)
for ep in range(n):
    t0 = time.time()
    r.make_env(ep)
    obs = r.get_init_obs()
    print(f"  ep{ep} reset ok in {time.time()-t0:.1f}s", flush=True)
    r.close_env()
print("CHILD_OK", flush=True)
'''

PY = "/home/qid/dzj/miniconda3/envs/robomme/bin/python"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2

results = {}
for group, tasks in (("with-demo (走 planner)", TASKS_WITH_DEMO),
                     ("no-demo", TASKS_NO_DEMO)):
    print(f"\n{'='*70}\n{group}\n{'='*70}", flush=True)
    for task in tasks:
        t0 = time.time()
        p = subprocess.run([PY, "-c", CHILD, task, str(N)],
                           capture_output=True, text=True, timeout=1800)
        dt = time.time() - t0
        ok = "CHILD_OK" in p.stdout
        # negative returncode == killed by signal; -11 is SIGSEGV
        if p.returncode < 0:
            verdict = f"CRASH signal {-p.returncode}" + (" (SIGSEGV)" if p.returncode == -11 else "")
        elif ok:
            verdict = "ok"
        else:
            verdict = f"fail rc={p.returncode}"
        results[task] = verdict
        print(f"{task:<20} {verdict:<28} {dt:6.1f}s", flush=True)
        if verdict != "ok":
            tail = (p.stderr or p.stdout).strip().splitlines()[-4:]
            for line in tail:
                print("      " + line[:150], flush=True)

print(f"\n{'='*70}\nSUMMARY")
bad = {k: v for k, v in results.items() if v != "ok"}
print(f"  {len(results)-len(bad)}/{len(results)} tasks reset cleanly")
for k, v in bad.items():
    print(f"  FAIL {k}: {v}")
