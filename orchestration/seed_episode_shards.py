"""Pre-seed per-shard progress.json so eval.py runs only its slice of episodes.

eval.py's loop is::

    for episode_id in range(num_episodes):
        if str(episode_id) in log_dict[task_name]:
            continue

It tests key *membership*, not the value, and log_dict is loaded straight from
progress.json. So writing a sentinel for every episode a shard does not own
makes it skip them — episode-level sharding with no change to eval.py.

The sentinel must not be "error" (setup_log_dict strips those and re-runs them)
and must be summable -- eval.py finishes with
``sum(log_dict[task].values())``, so a string sentinel makes that raise, log.json
never gets written, and the outer ``while not exists(log.json)`` spins forever.
``False`` is safe; ownership at merge time comes from ``ep % N == shard``, not
from the sentinel value.

Every shard takes all 16 tasks and the episodes with ``episode_id % N == shard``,
so each shard gets an identical task mix. That removes the makespan floor of
task-level sharding, where the slowest single task (VideoPlaceOrder, ~28 min for
its 50 episodes) bounded the whole run no matter how many shards were added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SENTINEL = False   # must be summable: eval.py does sum(log_dict[task].values())
TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run root holding shard<i>/ dirs")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--policy-name", default="framesamp-modul")
    ap.add_argument("--ckpt-id", type=int, default=79999)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    root = Path(a.out)
    for shard in range(a.num_shards):
        # mirror setup_save_directory's nesting so eval.py finds the file
        d = root / f"shard{shard}" / a.policy_name / f"ckpt{a.ckpt_id}" / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        progress = {
            task: {
                str(ep): SENTINEL
                for ep in range(a.episodes)
                if ep % a.num_shards != shard
            }
            for task in TASKS
        }
        (d / "progress.json").write_text(json.dumps(progress, indent=1))
        mine = sum(1 for ep in range(a.episodes) if ep % a.num_shards == shard)
        print(f"  shard{shard}: {mine} eps/task x {len(TASKS)} tasks = {mine*len(TASKS)}")


if __name__ == "__main__":
    main()
