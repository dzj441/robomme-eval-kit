"""Count completed owned episodes in a sharded RoboMME run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()

    ownership_path = args.run / "ownership.json"
    if not ownership_path.exists():
        print(sum(1 for _ in args.run.rglob("*.mp4")))
        return

    ownership = json.loads(ownership_path.read_text())
    if ownership.get("sharding") in (
        "dynamic_queue_v1",
        "balanced_steal_v1",
    ):
        completed = set()
        for progress_path in args.run.glob("shard*/**/progress.json"):
            progress = json.loads(progress_path.read_text())
            for task, episodes in progress.items():
                completed.update((task, episode_id) for episode_id in episodes)
        print(len(completed))
        return

    owners = ownership["owners"]
    completed = 0
    for progress_path in args.run.glob("shard*/**/progress.json"):
        shard = int(progress_path.relative_to(args.run).parts[0].removeprefix("shard"))
        progress = json.loads(progress_path.read_text())
        for task, episodes in progress.items():
            for episode_id in episodes:
                if int(owners[task][episode_id]) == shard:
                    completed += 1
    print(completed)


if __name__ == "__main__":
    main()
