"""Build dynamic-queue weights from measured per-episode wall times."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from seed_episode_shards import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--normalize-cold-start",
        action="store_true",
        help="replace each worker's first (JIT-inflated) timing with its task mean",
    )
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    cold_start_keys: set[tuple[str, int]] = set()
    for run in args.runs:
        for timing_path in run.glob("shard*/**/episode_timings.jsonl"):
            lines = [line for line in timing_path.read_text().splitlines() if line.strip()]
            for line_index, line in enumerate(lines):
                if not line.strip():
                    continue
                timing = json.loads(line)
                key = (timing["task"], int(timing["episode_id"]))
                if args.normalize_cold_start and line_index == 0:
                    cold_start_keys.add(key)
                else:
                    samples[key].append(float(timing["seconds"]))

    task_means = {}
    for task in TASKS:
        task_values = [
            value
            for (sample_task, _), values in samples.items()
            if sample_task == task
            for value in values
        ]
        if task_values:
            task_means[task] = sum(task_values) / len(task_values)

    weights: dict[str, dict[str, float]] = {}
    for task in TASKS:
        weights[task] = {}
        for episode_id in range(50):
            values = samples.get((task, episode_id), [])
            if not values and (task, episode_id) in cold_start_keys:
                values = [task_means[task]]
            if not values:
                raise RuntimeError(f"no timing for {task} episode {episode_id}")
            weights[task][str(episode_id)] = sum(values) / len(values)

    payload = {
        "metric": "mean_episode_wall_seconds",
        "source_runs": [str(run.resolve()) for run in args.runs],
        "weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out} from {len(args.runs)} runs")


if __name__ == "__main__":
    main()
