"""Compare episode-level RoboMME outcomes across complete evaluation runs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


EpisodeKey = tuple[str, int]
TASK_ORDER = [
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]
TASK_INDEX = {task: index for index, task in enumerate(TASK_ORDER)}


def shard_index(run: Path, progress_path: Path) -> int:
    return int(progress_path.relative_to(run).parts[0].removeprefix("shard"))


def load_outcomes(run: Path) -> dict[EpisodeKey, Any]:
    outcomes: dict[EpisodeKey, Any] = {}
    sources: dict[EpisodeKey, Path] = {}
    progress_paths = sorted(
        run.glob("shard*/**/progress.json"),
        key=lambda path: shard_index(run, path),
    )
    if not progress_paths:
        raise FileNotFoundError(f"no shard progress files below {run}")

    run_config_path = run / "run_config.txt"
    run_config = run_config_path.read_text() if run_config_path.exists() else ""
    global_round_robin = "sharding=global_round_robin_v1" in run_config
    episodes_per_task = 50
    for field in run_config.split():
        if field.startswith("episodes="):
            episodes_per_task = int(field.split("=", 1)[1])
            break

    ownership_path = run / "ownership.json"
    ownership_payload = (
        json.loads(ownership_path.read_text()) if ownership_path.exists() else {}
    )
    ownership = ownership_payload.get("owners")
    dynamic_queue = ownership_payload.get("sharding") in (
        "dynamic_queue_v1",
        "balanced_steal_v1",
    )

    for progress_path in progress_paths:
        shard = shard_index(run, progress_path)
        progress = json.loads(progress_path.read_text())
        for task, episodes in progress.items():
            for episode_id, value in episodes.items():
                episode = int(episode_id)
                if dynamic_queue:
                    owned = True
                elif ownership is not None:
                    owned = int(ownership[task][episode_id]) == shard
                elif len(progress_paths) == 1:
                    owned = True
                elif global_round_robin:
                    global_index = TASK_INDEX[task] * episodes_per_task + episode
                    owned = global_index % len(progress_paths) == shard
                else:
                    owned = episode % len(progress_paths) == shard
                if value == "skip" or not owned:
                    continue
                key = (task, episode)
                if key in outcomes and outcomes[key] != value:
                    raise RuntimeError(
                        f"conflicting result for {task} episode {episode_id}: "
                        f"{outcomes[key]!r} in {sources[key]} and "
                        f"{value!r} in {progress_path}"
                    )
                outcomes[key] = value
                sources[key] = progress_path
    return outcomes


def summarize(outcomes: dict[EpisodeKey, Any]) -> dict[str, int]:
    return {
        "episodes": len(outcomes),
        "successes": sum(value is True for value in outcomes.values()),
        "failures": sum(value is False for value in outcomes.values()),
        "errors": sum(value not in (True, False) for value in outcomes.values()),
    }


def compare(
    reference: dict[EpisodeKey, Any], candidate: dict[EpisodeKey, Any]
) -> dict[str, Any]:
    shared = sorted(reference.keys() & candidate.keys())
    flips = [key for key in shared if reference[key] != candidate[key]]
    return {
        "shared_episodes": len(shared),
        "flips": len(flips),
        "reference_only": len(reference.keys() - candidate.keys()),
        "candidate_only": len(candidate.keys() - reference.keys()),
        "reference_success_candidate_failure": sum(
            reference[key] is True and candidate[key] is False for key in flips
        ),
        "reference_failure_candidate_success": sum(
            reference[key] is False and candidate[key] is True for key in flips
        ),
        "flip_episodes": [f"{task}:{episode_id}" for task, episode_id in flips],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("provide a reference run and at least one candidate run")

    loaded = [load_outcomes(run) for run in args.runs]
    all_keys = set().union(*(outcomes.keys() for outcomes in loaded))
    varied_keys = sorted(
        key
        for key in all_keys
        if len({outcomes[key] for outcomes in loaded if key in outcomes}) > 1
    )
    payload = {
        "reference": str(args.runs[0]),
        "runs": [
            {"path": str(path), **summarize(outcomes)}
            for path, outcomes in zip(args.runs, loaded, strict=True)
        ],
        "comparisons": [
            {
                "candidate": str(args.runs[index]),
                **compare(loaded[0], loaded[index]),
            }
            for index in range(1, len(loaded))
        ],
        "all_pairwise_comparisons": [
            {
                "reference": str(args.runs[left]),
                "candidate": str(args.runs[right]),
                **compare(loaded[left], loaded[right]),
            }
            for left, right in itertools.combinations(range(len(loaded)), 2)
        ],
        "episodes_with_any_variation": len(varied_keys),
        "varied_episodes": [
            f"{task}:{episode_id}" for task, episode_id in varied_keys
        ],
    }

    print(json.dumps(payload, indent=2))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
