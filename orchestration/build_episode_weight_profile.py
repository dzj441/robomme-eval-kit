"""Build an episode-cost profile from completed RoboMME rollout videos."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import cv2

from seed_episode_shards import TASKS


VIDEO_NAME = re.compile(
    r"^(.+)_ep(\d+)_(?:success|fail|timeout|unknown)_"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    samples: dict[tuple[str, int], list[int]] = defaultdict(list)
    for run in args.runs:
        for video in run.rglob("*.mp4"):
            match = VIDEO_NAME.match(video.name)
            if match is None:
                continue
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count <= 0:
                raise RuntimeError(f"could not read frame count from {video}")
            samples[(match.group(1), int(match.group(2)))].append(frame_count)

    weights: dict[str, dict[str, float]] = {}
    for task in TASKS:
        weights[task] = {}
        for episode_id in range(50):
            values = samples.get((task, episode_id), [])
            if not values:
                raise RuntimeError(f"no samples for {task} episode {episode_id}")
            weights[task][str(episode_id)] = sum(values) / len(values)

    payload = {
        "metric": "mean_video_frame_count",
        "source_runs": [str(run.resolve()) for run in args.runs],
        "weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out} from {len(args.runs)} runs")


if __name__ == "__main__":
    main()
