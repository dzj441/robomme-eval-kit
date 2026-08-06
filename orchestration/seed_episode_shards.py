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
``False`` is safe; ownership at merge time comes from the deterministic global
round-robin formula, not from the sentinel value.

Every shard takes all 16 tasks. Flattening ``task x episode`` into one global
index and assigning ``global_index % N`` rotates each task's remainder across
shards. For 16 tasks x 50 episodes over 16 shards, every shard owns exactly 50
episodes while retaining 3 or 4 episodes from every task.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

SENTINEL = False   # must be summable: eval.py does sum(log_dict[task].values())
TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]


def owner_shard(task_index: int, episode_id: int, episodes: int, num_shards: int) -> int:
    return (task_index * episodes + episode_id) % num_shards


def balanced_owners(
    weights: dict[str, dict[str, float]], episodes: int, num_shards: int
) -> tuple[dict[str, dict[str, int]], list[float]]:
    jobs = []
    for task_index, task in enumerate(TASKS):
        task_weights = weights.get(task)
        if task_weights is None:
            raise ValueError(f"weight profile is missing task {task!r}")
        for episode_id in range(episodes):
            key = str(episode_id)
            if key not in task_weights:
                raise ValueError(
                    f"weight profile is missing {task} episode {episode_id}"
                )
            jobs.append(
                (-float(task_weights[key]), task_index, episode_id, task)
            )

    # Longest-processing-time scheduling. Including shard id in the heap makes
    # ties deterministic, so an ownership file can be recreated exactly.
    heap = [(0.0, shard) for shard in range(num_shards)]
    heapq.heapify(heap)
    owners = {task: {} for task in TASKS}
    loads = [0.0] * num_shards
    for neg_weight, _, episode_id, task in sorted(jobs):
        load, shard = heapq.heappop(heap)
        weight = -neg_weight
        owners[task][str(episode_id)] = shard
        load += weight
        loads[shard] = load
        heapq.heappush(heap, (load, shard))
    return owners, loads


def seed_dynamic_queue(
    root: Path,
    weights: dict[str, dict[str, float]],
    episodes: int,
) -> Path:
    """Create an atomic-file, longest-first episode queue."""
    jobs = []
    for task_index, task in enumerate(TASKS):
        task_weights = weights.get(task)
        if task_weights is None:
            raise ValueError(f"weight profile is missing task {task!r}")
        for episode_id in range(episodes):
            key = str(episode_id)
            if key not in task_weights:
                raise ValueError(
                    f"weight profile is missing {task} episode {episode_id}"
                )
            jobs.append(
                (-float(task_weights[key]), task_index, episode_id, task)
            )

    queue_root = root / "work_queue"
    if queue_root.exists() and any(queue_root.rglob("*.json")):
        raise RuntimeError(f"work queue is not empty: {queue_root}")
    pending_dir = queue_root / "pending"
    (queue_root / "running").mkdir(parents=True, exist_ok=True)
    (queue_root / "done").mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    for rank, (negative_weight, _, episode_id, task) in enumerate(sorted(jobs)):
        payload = {
            "rank": rank,
            "task": task,
            "episode_id": episode_id,
            "weight": -negative_weight,
        }
        filename = f"{rank:04d}__{task}__ep{episode_id:03d}.json"
        (pending_dir / filename).write_text(json.dumps(payload))
    return queue_root


def seed_balanced_steal_queue(
    root: Path,
    weights: dict[str, dict[str, float]],
    owners: dict[str, dict[str, int]],
    episodes: int,
    num_shards: int,
) -> Path:
    """Seed balanced per-worker queues that become stealable at the tail."""
    queue_root = root / "work_queue"
    if queue_root.exists() and any(queue_root.rglob("*.json")):
        raise RuntimeError(f"work queue is not empty: {queue_root}")
    pending_dir = queue_root / "pending"
    (queue_root / "running").mkdir(parents=True, exist_ok=True)
    (queue_root / "done").mkdir(parents=True, exist_ok=True)
    for shard in range(num_shards):
        (pending_dir / f"worker{shard:03d}").mkdir(parents=True, exist_ok=True)

    # Task-major order preserves the efficient execution pattern of the best
    # static run. Dynamic stealing begins only after a worker drains its own
    # balanced list, eliminating the final straggler without globally sorting
    # every worker onto the same expensive task.
    for task_rank, task in enumerate(TASKS):
        for episode_id in range(episodes):
            owner = int(owners[task][str(episode_id)])
            rank = task_rank * episodes + episode_id
            payload = {
                "rank": rank,
                "task": task,
                "episode_id": episode_id,
                "weight": float(weights[task][str(episode_id)]),
                "initial_owner": owner,
            }
            filename = f"{rank:04d}__{task}__ep{episode_id:03d}.json"
            (pending_dir / f"worker{owner:03d}" / filename).write_text(
                json.dumps(payload)
            )
    return queue_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run root holding shard<i>/ dirs")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--policy-name", default="framesamp-modul")
    ap.add_argument("--ckpt-id", type=int, default=79999)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--sharding",
        choices=(
            "global_round_robin_v1",
            "balanced_lpt_v1",
            "dynamic_queue_v1",
            "balanced_steal_v1",
        ),
        default="global_round_robin_v1",
    )
    ap.add_argument("--weight-profile", type=Path)
    a = ap.parse_args()

    root = Path(a.out)
    if a.sharding in (
        "balanced_lpt_v1",
        "dynamic_queue_v1",
        "balanced_steal_v1",
    ):
        if a.weight_profile is None:
            ap.error(f"--weight-profile is required for {a.sharding}")
        profile = json.loads(a.weight_profile.read_text())
        weights = profile.get("weights", profile)
        if a.sharding in ("balanced_lpt_v1", "balanced_steal_v1"):
            owners, estimated_loads = balanced_owners(
                weights, a.episodes, a.num_shards
            )
            if a.sharding == "balanced_steal_v1":
                queue_root = seed_balanced_steal_queue(
                    root,
                    weights,
                    owners,
                    a.episodes,
                    a.num_shards,
                )
        else:
            owners = None
            estimated_loads = None
            queue_root = seed_dynamic_queue(root, weights, a.episodes)
    else:
        owners = {
            task: {
                str(ep): owner_shard(task_index, ep, a.episodes, a.num_shards)
                for ep in range(a.episodes)
            }
            for task_index, task in enumerate(TASKS)
        }
        estimated_loads = None

    ownership = {
        "sharding": a.sharding,
        "num_shards": a.num_shards,
        "episodes": a.episodes,
        "owners": owners,
    }
    (root / "ownership.json").write_text(json.dumps(ownership, indent=1))

    for shard in range(a.num_shards):
        # mirror setup_save_directory's nesting so eval.py finds the file
        d = root / f"shard{shard}" / a.policy_name / f"ckpt{a.ckpt_id}" / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        if a.sharding in ("dynamic_queue_v1", "balanced_steal_v1"):
            progress = {task: {} for task in TASKS}
        else:
            progress = {
                task: {
                    str(ep): SENTINEL
                    for ep in range(a.episodes)
                    if owners[task][str(ep)] != shard
                }
                for task_index, task in enumerate(TASKS)
            }
        (d / "progress.json").write_text(json.dumps(progress, indent=1))
        if a.sharding in ("dynamic_queue_v1", "balanced_steal_v1"):
            worker_kind = (
                "dynamic" if a.sharding == "dynamic_queue_v1" else "balanced-steal"
            )
            print(f"  shard{shard}: {worker_kind} worker")
        else:
            mine_by_task = [
                sum(
                    1 for ep in range(a.episodes)
                    if owners[task][str(ep)] == shard
                )
                for task in TASKS
            ]
            print(
                f"  shard{shard}: {sum(mine_by_task)} episodes "
                f"({min(mine_by_task)}-{max(mine_by_task)} eps/task)"
                + (
                    f" estimated_load={estimated_loads[shard]:.1f}"
                    if estimated_loads is not None
                    else ""
                )
            )
    if a.sharding in ("dynamic_queue_v1", "balanced_steal_v1"):
        job_count = len(list((queue_root / "pending").rglob("*.json")))
        print(f"  queue: {job_count} jobs")


if __name__ == "__main__":
    main()
