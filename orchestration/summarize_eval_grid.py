#!/usr/bin/env python3
"""Summarize a checkpoint x seed RoboMME evaluation grid."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_timing(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for checkpoint in args.checkpoints:
        for seed in args.seeds:
            run = args.root / f"ckpt{checkpoint}" / f"seed{seed}"
            aggregate_path = run / "aggregate.json"
            timing = parse_timing(run / "TIMING.txt")
            row: dict[str, object] = {
                "checkpoint": checkpoint,
                "seed": seed,
                "run": str(run),
                "status": timing.get("status", "missing"),
            }
            if timing.get("wall_seconds", "").isdigit():
                row["wall_seconds"] = int(timing["wall_seconds"])
            if aggregate_path.exists():
                aggregate = json.loads(aggregate_path.read_text())
                per_task = aggregate.get("per_task", {})
                errors = sum(aggregate.get("errors", {}).values())
                successes = round(
                    sum(float(rate) * 0.5 for rate in per_task.values())
                )
                row.update(
                    {
                        "tasks": len(per_task),
                        "errors": errors,
                        "successes": successes,
                        "episodes": 800 if len(per_task) == 16 else None,
                        "sr": float(aggregate["avg"]),
                    }
                )
            rows.append(row)

    complete = [
        row
        for row in rows
        if row.get("status") == "complete"
        and row.get("tasks") == 16
        and row.get("errors") == 0
    ]
    by_checkpoint: dict[str, dict[str, object]] = {}
    for checkpoint in args.checkpoints:
        checkpoint_rows = [
            row for row in complete if row["checkpoint"] == checkpoint
        ]
        rates = [float(row["sr"]) for row in checkpoint_rows]
        if rates:
            by_checkpoint[checkpoint] = {
                "runs": len(rates),
                "mean_sr": statistics.mean(rates),
                "std_sr": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            }

    all_rates = [float(row["sr"]) for row in complete]
    payload = {
        "checkpoints": args.checkpoints,
        "seeds": args.seeds,
        "complete_runs": len(complete),
        "expected_runs": len(args.checkpoints) * len(args.seeds),
        "runs": rows,
        "by_checkpoint": by_checkpoint,
        "overall": {
            "mean_sr": statistics.mean(all_rates) if all_rates else None,
            "std_sr": statistics.stdev(all_rates) if len(all_rates) > 1 else 0.0,
        },
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# RoboMME 3-checkpoint x 3-seed evaluation",
        "",
        "| checkpoint | seed | success | SR | wall time | status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        wall = row.get("wall_seconds")
        wall_text = f"{int(wall) // 60}m {int(wall) % 60}s" if wall else "-"
        success_text = (
            f"{row['successes']}/800" if "successes" in row else "-"
        )
        sr_text = f"{float(row['sr']):.3f}%" if "sr" in row else "-"
        lines.append(
            f"| {row['checkpoint']} | {row['seed']} | {success_text} | "
            f"{sr_text} | {wall_text} | {row['status']} |"
        )
    lines.extend(["", "## Checkpoint means", ""])
    for checkpoint in args.checkpoints:
        stats = by_checkpoint.get(checkpoint)
        if stats:
            lines.append(
                f"- `{checkpoint}`: {stats['mean_sr']:.3f}% +/- "
                f"{stats['std_sr']:.3f}% ({stats['runs']}/3 runs)"
            )
        else:
            lines.append(f"- `{checkpoint}`: pending")
    if all_rates:
        lines.extend(
            [
                "",
                f"Overall: {statistics.mean(all_rates):.3f}% +/- "
                f"{(statistics.stdev(all_rates) if len(all_rates) > 1 else 0.0):.3f}% "
                f"({len(all_rates)}/{len(rows)} runs)",
            ]
        )
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
