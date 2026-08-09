#!/usr/bin/env python3
"""Summarize RoboMME checkpoint evaluations and plot SR versus train step."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


def read_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = []
    for checkpoint_text in args.checkpoints:
        checkpoint = int(checkpoint_text)
        run = args.root / f"ckpt{checkpoint_text}" / f"seed{args.seed}"
        aggregate_path = run / "aggregate.json"
        timing_path = run / "TIMING.txt"
        if not aggregate_path.is_file() or not timing_path.is_file():
            continue
        aggregate = json.loads(aggregate_path.read_text())
        timing = read_key_values(timing_path)
        if timing.get("status") != "complete":
            continue
        per_task = aggregate.get("per_task", {})
        errors = aggregate.get("errors", {})
        if len(per_task) != 16 or sum(errors.values()) != 0:
            continue
        successes = int(round(sum(float(value) * 0.5 for value in per_task.values())))
        sr = successes / 800 * 100
        rows.append(
            {
                "checkpoint": checkpoint,
                "train_step": 80000 if checkpoint == 79999 else checkpoint,
                "seed": args.seed,
                "successes": successes,
                "episodes": 800,
                "sr": sr,
                "wall_seconds": int(timing["wall_seconds"]),
                "run": str(run.resolve()),
            }
        )

    rows.sort(key=lambda row: row["train_step"])
    payload = {
        "seed": args.seed,
        "expected_checkpoints": [int(value) for value in args.checkpoints],
        "complete_runs": len(rows),
        "runs": rows,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "curve.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (args.root / "curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else ["checkpoint"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# RoboMME seed {args.seed}: SR vs. training step",
        "",
        "| checkpoint | plotted step | success | SR | wall time |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['train_step']} | "
            f"{row['successes']}/800 | {row['sr']:.3f}% | {row['wall_seconds']} s |"
        )
    if rows:
        lines.extend(
            [
                "",
                f"Mean across completed checkpoints: {statistics.mean(row['sr'] for row in rows):.3f}%",
            ]
        )
    (args.root / "CURVE.md").write_text("\n".join(lines) + "\n")

    if rows:
        steps = [row["train_step"] for row in rows]
        rates = [row["sr"] for row in rows]
        figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        axis.plot(steps, rates, marker="o", linewidth=2.0, color="#2563eb", label=f"seed {args.seed}")
        axis.axhline(44.51, color="#64748b", linestyle="--", linewidth=1.2, label="paper mean 44.51%")
        for step, rate in zip(steps, rates):
            axis.annotate(f"{rate:.2f}%", (step, rate), xytext=(0, 7), textcoords="offset points", ha="center")
        axis.set_xlabel("Training step")
        axis.set_ylabel("RoboMME success rate (%)")
        axis.set_title("RoboMME SR vs. training step")
        axis.set_xticks(steps)
        axis.grid(True, alpha=0.25)
        axis.legend()
        margin = max(1.5, (max(rates) - min(rates)) * 0.25)
        axis.set_ylim(max(0, min(rates + [44.51]) - margin), min(100, max(rates + [44.51]) + margin))
        figure.savefig(args.root / "sr_vs_step.png", dpi=180)
        figure.savefig(args.root / "sr_vs_step.svg")
        plt.close(figure)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
