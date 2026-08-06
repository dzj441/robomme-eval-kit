"""Measure what uniform frame sampling actually keeps, per task.

The claim worth testing is that `linspace(0, t, 32)` drops the evidence a task
depends on. That is only checkable against some notion of "when did something
happen", and the dataset supplies one: `info/simple_subgoal` changes exactly when
the episode moves to a new sub-behaviour, so its segments are the events.

Two numbers come out of it. Spacing is the memory's temporal resolution at the
moment an answer is needed. Missed-segment rate is the fraction of events that
receive no sampled frame at all -- an event the memory cannot contain, however
well the model reads it. A task can fail either because its evidence was dropped
(high missed rate) or in spite of the evidence being present (near-zero missed
rate), and those point at different fixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
MAX_SIZE = 32  # budget 512 / (token_per_image 16 * num_views 1)


def episode_events(grp):
    """Subgoal segment starts, the demo mask, and the episode length."""
    steps = sorted((k for k in grp if k.startswith("timestep_")),
                   key=lambda k: int(k.split("_")[1]))
    subs, demo = [], []
    for k in steps:
        info = grp[k]["info"]
        s = info["simple_subgoal"][()]
        subs.append(s.decode(errors="replace") if isinstance(s, bytes) else str(s))
        demo.append(bool(np.asarray(info["is_video_demo"])))
    starts = [0] + [i for i in range(1, len(subs)) if subs[i] != subs[i - 1]]
    return starts, np.array(demo), len(subs)


def sampled(step_idx: int) -> np.ndarray:
    if step_idx < MAX_SIZE:
        return np.arange(step_idx + 1)
    return np.linspace(0, step_idx, MAX_SIZE, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/datadrive1/dzj/RoboMME/dataset")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="/datadrive1/dzj/RoboMME/sampling_coverage.json")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()

    out = {}
    for task in (a.tasks.split(",") if a.tasks else TASKS):
        path = Path(a.dataset) / f"record_dataset_{task}.h5"
        if not path.exists():
            continue
        rows = []
        with h5py.File(path, "r") as f:
            eps = sorted((k for k in f if k.startswith("episode_")),
                         key=lambda k: int(k.split("_")[1]))[: a.episodes]
            for ep in eps:
                starts, demo, T = episode_events(f[ep])
                # Half the tasks carry no demo phase at all -- the evidence sits
                # in the opening execution frames instead. So score coverage at
                # the last step, where the history is longest and every earlier
                # event has had to survive; that is the worst case the policy
                # actually faces, and it is defined the same way for every task.
                last = T - 1
                keep = sampled(last)
                bounds = starts + [T]
                segs = [(bounds[i], bounds[i + 1]) for i in range(len(starts))]
                missed = sum(1 for lo, hi in segs
                             if not ((keep >= lo) & (keep < hi)).any())
                exec_start = int(np.argmin(demo)) if demo.any() else 0
                rows.append({
                    "T": T, "exec_start": exec_start, "demo": int(demo.sum()),
                    "n_seg": len(segs), "missed": missed,
                    "seg_len": float(np.mean([hi - lo for lo, hi in segs])),
                    "seg_min": int(min(hi - lo for lo, hi in segs)),
                    "spacing": last / (MAX_SIZE - 1) if last >= MAX_SIZE else 1.0,
                    "spacing_end": (T - 1) / (MAX_SIZE - 1),
                })
        if not rows:
            continue
        agg = {
            "n_ep": len(rows),
            "T": float(np.mean([r["T"] for r in rows])),
            "demo_frac": float(np.mean([r["demo"] / r["T"] for r in rows])),
            "exec_start": float(np.mean([r["exec_start"] for r in rows])),
            "spacing_at_decision": float(np.mean([r["spacing"] for r in rows])),
            "spacing_at_end": float(np.mean([r["spacing_end"] for r in rows])),
            "seg_per_ep": float(np.mean([r["n_seg"] for r in rows])),
            "seg_len": float(np.mean([r["seg_len"] for r in rows])),
            "seg_min": float(np.mean([r["seg_min"] for r in rows])),
            "missed_rate": float(np.sum([r["missed"] for r in rows])
                                 / max(1, np.sum([r["n_seg"] for r in rows]))),
        }
        out[task] = agg
        print(f"{task:<19} T={agg['T']:6.0f} demo={agg['demo_frac']*100:4.0f}% "
              f"spacing={agg['spacing_at_end']:6.1f} seg={agg['seg_per_ep']:5.1f} "
              f"seglen={agg['seg_len']:6.1f} segmin={agg['seg_min']:6.1f} "
              f"missed={agg['missed_rate']*100:5.1f}%", flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
