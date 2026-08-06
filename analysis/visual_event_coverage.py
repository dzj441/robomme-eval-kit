"""Does uniform sampling cover the moments where the scene visibly changes?

The subgoal-segment analysis answers this at the granularity of behaviour, where
segments run 50-100 frames and nothing is ever missed. But a highlight flashing
on, or a container dropping over a cube, is a visual event far shorter than the
behaviour containing it, and the subgoal never moves for it.

So find those events without labels: consecutive frames of the cached SigLIP
embeddings, and the moments where the frame-to-frame distance spikes. Then ask
what fraction of them fall within half a sampling interval of a kept frame --
close enough that the memory holds a frame showing the changed scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEAT = Path("/datadrive1/dzj/RoboMME/preprocessed/features/extracted")
MAX_SIZE = 32
BLOCK = {
    "PatternLock": 0, "ButtonUnmaskSwap": 100, "ButtonUnmask": 200,
    "VideoPlaceButton": 300, "VideoUnmask": 400, "PickXtimes": 500,
    "StopCube": 600, "SwingXtimes": 700, "PickHighlight": 800,
    "MoveCube": 900, "InsertPeg": 1000, "RouteStick": 1100,
    "BinFill": 1200, "VideoPlaceOrder": 1300, "VideoRepick": 1400,
    "VideoUnmaskSwap": 1500,
}


def episode_embeddings(ep: int) -> np.ndarray | None:
    d = FEAT / f"episode_{ep}"
    if not d.is_dir():
        return None
    steps = sorted((int(p.stem.split("_")[-1]) for p in d.glob("token_emb_*.npy")))
    if len(steps) < 8:
        return None
    out = []
    for t in steps:
        e = np.load(d / f"token_emb_{t}.npy", allow_pickle=True).item()
        out.append(np.asarray(e["image_emb_4x4"], dtype=np.float32)[0].mean(axis=0))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", default="/datadrive1/dzj/RoboMME/visual_events.json")
    a = ap.parse_args()

    out = {}
    for task in (a.tasks.split(",") if a.tasks else BLOCK):
        base = BLOCK[task]
        rows = []
        for ep in range(base, base + a.episodes):
            F = episode_embeddings(ep)
            if F is None:
                continue
            T = len(F)
            d = np.linalg.norm(np.diff(F, axis=0), axis=1)
            thr = d.mean() + a.sigma * d.std()
            events = np.flatnonzero(d > thr) + 1
            if not len(events):
                rows.append({"T": T, "n_ev": 0, "covered": 0, "dur": 0.0,
                             "n_state": 1, "inside": 1, "long_n": 1, "long_hit": 1,
                             "short_n": 0, "short_hit": 0, "state_len": float(T),
                             "state_min": T, "gap_med": 0.0})
                continue
            keep = np.linspace(0, T - 1, MAX_SIZE, dtype=np.int64)
            half = (T - 1) / (MAX_SIZE - 1) / 2
            # Nearness is the right test for a change that persists -- a
            # container drops and stays down, so a frame from shortly after it
            # still shows the new scene.
            gap = np.abs(events[:, None] - keep[None, :]).min(axis=1)
            covered = int((gap <= half).sum())
            # Nearness is *not* enough for a transient: a highlight that flashes
            # on and off leaves the scene unchanged a few frames later. So also
            # treat consecutive events as the boundaries of a visual state and
            # ask, strictly, whether any kept frame falls inside it -- the same
            # question the subgoal analysis asks, at visual granularity.
            bounds = np.concatenate([[0], events, [T]])
            states = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
                      if bounds[i + 1] > bounds[i]]
            hit = [bool(((keep >= lo) & (keep < hi)).any()) for lo, hi in states]
            inside = sum(hit)
            # A state one or two frames long is a blip between two detector
            # spikes, and missing it means nothing. Stratify by duration so the
            # claim is about states long enough to carry a fact.
            long_n = sum(1 for (lo, hi) in states if hi - lo >= 10)
            long_hit = sum(h for h, (lo, hi) in zip(hit, states) if hi - lo >= 10)
            short_n = len(states) - long_n
            short_hit = inside - long_hit
            runs, cur = [], 1
            for i in range(1, len(events)):
                if events[i] == events[i - 1] + 1:
                    cur += 1
                else:
                    runs.append(cur); cur = 1
            runs.append(cur)
            rows.append({"T": T, "n_ev": len(events), "covered": covered,
                         "n_state": len(states), "inside": inside,
                         "long_n": long_n, "long_hit": long_hit,
                         "short_n": short_n, "short_hit": short_hit,
                         "state_len": float(np.mean([hi - lo for lo, hi in states])),
                         "state_min": int(min(hi - lo for lo, hi in states)),
                         "dur": float(np.mean(runs)), "gap_med": float(np.median(gap))})
        if not rows:
            continue
        n_ev = sum(r["n_ev"] for r in rows)
        agg = {
            "n_ep": len(rows), "T": float(np.mean([r["T"] for r in rows])),
            "events_per_ep": n_ev / len(rows),
            "event_dur": float(np.mean([r["dur"] for r in rows if r["n_ev"]]) or 0),
            "covered": (sum(r["covered"] for r in rows) / n_ev) if n_ev else float("nan"),
            "state_missed": 1 - sum(r["inside"] for r in rows) / sum(r["n_state"] for r in rows),
            "state_len": float(np.mean([r["state_len"] for r in rows])),
            "state_min": float(np.mean([r["state_min"] for r in rows])),
            "missed_long": 1 - sum(r["long_hit"] for r in rows) / max(1, sum(r["long_n"] for r in rows)),
            "missed_short": 1 - sum(r["short_hit"] for r in rows) / max(1, sum(r["short_n"] for r in rows)),
            "n_long": sum(r["long_n"] for r in rows) / len(rows),
            "n_short": sum(r["short_n"] for r in rows) / len(rows),
            "gap_med": float(np.mean([r["gap_med"] for r in rows if r["n_ev"]]) or 0),
            "spacing": float(np.mean([(r["T"] - 1) / (MAX_SIZE - 1) for r in rows])),
        }
        out[task] = agg
        print(f"{task:<19} T={agg['T']:6.0f} spacing={agg['spacing']:5.1f} "
              f"events/ep={agg['events_per_ep']:5.1f} dur={agg['event_dur']:4.1f}f "
              f"| >=10f: n={agg['n_long']:5.1f} missed={agg['missed_long']*100:5.1f}%  "
              f"<10f: n={agg['n_short']:5.1f} missed={agg['missed_short']*100:5.1f}%", flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
