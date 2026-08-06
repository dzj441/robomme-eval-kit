"""Split the 61% memory decodability into per-frame encoding vs aggregation.

The aggregate probe (mean+max over 32 sampled frames) reads the highlighted
colour at ~61%. Two very different failures produce that number: the frozen
SigLIP embedding of a single ring-visible frame barely encodes which cube the
ring is under, or the per-frame encoding is strong and pooling across 32 frames
dilutes it. The fix differs accordingly -- representation vs aggregation -- so
probe single frames from inside the ring window, no cross-frame pooling at all.

Ring window: in PickHighlight the ring is visible from t=0 until the button
press, and the label step t0 (first subgoal naming the colour) sits just after
the press. Frames in [0, t0 - margin] are ring-visible; a margin absorbs the
few-frame slack between press and subgoal switch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT = Path("/datadrive1/dzj/RoboMME/preprocessed/features/extracted")
LABELS = Path("/datadrive1/dzj/RoboMME/probe_labels_PickHighlight.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-ep", type=int, default=6)
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--out", default="/datadrive1/dzj/RoboMME/probe_single_frame.json")
    a = ap.parse_args()

    per_ep = {int(k): v for k, v in json.loads(LABELS.read_text()).items()}
    print(f"labels: {len(per_ep)} episodes", flush=True)

    X_pool, X_flat, y, groups, used_t = [], [], [], [], []
    for ep, rec in sorted(per_ep.items()):
        hi = rec["step"] - a.margin
        if hi < 4:
            continue
        for t in np.linspace(0, hi, a.frames_per_ep).astype(int):
            f = FEAT / f"episode_{ep}" / f"token_emb_{int(t)}.npy"
            if not f.exists():
                continue
            e = np.load(f, allow_pickle=True).item()
            g = np.asarray(e["image_emb_4x4"], dtype=np.float32)[0]     # (16, 2048)
            X_pool.append(g.mean(axis=0))                                # 2048
            X_flat.append(g.reshape(-1))                                 # 32768
            y.append(rec["label"])
            groups.append(ep)
            used_t.append(int(t))

    y = np.array(y); groups = np.array(groups)
    chance = max(np.mean(y == c) for c in set(y.tolist()))
    print(f"n={len(y)} single frames over {len(set(groups.tolist()))} episodes, "
          f"majority {chance*100:.1f}%, frame t: min={min(used_t)} med={int(np.median(used_t))} "
          f"max={max(used_t)}", flush=True)

    res = {"n": int(len(y)), "chance": float(chance),
           "aggregate_reference": {"samp32_linear": 0.615, "note": "32-frame mean+max, C=1.0 spatial"}}
    for name, X, spatial in (("pooled_2048", np.stack(X_pool), False),
                             ("spatial_32768", np.stack(X_flat), True)):
        steps = [StandardScaler()]
        if spatial:
            steps.append(PCA(n_components=min(120, len(y) - len(y) // 5 - 1),
                             svd_solver="randomized", random_state=0))
        steps.append(LogisticRegression(C=1.0, max_iter=3000))
        sc = cross_val_score(make_pipeline(*steps), X, y, groups=groups,
                             cv=GroupKFold(n_splits=5), n_jobs=1)
        res[name] = {"acc": float(sc.mean()), "sd": float(sc.std())}
        print(f"  single-frame {name:<14} acc={sc.mean()*100:5.1f}% +-{sc.std()*100:4.1f}",
              flush=True)

    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n>=90%: per-frame encoding is fine and the deficit is aggregation/readout."
          f"\n~60%:  the frozen encoder itself barely separates the binding -- representation ceiling."
          f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
