"""Is the answer linearly readable from the memory the policy is actually given?

Two failure stories predict the same 21% on PickHighlight. Either the evidence
never reaches the memory -- the highlight flashes between two sampled frames --
or it reaches the memory and the model fails to use it. They call for different
fixes, so it is worth separating them before choosing one.

The test: fit the same linear classifier on the same cached SigLIP features under
two views of the history. `samp32` is exactly what FrameSamp hands the policy,
`full` is every frame up to the same instant. If `samp32` trails `full` badly,
sampling is destroying the evidence. If they match -- and both are well above
chance -- the evidence survives sampling and the bottleneck is downstream.

Folds are grouped by episode, so the several decision timesteps drawn from one
episode never straddle train and test.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/datadrive1/dzj/RoboMME/preprocessed")
DATA = ROOT / "data/extracted"
FEAT = ROOT / "features/extracted"
MAX_SIZE = 32

# 1600 training episodes land in contiguous 100-episode blocks, one per task.
BLOCK = {
    "PatternLock": 0, "ButtonUnmaskSwap": 100, "ButtonUnmask": 200,
    "VideoPlaceButton": 300, "VideoUnmask": 400, "PickXtimes": 500,
    "StopCube": 600, "SwingXtimes": 700, "PickHighlight": 800,
    "MoveCube": 900, "InsertPeg": 1000, "RouteStick": 1100,
    "BinFill": 1200, "VideoPlaceOrder": 1300, "VideoRepick": 1400,
    "VideoUnmaskSwap": 1500,
}
# Per task, the pattern that pulls the answer out of the fine-grained subgoal.
# Only a few tasks name their answer in text at all: where the answer is a
# position ("which container"), the subgoal says only "the correct target".
LABEL_RE = {
    "PickHighlight": re.compile(r"which is (red|green|blue)"),
    "MoveCube": re.compile(r"(place) the cube onto|(push) the cube to|(hook) the cube to"),
}


def label_of(task: str, text: str) -> str | None:
    m = LABEL_RE[task].search(text)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def _field(d, k):
    v = d[k]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    v = np.ravel(v)
    return v[0].decode(errors="replace") if isinstance(v[0], bytes) else v[0]


def pkl_index_of_episode(target: int, lo: int, hi: int) -> int:
    """Files are written in episode order, so the start of one is a bisection."""
    while lo < hi:
        mid = (lo + hi) // 2
        if int(_field(pickle.load(open(DATA / f"{mid}.pkl", "rb")), "epis_idx")) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def frame_feats(ep: int, idxs, spatial: bool = False, grid: str = "4x4") -> np.ndarray | None:
    """Mean and max over the given frames of each frame's embedding.

    Max matters as much as mean: evidence appearing in one frame out of 32 is
    diluted 32-fold by averaging but survives a max.

    `spatial` keeps the 4x4 patch grid instead of averaging over it. The model
    receives the patches intact, so pooling them away makes the probe strictly
    weaker than the model's own input -- which matters for answers that are
    partly about *where* something was, like which cube got highlighted.
    """
    vecs = []
    for i in idxs:
        p = FEAT / f"episode_{ep}" / f"token_emb_{int(i)}.npy"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True).item()
        e = np.asarray(d[f"image_emb_{grid}"], dtype=np.float32)[0]  # (P, 2048)
        vecs.append(e.reshape(-1) if spatial else e.mean(axis=0))
    if not vecs:
        return None
    v = np.stack(vecs)
    return np.concatenate([v.mean(axis=0), v.max(axis=0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickHighlight")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--anchor", choices=("label", "exec"), default="label")
    ap.add_argument("--grid", choices=("2x2", "4x4", "8x8"), default="4x4",
                    help="tokens per frame: 4, 16 (the deployed setting) or 64")
    ap.add_argument("--clf", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--views", default="",
                    help="comma-separated subset; `full` costs ~200 frame loads "
                         "per sample and is what makes many-sample runs slow")
    ap.add_argument("--spatial", action="store_true",
                    help="keep the 4x4 patch grid (32768-d), reduced by PCA before fitting")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    base = BLOCK[a.task]
    lo = pkl_index_of_episode(base, 0, 476857)
    hi = pkl_index_of_episode(base + 100, lo, 476857)
    print(f"{a.task}: episodes {base}-{base+99}, pkl {lo}-{hi}", flush=True)

    # walk the block once, recording each episode's label, the step at which the
    # subgoal first names it, and how long the episode runs. Cached, because the
    # walk unpickles tens of thousands of files and the views below get revised.
    cache = Path(f"/datadrive1/dzj/RoboMME/probe_labels_{a.task}.json")
    if cache.exists():
        per_ep = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        print(f"  labels from cache ({len(per_ep)} episodes)", flush=True)
    else:
        per_ep = {}
        for i in range(lo, hi, 3):
            d = pickle.load(open(DATA / f"{i}.pkl", "rb"))
            ep = int(_field(d, "epis_idx"))
            step = int(_field(d, "step_idx"))
            rec = per_ep.setdefault(ep, {"label": None, "step": 10**9, "T": 0, "exec": 0})
            rec["T"] = max(rec["T"], step)
            # Tasks with a demo phase state the answer in the demo's own
            # subgoals, so only execution-phase steps count as the moment the
            # policy has to recall it.
            exec_start = int(_field(d, "exec_start_idx"))
            rec["exec"] = max(rec["exec"], exec_start)
            lab = label_of(a.task, str(_field(d, "simple_subgoal")))
            if lab and step >= exec_start and step < rec["step"]:
                rec.update(label=lab, step=step)
        per_ep = {k: v for k, v in per_ep.items() if v["label"]}
        cache.write_text(json.dumps({str(k): v for k, v in per_ep.items()}, indent=1))
    print(f"  labelled episodes: {len(per_ep)}  "
          f"dist={ dict(Counter(v['label'] for v in per_ep.values())) }",
          flush=True)

    VIEWS = tuple(a.views.split(",")) if a.views else ("cur1", "samp32", "full", "past32")
    X = {v: [] for v in VIEWS}
    y, groups = [], []
    for n_done, (ep, rec) in enumerate(sorted(per_ep.items())[: a.episodes]):
        # Where the decision actually sits differs by task. In PickHighlight the
        # evidence appears mid-episode and the choice follows it, so anchor on
        # the step that names the answer. In MoveCube the evidence is the demo
        # and the choice is made the moment execution starts -- anchoring on the
        # naming step instead would place it after the arm has already picked up
        # the peg, at which point the answer is visible in the current frame.
        t0 = rec["exec"] if a.anchor == "exec" else rec["step"]
        T = rec["T"]
        if t0 >= T:
            continue
        # spread the decision points across the whole execution phase, so the
        # sparse end of the episode is represented and not just the dense start
        for t in np.linspace(t0, max(t0, T - 1), a.per_episode).astype(int):
            keep = (np.arange(t + 1) if t < MAX_SIZE
                    else np.linspace(0, t, MAX_SIZE, dtype=np.int64))
            # cur1 is the control that decides whether any of this is about
            # memory: if one current frame already carries the answer, the
            # probe is reading the observation, not the history.
            past = np.linspace(0, t0 - 1, MAX_SIZE, dtype=np.int64) if t0 > 1 else np.zeros(1, int)
            sp, gr = a.spatial, a.grid
            want = {"cur1": lambda: frame_feats(ep, [t], sp, gr),
                    "samp32": lambda: frame_feats(ep, keep, sp, gr),
                    "full": lambda: frame_feats(ep, np.arange(0, t + 1, 2), sp, gr),
                    "past32": lambda: frame_feats(ep, past, sp, gr)}
            feats = {v: want[v]() for v in VIEWS}
            if any(f is None for f in feats.values()):
                continue
            for v in VIEWS:
                X[v].append(feats[v])
            y.append(rec["label"])
            groups.append(ep)
        if (n_done + 1) % 25 == 0:
            print(f"  ... {n_done+1} episodes, {len(y)} samples", flush=True)

    y = np.array(y)
    groups = np.array(groups)
    n_cls = len(set(y.tolist()))
    chance = max(np.mean(y == c) for c in set(y.tolist()))
    print(f"\nn={len(y)} samples over {len(set(groups.tolist()))} episodes, "
          f"{n_cls} classes, majority baseline {chance*100:.1f}%")

    res = {"task": a.task, "n": int(len(y)), "n_ep": int(len(set(groups.tolist()))),
           "chance": float(chance)}
    for view in VIEWS:
        Xv = np.stack(X[view])
        # p >> n once the patch grid is kept, so project first; PCA is fitted
        # inside the pipeline and therefore inside each CV fold.
        steps = [StandardScaler()]
        if a.spatial:
            steps.append(PCA(n_components=min(120, len(y) - len(y) // 5 - 1),
                             svd_solver="randomized", random_state=0))
        if a.clf == "mlp":
            # Distinguishes a limit of linear read-out from a limit of the
            # representation: if a nonlinear head recovers what a linear one
            # cannot, the information is present and the readout is what fails.
            if not a.spatial:
                steps.append(PCA(n_components=min(120, len(y) - len(y) // 5 - 1),
                                 svd_solver="randomized", random_state=0))
            steps.append(MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2,
                                       max_iter=1500, random_state=0))
        else:
            steps.append(LogisticRegression(C=0.01 if not a.spatial else 1.0, max_iter=3000))
        clf = make_pipeline(*steps)
        sc = cross_val_score(clf, Xv, y, groups=groups,
                             cv=GroupKFold(n_splits=5), n_jobs=1)
        res[view] = {"acc": float(sc.mean()), "sd": float(sc.std()), "dim": int(Xv.shape[1])}
        print(f"  {view:<8} acc={sc.mean()*100:5.1f}% +-{sc.std()*100:4.1f}  (dim {Xv.shape[1]})")

    Path(a.out or f"/datadrive1/dzj/RoboMME/probe_{a.task}_{a.anchor}{'_sp' if a.spatial else ''}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote probe_{a.task}_{a.anchor}_{a.grid}.json")


if __name__ == "__main__":
    main()
