"""Is the 58.2% single-frame number an artifact of the probe's own choices?

Three attackable choices, swept on identical data (same 600 frames, same labels,
same episode-grouped folds):

  PCA        variance-based selection could discard low-variance discriminative
             directions and so *underestimate* -- drop it entirely, and sweep
             its dimension when kept. L2 logistic regression is a well-posed
             convex problem at p >> n, so no reduction is required.
  C          one regularisation strength is a choice; sweep four orders.
  optimiser  LBFGS-LR and an SGD-trained linear head share the hypothesis
             class, so this axis cannot change what is *representable* -- it is
             covered by the C sweep.

An MLP row is included as reference only: probing literature (control tasks)
reads expressive-probe gains as decoder capacity, not representation content.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT = Path("/datadrive1/dzj/RoboMME/preprocessed/features/extracted")
LABELS = Path("/datadrive1/dzj/RoboMME/probe_labels_PickHighlight.json")
MARGIN, PER_EP = 12, 6

per_ep = {int(k): v for k, v in json.loads(LABELS.read_text()).items()}
X, y, groups = [], [], []
for ep, rec in sorted(per_ep.items()):
    hi = rec["step"] - MARGIN
    if hi < 4:
        continue
    for t in np.linspace(0, hi, PER_EP).astype(int):
        f = FEAT / f"episode_{ep}" / f"token_emb_{int(t)}.npy"
        if f.exists():
            e = np.load(f, allow_pickle=True).item()
            X.append(np.asarray(e["image_emb_4x4"], dtype=np.float32)[0].reshape(-1))
            y.append(rec["label"]); groups.append(ep)
X = np.stack(X); y = np.array(y); groups = np.array(groups)
print(f"n={len(y)} frames, {len(set(groups.tolist()))} episodes, dim={X.shape[1]}, "
      f"majority {max(np.mean(y==c) for c in set(y.tolist()))*100:.1f}%", flush=True)

def run(name, steps):
    sc = cross_val_score(make_pipeline(*steps), X, y, groups=groups,
                         cv=GroupKFold(n_splits=5), n_jobs=1)
    print(f"  {name:<34} acc={sc.mean()*100:5.1f}% +-{sc.std()*100:4.1f}", flush=True)
    return float(sc.mean())

res = {}
print("\n-- no PCA, full 32768-d, C sweep --", flush=True)
for C in (0.001, 0.01, 0.1, 1.0, 10.0):
    res[f"noPCA_C{C}"] = run(f"LR(C={C}) raw 32768-d",
                             [StandardScaler(), LogisticRegression(C=C, max_iter=5000)])

print("\n-- PCA dimension sweep (C=1.0) --", flush=True)
for d in (60, 120, 240, 400):
    res[f"PCA{d}"] = run(f"PCA({d}) -> LR(C=1.0)",
                         [StandardScaler(),
                          PCA(n_components=d, svd_solver="randomized", random_state=0),
                          LogisticRegression(C=1.0, max_iter=5000)])

print("\n-- expressive reference (capacity, not content) --", flush=True)
res["mlp"] = run("PCA(120) -> MLP(256)",
                 [StandardScaler(),
                  PCA(n_components=120, svd_solver="randomized", random_state=0),
                  MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2,
                                max_iter=2000, random_state=0)])

Path("/datadrive1/dzj/RoboMME/probe_sensitivity.json").write_text(
    json.dumps(res, indent=2) + "\n")
print("\nIf every row sits in the mid-50s to low-60s, the number is a property of "
      "the features, not of PCA or C. A no-PCA row well above the PCA rows would "
      "mean PCA had been eating signal and the earlier figures underestimate.")
