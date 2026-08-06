"""Do the cached training features match what the eval path computes at run time?

Training reads `features/extracted/episode_N/token_emb_T.npy`. Evaluation never
touches them: it runs the simulator and computes embeddings live inside
`MemoryBuffer.add_buffer` -- normalise to [-1,1], resize_with_pad to 224, SigLIP,
then `pool_tokens_to_size`. If those two pipelines disagree, the policy trains on
one distribution and is scored on another, silently.

So take the pickle's stored image (what the cache was built from), push it through
the eval-time path verbatim, and compare against the cached tensor for the same
(episode, timestep).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/datadrive1/dzj/RoboMME/preprocessed")
DATA = ROOT / "data/extracted"
FEAT = ROOT / "features/extracted"
BLOCK = {"PickHighlight": 800, "MoveCube": 900, "VideoPlaceOrder": 1300}


def field(d, k):
    v = d[k]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    v = np.ravel(v)
    return v[0].decode(errors="replace") if isinstance(v[0], bytes) else v[0]


def pkl_start(target: int, lo: int, hi: int) -> int:
    while lo < hi:
        mid = (lo + hi) // 2
        if int(field(pickle.load(open(DATA / f"{mid}.pkl", "rb")), "epis_idx")) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickHighlight")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--per-episode", type=int, default=3)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.shared import image_tools
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size

    # the eval path's encoder is the policy's own SigLIP tower; load it the same
    # way serve_policy does, via the checkpoint the runs used
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.training import config as _config

    CKPT = Path("/datadrive1/dzj/RoboMME/ckpt/perceptual-framesamp-modul/home/daiyp/"
                "MME-VLA-Suite/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999")
    train_cfg = _config.get_config("mme_vla_suite")
    policy = _policy_config.create_trained_policy(train_cfg, CKPT)
    vision_enc = policy.mem_buffer.vision_enc
    print(f"loaded encoder from {CKPT.name}", flush=True)

    base = BLOCK[a.task]
    lo = pkl_start(base, 0, 476857)
    hi = pkl_start(base + 100, lo, 476857)
    print(f"{a.task}: episodes {base}-{base+99}, pkl {lo}-{hi}\n", flush=True)

    rows = []
    for e in range(a.episodes):
        ep = base + e
        s = pkl_start(ep, lo, hi)
        n_pkl = pkl_start(ep + 1, s, hi) - s
        # Pickles cover only the execution phase -- the demo is memory, never a
        # training target -- so a pickle's `step_idx` is a global timestep that
        # starts at exec_start, not at 0. Features do cover the whole episode.
        exec_start = int(field(pickle.load(open(DATA / f"{s}.pkl", "rb")), "step_idx"))
        for t in np.linspace(exec_start, exec_start + n_pkl - 1, a.per_episode).astype(int):
            off = s + (int(t) - exec_start)
            f = FEAT / f"episode_{ep}" / f"token_emb_{int(t)}.npy"
            if not f.exists():
                rows.append((ep, int(t), "no cached npy", None, None))
                continue
            cached = np.load(f, allow_pickle=True).item()
            d = pickle.load(open(DATA / f"{off}.pkl", "rb"))
            assert int(field(d, "epis_idx")) == ep and int(field(d, "step_idx")) == int(t), (
                f"pickle {off} is ep{int(field(d,'epis_idx'))} t={int(field(d,'step_idx'))}, "
                f"expected ep{ep} t={int(t)}")

            img = np.asarray(d["image"])[None, None]          # (t=1, v=1, h, w, 3)
            x = jnp.array(img.astype(np.float32) / 255.0 * 2.0 - 1.0)
            x = x.reshape(-1, *x.shape[2:])
            x = image_tools.resize_with_pad(x, 224, 224)
            x = x.reshape(1, 1, 224, 224, 3)
            live = vision_enc(x)                              # (1, 1, 64, 2048)
            live_4x4 = np.asarray(jax.device_get(pool_tokens_to_size(live, 16))[0])

            ref = np.asarray(cached["image_emb_4x4"], dtype=np.float32)
            got = live_4x4.astype(np.float32)
            if ref.shape != got.shape:
                rows.append((ep, int(t), f"shape {ref.shape} vs {got.shape}", None, None))
                continue
            absd = float(np.abs(ref - got).max())
            rel = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9))
            cos = float((ref.ravel() @ got.ravel()) /
                        (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-9))
            rows.append((ep, int(t), "ok", absd, cos))
            print(f"  ep{ep} t={int(t):<5} max|diff|={absd:.5f}  rel={rel:.2e}  cos={cos:.6f}",
                  flush=True)

    good = [r for r in rows if r[2] == "ok"]
    print(f"\ncompared {len(good)} frames")
    if good:
        print(f"  worst max|diff| : {max(r[3] for r in good):.5f}")
        print(f"  worst cosine    : {min(r[4] for r in good):.6f}")
        print("\ncosine ~1.0 and a small max|diff| mean the cached features are the same "
              "tensors the eval path would produce, so training on them is consistent with "
              "how the policy is scored. A cosine noticeably below 1 means a silent "
              "train/eval mismatch.")
    for r in rows:
        if r[2] != "ok":
            print(f"  UNCHECKED ep{r[0]} t={r[1]}: {r[2]}")


if __name__ == "__main__":
    main()
