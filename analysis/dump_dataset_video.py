"""Render an episode from the RoboMME h5 dataset as an mp4.

The release ships per-timestep arrays, not video, so there is nothing to just
open. This stitches obs/front_rgb back into a clip and burns in the subgoal and
the video-demo flag, which is the part worth seeing: the conditioning video the
policy is supposed to remember plays out in the same stream as the execution.
"""

from __future__ import annotations

import argparse
import re

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw


def episode_frames(grp):
    steps = sorted(
        (k for k in grp if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )
    for k in steps:
        ts = grp[k]
        rgb = np.asarray(ts["obs"]["front_rgb"])
        info = ts["info"]
        sub = info["simple_subgoal"][()]
        if isinstance(sub, bytes):
            sub = sub.decode(errors="replace")
        yield rgb, str(sub), bool(np.asarray(info["is_video_demo"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="StopCube")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="/tmp/robomme_episode.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    path = f"/datadrive1/dzj/RoboMME/dataset/record_dataset_{a.task}.h5"
    with h5py.File(path, "r") as f:
        grp = f[f"episode_{a.episode}"]
        goal = grp["setup"]["task_goal"][()]
        goal = [g.decode(errors="replace") if isinstance(g, bytes) else str(g) for g in np.atleast_1d(goal)]
        print(f"task={a.task} episode={a.episode}")
        print(f"  goal: {goal[0][:110]}")

        out, n_demo = [], 0
        for i, (rgb, sub, is_demo) in enumerate(episode_frames(grp)):
            if i % a.stride:
                continue
            n_demo += is_demo
            img = Image.fromarray(rgb).resize((512, 512), Image.NEAREST)
            d = ImageDraw.Draw(img)
            tag = "DEMO VIDEO" if is_demo else "EXECUTION"
            d.rectangle([0, 0, 512, 34], fill=(0, 0, 0))
            d.text((6, 3), f"t={i}  {tag}", fill=(255, 220, 0))
            d.text((6, 19), re.sub(r"\s+", " ", sub)[:74], fill=(180, 255, 180))
            out.append(np.asarray(img))

    imageio.mimwrite(a.out, out, fps=a.fps, macro_block_size=1)
    print(f"  {len(out)} frames ({n_demo} demo-video frames) -> {a.out}")


if __name__ == "__main__":
    main()
