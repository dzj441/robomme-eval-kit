"""Dump exactly the frames FrameSamp keeps, from our own rollouts, for inspection.

Two panels per episode. The dense strip is the episode itself at a fine stride,
so the moment of interest can be located by eye. The sampled panel is the 32
frames the memory actually holds at that point, labelled with their true step
indices. Put side by side, they answer whether the evidence a task depends on
survives into the memory -- without trusting any of my own event detectors.

Faithfulness notes. The rule is re-evaluated only at action-chunk boundaries
(`obs_horizon` = 16), not every step, so the sampling step is rounded down to a
multiple of 16. The memory holds the front view only (`num_views: 1`), so the
wrist half of the recorder's composite is cropped away.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

IMG_TOP = 208     # recorder stacks a text panel above the 256x512 imagery
FRONT_W = 256     # front | wrist, side by side; memory sees only the front
CHUNK = 16        # obs_horizon: the rule is evaluated at these boundaries
MAX_SIZE = 32     # budget 512 / (token_per_image 16 * num_views 1)


def sampled_indices(step_idx: int) -> np.ndarray:
    if step_idx < MAX_SIZE:
        return np.arange(step_idx + 1)
    return np.linspace(0, step_idx, MAX_SIZE, dtype=np.int64)


def front(frame: np.ndarray) -> np.ndarray:
    return frame[IMG_TOP:, :FRONT_W]


def strip(frames, idxs, cell, cols, title, mark=None):
    rows = int(np.ceil(len(idxs) / cols))
    W, H = cols * cell, rows * cell + 22
    sheet = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill=(255, 255, 255))
    for k, i in enumerate(idxs):
        img = Image.fromarray(front(np.asarray(frames[int(i)], dtype=np.uint8))).resize((cell, cell))
        dd = ImageDraw.Draw(img)
        hit = mark is not None and mark[0] <= i <= mark[1]
        dd.text((3, 2), f"{int(i)}", fill=(255, 60, 60) if hit else (255, 220, 0))
        if hit:
            dd.rectangle([0, 0, cell - 1, cell - 1], outline=(255, 0, 0), width=3)
        sheet.paste(img, ((k % cols) * cell, 22 + (k // cols) * cell))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/datadrive1/dzj/RoboMME/eval_out/ultra3_seed7")
    ap.add_argument("--task", default="PickHighlight")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--outcome", default="fail")
    ap.add_argument("--at", default="end", help="'end' or an integer step")
    ap.add_argument("--dense-stride", type=int, default=4)
    ap.add_argument("--cell", type=int, default=120)
    ap.add_argument("--out", default="/datadrive1/dzj/RoboMME/sampled_frames")
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(a.run).glob(f"shard*/**/videos/{a.task}_ep*_{a.outcome}_*.mp4"))
    if not vids:
        print(f"no {a.outcome} videos for {a.task}")
        return

    for p in vids[: a.n]:
        v = iio.mimread(p, memtest=False)
        T = len(v)
        step = (T - 1) if a.at == "end" else int(a.at)
        step = (step // CHUNK) * CHUNK          # the rule only fires on chunk boundaries
        keep = sampled_indices(step)
        ep = p.name.split("_ep")[1].split("_")[0]

        dense_idx = np.arange(0, T, a.dense_stride)
        s1 = strip(v, dense_idx, a.cell, 16,
                   f"{a.task} ep{ep} [{a.outcome}] — FULL episode, every {a.dense_stride}th of {T} frames")
        s2 = strip(v, keep, a.cell, 16,
                   f"same episode — the {len(keep)} frames the memory holds at step {step} "
                   f"(spacing {step/(MAX_SIZE-1):.1f})")

        sheet = Image.new("RGB", (max(s1.width, s2.width), s1.height + s2.height + 14), "black")
        sheet.paste(s1, (0, 0))
        sheet.paste(s2, (0, s1.height + 14))
        dst = outdir / f"{a.task}_ep{ep}_{a.outcome}.png"
        sheet.save(dst)
        print(f"{a.task} ep{ep}: T={T} step={step} kept={list(map(int, keep))[:6]}... -> {dst.name}",
              flush=True)


if __name__ == "__main__":
    main()
