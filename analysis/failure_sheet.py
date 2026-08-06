"""Build one contact sheet per task: several failures over a couple of successes.

Each row is one episode sampled evenly across its length, so a whole task's
failure modes can be compared side by side instead of scrubbing 50 clips. The
last rows are successes, which is what makes a failure legible — the same task
done right is the reference for what went wrong above it.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]


# The recorder stacks a text panel (goal / action / state) above the imagery, so
# the top ~45% of every frame is unreadable at contact-sheet scale and worth
# dropping. What remains is 256x512: front and wrist views side by side.
IMG_TOP = 208


def row_frames(path: Path, n_cols: int, cell: int) -> tuple[list[np.ndarray], int]:
    v = iio.mimread(path, memtest=False)
    idx = np.linspace(0, len(v) - 1, n_cols).astype(int)
    out = []
    for i in idx:
        fr = np.asarray(v[i], dtype=np.uint8)[IMG_TOP:]
        img = Image.fromarray(fr).resize((cell * 2, cell))
        d = ImageDraw.Draw(img)
        d.text((3, 2), f"{i}", fill=(255, 220, 0))
        out.append(np.asarray(img))
    return out, len(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/datadrive1/dzj/RoboMME/eval_out/ultra3_seed7")
    ap.add_argument("--out", default="/datadrive1/dzj/RoboMME/failure_sheets")
    ap.add_argument("--fails", type=int, default=8)
    ap.add_argument("--succ", type=int, default=2)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--cell", type=int, default=150)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()

    run, outdir = Path(a.run), Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vids = list(run.glob("shard*/**/videos/*.mp4"))
    rng = random.Random(0)

    for task in (a.tasks.split(",") if a.tasks else TASKS):
        mine = [p for p in vids if p.name.startswith(f"{task}_ep")]
        fails = sorted(p for p in mine if "_fail_" in p.name or "_timeout_" in p.name)
        succs = sorted(p for p in mine if "_success_" in p.name)
        rng.shuffle(fails)
        rng.shuffle(succs)
        rows = fails[: a.fails] + succs[: a.succ]
        if not rows:
            print(f"  {task}: no videos")
            continue

        pad, label_w = 18, 210
        W = label_w + a.cols * a.cell * 2
        H = len(rows) * (a.cell + pad) + pad
        sheet = Image.new("RGB", (W, H), "black")
        draw = ImageDraw.Draw(sheet)
        goal = re.sub(r"_(easy|medium|hard)$", "", rows[0].stem.split("_", 3)[-1])
        draw.text((6, 4), f"{task}   |   {goal[:120]}", fill=(255, 255, 255))

        for r, p in enumerate(rows):
            frames, n = row_frames(p, a.cols, a.cell)
            y = pad + r * (a.cell + pad)
            ok = "_success_" in p.name
            ep = p.name.split("_ep")[1].split("_")[0]
            diff = p.stem.rsplit("_", 1)[-1]
            draw.text((6, y + 2), f"{'OK  ' if ok else 'FAIL'} ep{ep} [{diff}] {n}f",
                      fill=(120, 255, 120) if ok else (255, 130, 130))
            for c, fr in enumerate(frames):
                sheet.paste(Image.fromarray(fr), (label_w + c * a.cell * 2, y))

        dst = outdir / f"{task}.png"
        sheet.save(dst)
        print(f"  {task}: {len(fails[:a.fails])} fail + {len(succs[:a.succ])} ok -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()
