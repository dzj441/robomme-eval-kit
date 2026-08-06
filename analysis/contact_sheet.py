"""Lay a rollout video out as a labelled contact sheet so a whole episode can be
eyeballed in one image instead of watched for four minutes.

usage: contact_sheet.py <video.mp4> <out.png> [n_cols] [n_rows]
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 6

frames = iio.mimread(src, memtest=False)
n = cols * rows
idx = np.linspace(0, len(frames) - 1, n).astype(int)

h, w = frames[0].shape[:2]
pad = 14  # room for the frame-number caption
sheet = Image.new("RGB", (cols * w, rows * (h + pad)), "black")
draw = ImageDraw.Draw(sheet)

for k, i in enumerate(idx):
    r, c = divmod(k, cols)
    sheet.paste(Image.fromarray(np.asarray(frames[i], dtype=np.uint8)), (c * w, r * (h + pad) + pad))
    draw.text((c * w + 3, r * (h + pad) + 2), f"t={i}", fill="yellow")

sheet.save(dst)
print(f"{src.name}: {len(frames)} frames -> {dst} ({cols}x{rows} grid)")
