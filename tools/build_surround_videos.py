"""Stitch per-view outputs into a 2x3 surround-view video (no labels).

Layout (each cell 480x832):
    [ cross_left  | front(input) | cross_right ]     <- Front-left, Front, Front-right
    [ rear_right  | rear_tele    | rear_left   ]     <- Rear-right,  Rear,  Rear-left

  python tools/build_surround_videos.py --infer-root <dir> --fps 12
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np

# (filename, grid position) — row-major 2x3
LAYOUT = [
    ("pred_cross_left.mp4", (0, 0)),
    ("front_input.mp4",     (0, 1)),
    ("pred_cross_right.mp4", (0, 2)),
    ("pred_rear_right.mp4", (1, 0)),
    ("pred_rear_tele.mp4",  (1, 1)),
    ("pred_rear_left.mp4",  (1, 2)),
]


def build_clip(clip_dir: Path, fps: int) -> Path | None:
    vids = {}
    for name, _ in LAYOUT:
        p = clip_dir / name
        if not p.exists():
            return None
        vids[name] = iio.imread(p, plugin="pyav")  # (T,H,W,3)
    T = min(v.shape[0] for v in vids.values())
    H, W = vids["front_input.mp4"].shape[1:3]
    frames = []
    for t in range(T):
        rows = []
        for r in range(2):
            cells = []
            for c in range(3):
                name = next(n for n, pos in LAYOUT if pos == (r, c))
                fr = vids[name][t]
                if fr.shape[:2] != (H, W):
                    from PIL import Image
                    fr = np.array(Image.fromarray(fr).resize((W, H)))
                cells.append(fr)
            rows.append(np.concatenate(cells, axis=1))   # (H, 3W)
        frames.append(np.concatenate(rows, axis=0))       # (2H, 3W)
    out = clip_dir / "surround.mp4"
    iio.imwrite(out, np.stack(frames), plugin="pyav", codec="libx264", fps=fps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    clip_dirs = sorted({Path(p).parent for p in
                        glob.glob(str(args.infer_root / "**" / "front_input.mp4"), recursive=True)})
    n = 0
    for cd in clip_dirs:
        out = build_clip(cd, args.fps)
        if out is not None:
            n += 1
            print(f"  {out}", flush=True)
    print(f"DONE: {n} surround videos", flush=True)


if __name__ == "__main__":
    main()
