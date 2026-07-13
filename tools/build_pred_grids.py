"""Assemble labeled 1x6 preview grids [front | 5 predicted views] from the
per-view inference outputs.

    python tools/build_pred_grids.py --infer-root <infer_dir>
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

PANELS = [
    ("front_input", "FRONT (input)"),
    ("pred_cross_left", "cross_left"),
    ("pred_cross_right", "cross_right"),
    ("pred_rear_left", "rear_left"),
    ("pred_rear_right", "rear_right"),
    ("pred_rear_tele", "rear_tele"),
]


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, 8 * len(text) + 8, 18), fill=(0, 0, 0))
    dr.text((3, 3), text, fill=(255, 255, 255))
    return np.array(img)


def build_clip_grid(clip_dir: Path, fps: int) -> Path | None:
    vids = {}
    for name, _ in PANELS:
        p = clip_dir / f"{name}.mp4"
        if not p.exists():
            return None
        vids[name] = iio.imread(p, plugin="pyav")
    T = min(v.shape[0] for v in vids.values())
    frames = []
    for t in range(T):
        row = [label(vids[name][t].copy(), lab) for name, lab in PANELS]
        frames.append(np.concatenate(row, axis=1))
    out = clip_dir / "grid.mp4"
    iio.imwrite(out, np.stack(frames), plugin="pyav", codec="libx264", fps=fps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    clip_dirs = sorted({Path(p).parent for p in
                        glob.glob(str(args.infer_root / "**" / "pred_cross_left.mp4"), recursive=True)})
    print(f"building grids for {len(clip_dirs)} clips", flush=True)
    n = 0
    for cd in clip_dirs:
        out = build_clip_grid(cd, args.fps)
        if out is not None:
            n += 1
            print(f"  {out.relative_to(args.infer_root)}", flush=True)
    print(f"DONE: {n} grids written", flush=True)


if __name__ == "__main__":
    main()
