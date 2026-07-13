"""Filter out night/dark clips (front brightness < thresh), then concatenate the
remaining daytime grids into batch videos (default 15 grids per video).

  python tools/filter_and_concat_grids.py --infer-root <infer_root> \
      --out-dir <out_dir> --bright-thresh 40 --group-size 15
"""
from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np


def front_brightness(clip_dir: str) -> tuple[str, float]:
    """Mean brightness over a few sampled frames of the front input (with retry)."""
    for _ in range(4):
        try:
            v = iio.imread(os.path.join(clip_dir, "front_input.mp4"), plugin="pyav")
            idx = np.linspace(0, len(v) - 1, min(5, len(v))).astype(int)
            return clip_dir, float(v[idx].astype(np.float32).mean())
        except Exception:
            continue
    return clip_dir, -1.0


def concat_batch(args) -> str:
    """Concatenate a list of grid.mp4 into one video; return output path."""
    out_path, grids, fps = args
    with iio2.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for g in grids:
            for fr in iio.imread(g, plugin="pyav"):
                w.append_data(fr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bright-thresh", type=float, default=40.0)
    ap.add_argument("--group-size", type=int, default=15)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted({os.path.dirname(p) for p in
                        glob.glob(str(args.infer_root / "**" / "grid.mp4"), recursive=True)})
    print(f"scanning brightness of {len(clip_dirs)} clips ...", flush=True)
    # low worker count avoids pyav parallel-decode failures; front_brightness also retries
    with Pool(4) as pool:
        bright = pool.map(front_brightness, clip_dirs)
    fails = [d for d, b in bright if b < 0]
    if fails:  # last-resort sequential retry so no clip is silently dropped
        print(f"  retrying {len(fails)} failed reads single-threaded ...", flush=True)
        fixed = dict(front_brightness(d) for d in fails)
        bright = [(d, fixed.get(d, b) if b < 0 else b) for d, b in bright]

    def stem(d):
        return d.split("/chunk_900/")[1].split("/")[0]

    night = [(stem(d), b) for d, b in bright if 0 <= b < args.bright_thresh]
    day = sorted([d for d, b in bright if b >= args.bright_thresh])
    (args.out_dir / "night_clips.txt").write_text(
        "\n".join(f"{s}\t{b:.1f}" for s, b in sorted(night, key=lambda x: x[1])) + "\n")
    (args.out_dir / "daytime_clips.txt").write_text("\n".join(stem(d) for d in day) + "\n")
    print(f"night/dark (<{args.bright_thresh}): {len(night)}  |  daytime kept: {len(day)}", flush=True)

    # group daytime grids into batches, concat each
    day_grids = [os.path.join(d, "grid.mp4") for d in day]
    batches = [day_grids[i:i + args.group_size] for i in range(0, len(day_grids), args.group_size)]
    jobs = [(str(args.out_dir / f"daytime_grids_{i:03d}.mp4"), b, args.fps) for i, b in enumerate(batches)]
    print(f"building {len(jobs)} videos ({args.group_size} grids each) ...", flush=True)
    with Pool(min(args.workers, 8)) as pool:
        outs = pool.map(concat_batch, jobs)
    print(f"DONE: {len(outs)} daytime videos -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
