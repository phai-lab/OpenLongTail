"""Shard-level MapAnything pose extraction: load the model ONCE, write pose.pt for
every clip in a TestData shard (chunk_900/<stem>/clip_000000/frames/*.png).

Run with the mapanything venv:
  mapanything_venv/bin/python tools/mapanything_shard_poses.py \
      --test-data-root <shard TD dir>

Writes <clip>/pose.pt = {"T_anchor_front": (11,4,4)} using the pose convention
T_anchor_front[i] = inv(P_c2w[0]) @ P_c2w[4i], OpenCV +Z-forward (MapAnything native).
Resume-safe: skips clips that already have pose.pt.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from natsort import natsorted

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pose_from_mapanything import build_T_anchor_front  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data-root", type=Path, required=True)
    ap.add_argument("--model", type=str, default="facebook/map-anything-apache")
    ap.add_argument("--resolution-set", type=int, default=518)
    ap.add_argument("--amp-dtype", type=str, default="bf16")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    clip_dirs = natsorted(
        {str(Path(p).parent) for p in glob.glob(str(args.test_data_root / "**" / "frames"), recursive=True)}
    )
    clip_dirs = [Path(c) for c in clip_dirs]
    todo = [c for c in clip_dirs if args.overwrite or not (c / "pose.pt").exists()]
    print(f"[mapanything_shard] {len(clip_dirs)} clips, {len(todo)} need poses", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mapanything_shard] loading {args.model} on {device}", flush=True)
    model = MapAnything.from_pretrained(args.model).to(device).eval()

    ok = 0
    for i, clip in enumerate(todo, 1):
        try:
            paths = natsorted(glob.glob(str(clip / "frames" / "*.png")))
            if len(paths) < 41:
                print(f"  [{i}/{len(todo)}] SKIP {clip.name}: only {len(paths)} frames", flush=True)
                continue
            views = load_images(paths, resize_mode="fixed_mapping",
                                resolution_set=args.resolution_set, verbose=False)
            with torch.no_grad():
                preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                    use_amp=(args.amp_dtype != "fp32"), amp_dtype=args.amp_dtype,
                                    apply_mask=True, mask_edges=True, apply_confidence_mask=False)
            poses = torch.stack([p["camera_poses"][0].float().cpu() for p in preds], dim=0)  # (N,4,4) c2w OpenCV
            T = build_T_anchor_front(poses, is_c2w=True, is_opengl=False)  # (11,4,4)
            torch.save({"T_anchor_front": T, "source": "mapanything",
                        "forward_m": float(T[-1, 2, 3])}, clip / "pose.pt")
            ok += 1
            if i % 5 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {clip.name}  fwd={float(T[-1,2,3]):.1f}m", flush=True)
        except Exception as e:  # never let one clip kill the shard
            print(f"  [{i}/{len(todo)}] ERROR {clip.name}: {e!r}", flush=True)
    print(f"[mapanything_shard] wrote {ok}/{len(todo)} poses", flush=True)


if __name__ == "__main__":
    main()
