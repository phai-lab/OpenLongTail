#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mapanything_poses.py
========================

Estimate per-frame camera poses (+ intrinsics, optional depth) for an ORDERED
sequence of images (e.g. a dashcam clip) using Meta's MapAnything
(feed-forward metric 3D reconstruction, arXiv:2509.13414).

--------------------------------------------------------------------------------
OUTPUT CONVENTION  (validated empirically on a forward-driving Nexar clip)
--------------------------------------------------------------------------------
MapAnything's `camera_poses` output is documented in the repo README as:

    "OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame"

We SAVE exactly that, unchanged:

  poses_c2w[i]  = 4x4 SE(3) matrix that maps a point expressed in CAMERA i
                  coordinates to WORLD coordinates:   X_world = poses_c2w[i] @ X_cam
                  i.e. this is the CAMERA-TO-WORLD (cam2world) extrinsic.
                  The camera axes are OpenCV:  +X = right, +Y = down, +Z = forward
                  (optical axis, pointing INTO the scene).
                  Translation block poses_c2w[i][:3,3] is the camera CENTER in
                  world coordinates, in METRIC units (meters) -- MapAnything
                  predicts metric scale.

  To get world-to-cam (a.k.a. the "extrinsic" / view matrix) invert it:
                  T_w2c = inv(poses_c2w)

  intrinsics[i] = 3x3 pinhole K for the *internally processed* image resolution
                  (see proc_hw below), pixel units:
                      [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

  depth[i]      = per-pixel Z-DEPTH in the camera frame (distance along +Z /
                  optical axis, NOT distance along the ray), meters, at proc_hw.

  proc_hw       = (H, W) that MapAnything actually ran at. NOTE: MapAnything
                  internally rescales/crops every image to a fixed-aspect grid
                  (default resolution_set=518 -> longest side ~518, divisible by
                  the patch size). Poses are resolution-independent, but the
                  intrinsics and depth are given at proc_hw. To use intrinsics at
                  a different resolution (Wt, Ht), rescale:
                      fx,cx *= Wt / W ;  fy,cy *= Ht / H

Downstream helper (the reason this script exists):
  For frame0 = identity, +Z = forward (OpenCV), the anchor-relative pose of
  frame j w.r.t. frame 0 is:
      T_rel[j] = inv(poses_c2w[0]) @ poses_c2w[j]
  which is already in OpenCV cam2world with frame0 at the origin. No axis flip
  is needed -- MapAnything is natively OpenCV (+Z forward). On a forward-driving
  clip the dominant component of T_rel[N][:3,3] is +Z with a positive sign of a
  few meters.

--------------------------------------------------------------------------------
SAVED .pt  (a python dict, load with torch.load(path))
--------------------------------------------------------------------------------
  {
    "poses_c2w":   float32 (N,4,4)   OpenCV cam2world, metric meters
    "intrinsics":  float32 (N,3,3)   pinhole K at proc_hw
    "depth":       float32 (N,H,W)   camera-frame Z-depth, meters  (or None if --no_depth)
    "proc_hw":     int tuple (H, W)  resolution poses/intrinsics/depth correspond to
    "image_paths": list[str]         input image paths, in the SAME ORDER as N
    "convention":  str               human-readable convention string
    "model_name":  str
  }

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  # a directory of ordered frames (natural-sorted by filename):
  python run_mapanything_poses.py --images /path/to/frames_dir --output out.pt

  # or an explicit ordered list of files:
  python run_mapanything_poses.py --images f0.png f1.png f2.png --output out.pt

Environment (set these BEFORE running so nothing touches $HOME):
  HF_HOME=/path/to/.hf_cache
  HF_TOKEN=<your-hf-token>               (for first-time weight download)
  TORCH_HOME=/path/to/.torch_cache       (DINOv2 ViT-g/14 backbone cache)
"""

import argparse
import os

import torch

# Better fragmentation behaviour for many-view inference on big GPUs.
if torch.cuda.is_available():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from natsort import natsorted

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

_SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")


def _resolve_image_list(images_arg):
    """Return an ORDERED list of image file paths.

    - If a single existing directory is given: gather supported images inside it
      and natural-sort by filename (so frame_0002 < frame_0010).
    - Otherwise: treat the args as an explicit, already-ordered list of files
      and keep the given order.
    """
    if len(images_arg) == 1 and os.path.isdir(images_arg[0]):
        d = images_arg[0]
        files = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(_SUPPORTED_EXT)
        ]
        files = natsorted(files)
        if not files:
            raise ValueError(f"No supported images found in directory: {d}")
        return files
    # explicit list -> preserve caller order
    for p in images_arg:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")
    return list(images_arg)


def main():
    ap = argparse.ArgumentParser(
        description="Estimate per-frame OpenCV cam2world poses (+intrinsics/depth) "
        "for an ordered image sequence using MapAnything."
    )
    ap.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="A single directory of ordered frames, OR an explicit ordered list "
        "of image files.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output .pt path (a torch dict, see module docstring).",
    )
    ap.add_argument(
        "--model",
        default="facebook/map-anything-apache",
        help="HF model id. 'facebook/map-anything-apache' (Apache-2.0) or "
        "'facebook/map-anything' (CC-BY-NC-4.0). Default: apache.",
    )
    ap.add_argument(
        "--resolution_set",
        type=int,
        default=518,
        choices=[518, 512, 504],
        help="MapAnything internal fixed-mapping resolution set. Default 518.",
    )
    ap.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size for memory-efficient inference (1 = lowest VRAM).",
    )
    ap.add_argument(
        "--amp_dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Autocast dtype for inference. Default bf16.",
    )
    ap.add_argument(
        "--no_depth",
        action="store_true",
        help="Do not save the per-frame depth (smaller output file).",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_mapanything_poses] device = {device}")

    image_paths = _resolve_image_list(args.images)
    print(f"[run_mapanything_poses] {len(image_paths)} ordered images")
    for i, p in enumerate(image_paths):
        print(f"    [{i:03d}] {p}")

    # ------------------------------------------------------------------ model
    print(f"[run_mapanything_poses] loading model: {args.model}")
    model = MapAnything.from_pretrained(args.model).to(device).eval()

    # ------------------------------------------------------------------ input
    # load_images preserves the order of the input list and returns one view
    # dict per image. It resizes every image to a common fixed-mapping grid.
    views = load_images(
        image_paths,
        resize_mode="fixed_mapping",
        resolution_set=args.resolution_set,
        verbose=True,
    )

    use_amp = args.amp_dtype != "fp32"
    amp_dtype = "bf16" if args.amp_dtype == "bf16" else "fp16"

    # ------------------------------------------------------------- inference
    print("[run_mapanything_poses] running inference ...")
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=args.minibatch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    print(f"[run_mapanything_poses] got {len(preds)} predictions")

    # ------------------------------------------------------------- collect
    poses, intr, depths = [], [], []
    proc_hw = None
    for pred in preds:
        # each pred tensor has a leading batch dim of 1
        pose = pred["camera_poses"][0].float().cpu().numpy()  # (4,4) OpenCV cam2world, metric
        K = pred["intrinsics"][0].float().cpu().numpy()  # (3,3) at proc_hw
        poses.append(pose)
        intr.append(K)
        if not args.no_depth:
            dz = pred["depth_z"][0].squeeze(-1).float().cpu().numpy()  # (H,W) camera-frame Z, meters
            depths.append(dz)
            if proc_hw is None:
                proc_hw = (int(dz.shape[0]), int(dz.shape[1]))
        elif proc_hw is None:
            # infer proc_hw from img_no_norm if depth not requested
            img = pred["img_no_norm"][0]
            proc_hw = (int(img.shape[0]), int(img.shape[1]))

    poses = np.stack(poses, axis=0).astype(np.float32)  # (N,4,4)
    intr = np.stack(intr, axis=0).astype(np.float32)  # (N,3,3)
    depth = (
        np.stack(depths, axis=0).astype(np.float32) if depths else None
    )  # (N,H,W) or None

    convention = (
        "poses_c2w: OpenCV camera-to-world SE(3) (X_world = T @ X_cam); "
        "camera axes +X=right,+Y=down,+Z=forward (optical axis into scene); "
        "translation = camera center in world, METRIC meters. "
        "intrinsics/depth are at proc_hw (H,W). depth = camera-frame Z (meters)."
    )

    out = {
        "poses_c2w": torch.from_numpy(poses),  # (N,4,4) float32
        "intrinsics": torch.from_numpy(intr),  # (N,3,3) float32
        "depth": (torch.from_numpy(depth) if depth is not None else None),
        "proc_hw": proc_hw,
        "image_paths": image_paths,
        "convention": convention,
        "model_name": args.model,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"[run_mapanything_poses] saved -> {args.output}")
    print(f"    poses_c2w  {tuple(poses.shape)}")
    print(f"    intrinsics {tuple(intr.shape)}")
    print(f"    depth      {None if depth is None else tuple(depth.shape)}")
    print(f"    proc_hw    {proc_hw}")

    # ---- quick self-check: relative pose of last frame w.r.t. first ----
    if len(poses) >= 2:
        T_rel = np.linalg.inv(poses[0]) @ poses[-1]  # OpenCV cam2world, frame0 at origin
        t = T_rel[:3, 3]
        ax = int(np.argmax(np.abs(t)))
        axis_name = {0: "X(right)", 1: "Y(down)", 2: "Z(forward)"}[ax]
        print(
            f"[self-check] T_rel(frame0->frame{len(poses)-1}) translation = "
            f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m ; "
            f"dominant axis = {axis_name} sign {'+' if t[ax] >= 0 else '-'} ; "
            f"|t| = {np.linalg.norm(t):.3f} m"
        )


if __name__ == "__main__":
    main()
