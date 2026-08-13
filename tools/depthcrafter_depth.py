#!/usr/bin/env python
"""
depthcrafter_depth.py  —  DepthCrafter front-depth in the OpenLongTail contract.

Runs Tencent DepthCrafter (SVD-img2vid-xt base + DepthCrafter UNet) on a single
front dashcam mp4 and writes a `front_depth.pt` that is byte-compatible with what
`openlongtail/scripts/build_warp.py` and the raw `inference.py` warp provider
expect:

    { "depth_sequence": FloatTensor (T,1,H,W)   normalized-inverse depth in [0,1],
                                                 LARGER = NEARER (disparity-like),
      "source_indices": LongTensor  (T,) }       = arange(T)

This matches the cached demo clips, whose sidecars record
`depth_source = 'depthcrafter_fullseq_h_384_w_672'` — i.e. the SAME estimator and
the SAME (H=384, W=672) full-sequence convention. We deliberately do NOT use
DepthCrafter's own run.py (it picks frame count / resolution heuristically); we
sample EXACTLY T frames and run at the target H×W so the output drops straight
into the warp pipeline.

Polarity note: DepthCrafter emits SVD-style depth normalized per-sequence to
[0,1]. Empirically its large values correspond to NEAR (disparity/inverse-depth),
which is exactly the "normalized inverse depth" the loader's `1/depth` branch
wants. `--assert-polarity` verifies this at runtime (bottom-of-frame road should
read nearer than the top-of-frame sky) and refuses to write a wrong-polarity map.

Runs in the isolated depthcrafter_venv, not wm_venv.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _round64(x: int) -> int:
    return max(64, round(x / 64) * 64)


def _load_frames(video_path: str, num_frames: int, h: int, w: int) -> np.ndarray:
    """Sample exactly `num_frames` evenly across the clip, resized to (h,w). -> (T,h,w,3) float32 [0,1].

    NOTE: (h,w) here is the INFERENCE resolution and MUST be /64-aligned — SVD's
    UNet down/up-samples 4× and non-/64 sizes produce off-by-one skip-connection
    mismatches (e.g. width 672 -> "expected 22 got 21"). The stored depth is
    resized to the contract size (384x672) afterwards, so inference res only needs
    to preserve aspect ratio, not equal the storage size."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
    n = len(vr)
    if n >= num_frames:
        idx = np.linspace(0, n - 1, num_frames).round().astype(int).tolist()
    else:  # pad by repeating the last frame if the clip is short
        idx = list(range(n)) + [n - 1] * (num_frames - n)
    frames = vr.get_batch(idx).asnumpy().astype("float32") / 255.0
    return frames  # (T,h,w,3)


@torch.no_grad()
def run_depthcrafter(frames: np.ndarray, unet_path: str, svd_path: str,
                     num_inference_steps: int, guidance: float, window: int,
                     overlap: int, seed: int, device: str) -> np.ndarray:
    """frames (T,H,W,3) [0,1] -> depth (T,H,W) float32 in [0,1], per-sequence normalized."""
    sys.path.insert(0, "/mnt/localssd/DepthCrafter")
    from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
    from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        unet_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    pipe = DepthCrafterPipeline.from_pretrained(
        svd_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"[depthcrafter] xformers unavailable ({e}); continuing", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    res = pipe(
        frames,
        height=frames.shape[1],
        width=frames.shape[2],
        output_type="np",
        guidance_scale=guidance,
        num_inference_steps=num_inference_steps,
        window_size=window,
        overlap=overlap,
        generator=gen,
    ).frames[0]                         # (T,H,W,3), 3 identical channels
    res = res.sum(-1) / res.shape[-1]   # -> (T,H,W) grayscale
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # per-seq [0,1]
    return res.astype("float32")


def _check_polarity(depth: np.ndarray) -> tuple[bool, float, float]:
    """Road (bottom third) should be NEARER than sky (top third) for inverse depth.
    Returns (near_is_large, bottom_mean, top_mean)."""
    T, H, W = depth.shape
    top = depth[:, : H // 3, :].mean()
    bottom = depth[:, 2 * H // 3 :, :].mean()
    return (bottom > top), float(bottom), float(top)


def main() -> int:
    ap = argparse.ArgumentParser(description="DepthCrafter -> OpenLongTail front_depth.pt")
    ap.add_argument("--video", required=True, help="front dashcam mp4")
    ap.add_argument("--out", required=True, help="writes front_depth.pt here")
    ap.add_argument("--frames", type=int, default=41)
    # storage (contract) size — what build_warp / the loader read: h_384_w_672
    ap.add_argument("--height", type=int, default=384, help="STORED depth height (contract)")
    ap.add_argument("--width", type=int, default=672, help="STORED depth width (contract)")
    # inference size — MUST be /64-aligned for SVD UNet. Default = /64-round of the
    # rig's native 480x832 (-> 512x832), preserving aspect ratio; resized to store size after.
    ap.add_argument("--infer-height", type=int, default=512, help="DepthCrafter inference height (/64)")
    ap.add_argument("--infer-width", type=int, default=832, help="DepthCrafter inference width (/64)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=1.2)
    ap.add_argument("--window", type=int, default=110)
    ap.add_argument("--overlap", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unet-path", default="tencent/DepthCrafter")
    ap.add_argument("--svd-path", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--assert-polarity", action="store_true",
                    help="fail if road doesn't read nearer than sky")
    ap.add_argument("--save-vis", default=None, help="optional: write an inferno depth mp4 here")
    a = ap.parse_args()

    ih, iw = _round64(a.infer_height), _round64(a.infer_width)
    if (ih, iw) != (a.infer_height, a.infer_width):
        print(f"[depthcrafter] infer res {a.infer_height}x{a.infer_width} -> /64-aligned {ih}x{iw}", flush=True)
    print(f"[depthcrafter] {a.video}: infer @ {ih}x{iw}, store @ {a.height}x{a.width}, {a.frames}f", flush=True)
    frames = _load_frames(a.video, a.frames, ih, iw)
    print(f"[depthcrafter] frames {frames.shape}; running pipeline "
          f"(steps={a.steps} window={a.window} overlap={a.overlap})", flush=True)
    depth = run_depthcrafter(frames, a.unet_path, a.svd_path, a.steps, a.guidance,
                             a.window, a.overlap, a.seed, a.device)     # (T,ih,iw) [0,1]

    # resize inference-res depth -> stored contract size (384x672), matching the
    # depthcrafter_cache/fullseq_h_384_w_672 sidecar convention.
    if depth.shape[1:] != (a.height, a.width):
        import torch.nn.functional as F
        d = torch.from_numpy(depth).unsqueeze(1)                       # (T,1,ih,iw)
        d = F.interpolate(d, size=(a.height, a.width), mode="bilinear", align_corners=False)
        depth = d.squeeze(1).numpy()
        print(f"[depthcrafter] resized depth -> {depth.shape}", flush=True)

    near_large, bot, top = _check_polarity(depth)
    print(f"[depthcrafter] polarity: bottom(road)={bot:.3f} top(sky)={top:.3f} "
          f"-> near_is_large={near_large}", flush=True)
    if not near_large:
        msg = ("depth polarity looks inverted (sky reads nearer than road). "
               "DepthCrafter should be disparity-like (near=large).")
        if a.assert_polarity:
            print(f"[depthcrafter] FATAL: {msg}", file=sys.stderr)
            return 2
        print(f"[depthcrafter] WARNING: {msg} — writing as-is (no flip).", flush=True)

    seq = torch.from_numpy(depth).float().unsqueeze(1)  # (T,1,H,W)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"depth_sequence": seq,
                "source_indices": torch.arange(seq.shape[0]),
                "depth_source": f"depthcrafter_fullseq_h_{a.height}_w_{a.width}",
                "polarity": "normalized_inverse_near_large"}, out)
    print(f"[depthcrafter] wrote {out}  depth_sequence={tuple(seq.shape)} "
          f"range=[{seq.min():.3f},{seq.max():.3f}]", flush=True)

    if a.save_vis:
        try:
            sys.path.insert(0, "/mnt/localssd/DepthCrafter")
            from depthcrafter.utils import save_video, vis_sequence_depth
            vis = vis_sequence_depth(depth)
            save_video(vis, a.save_vis, fps=10)
            print(f"[depthcrafter] vis -> {a.save_vis}", flush=True)
        except Exception as e:
            print(f"[depthcrafter] vis skipped: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
