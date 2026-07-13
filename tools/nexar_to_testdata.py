"""Convert Nexar dashcam clips into the TestData format (for --treat-as-nv inference).

Nexar (https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) is a
monocular front-facing dashcam dataset — no camera rig, no calibration, no ego
pose. Inference expects, per clip:

    chunk_<c>/<uuid>/<clip_id>/
        front.mp4         41 RGB frames @ 480x832
        front_depth.pt    {"depth_sequence": (41,1,H,W)}  (raw inverse depth in [0,1])
        pose.pt           {"T_anchor_front": (11,4,4)}    (front cam pose / latent frame)
        meta.pt           {"K": (3,3), "src_fps": int, ...}
    manifest_clips.jsonl  one {"chunk","uuid","clip_id"} per line

Because Nexar has no calibration/pose, we:
  * K, E  -> supplied at inference by `--treat-as-nv` from the NV reference rig
            (the placeholder meta["K"] here is ignored under that flag).
  * depth -> monocular Depth-Anything-V2 (per frame), stored as normalized inverse
            depth so the loader's `1/depth` branch recovers pseudo-metric meters.
  * pose  -> synthetic constant-velocity forward motion (+Z), matching the data
            convention (frame 0 = identity, forward translation grows in Z).

Run on a GPU node (depth needs CUDA):
    python tools/nexar_to_testdata.py \
        --nexar-root <nexar_root> \
        --out-root   <out_root> \
        --num-clips 20
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openlongtail.scripts.inference_p3_smoke import _write_mp4

CLIP_FRAMES = 41           # RGB frames per clip (model contract)
OUT_H, OUT_W = 480, 832    # model input resolution
FWD_DISPLACEMENT_M = 10.0  # synthetic forward travel across the clip


def select_frame_indices(n_frames: int) -> list[int]:
    """Pick 41 CONSECUTIVE frames (stride 1) to match the model input contract.

    The model consumes 41 frames at stride 1 from the 30fps source; the 11
    latent-frame poses (T_anchor_front) are sampled at frames [0,4,8,...,40].
    We center the 41-frame window in the clip to skip startup and any
    end-of-clip collision (Nexar positives crash in the last ~1s).
    """
    if n_frames >= CLIP_FRAMES:
        start = max(0, (n_frames - CLIP_FRAMES) // 2)
        return [start + i for i in range(CLIP_FRAMES)]
    idx = list(range(n_frames))
    idx += [idx[-1]] * (CLIP_FRAMES - len(idx))
    return idx


def load_and_resize(video_path: Path) -> tuple[torch.Tensor, float]:
    """Return (41,3,480,832) uint8 tensor + effective fps."""
    import imageio.v3 as iio

    meta = iio.immeta(video_path, plugin="pyav")
    src_fps = float(meta.get("fps", 30.0) or 30.0)
    frames = iio.imread(video_path, plugin="pyav")  # (N,H,W,3) uint8
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB video at {video_path}, got {frames.shape}")
    idx = select_frame_indices(frames.shape[0])
    stride = max(1, idx[1] - idx[0]) if len(idx) > 1 else 1
    sel = frames[idx]                                             # (41,H,W,3)
    x = torch.from_numpy(sel).permute(0, 3, 1, 2).float()        # (41,3,H,W)

    # center-crop to the 832:480 aspect, then resize.
    _, _, h, w = x.shape
    target_ar = OUT_W / OUT_H
    if w / h > target_ar:                        # too wide -> crop width
        new_w = int(round(h * target_ar))
        off = (w - new_w) // 2
        x = x[:, :, :, off:off + new_w]
    else:                                        # too tall -> crop height
        new_h = int(round(w / target_ar))
        off = (h - new_h) // 2
        x = x[:, :, off:off + new_h, :]
    x = F.interpolate(x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8), src_fps / stride


@torch.no_grad()
def estimate_depth(frames_u8: torch.Tensor, model, processor, device) -> torch.Tensor:
    """Depth-Anything-V2 inverse depth, normalized to [0,1], shape (41,1,480,832) f16."""
    imgs = [f.permute(1, 2, 0).numpy() for f in frames_u8]        # list of HWC uint8
    inp = processor(images=imgs, return_tensors="pt").to(device, torch.float16)
    pred = model(**inp).predicted_depth.float()                  # (41,Hd,Wd), higher=closer
    pred = F.interpolate(pred.unsqueeze(1), size=(OUT_H, OUT_W),
                         mode="bilinear", align_corners=False)    # (41,1,480,832)
    pred = pred.clamp(min=0)
    pred = pred / (pred.max() + 1e-6)                             # normalized inverse depth
    return pred.to(torch.float16).cpu()


def synthetic_pose() -> torch.Tensor:
    """Constant-velocity forward motion, (11,4,4). Frame 0 = identity, +Z forward."""
    T = torch.eye(4).unsqueeze(0).repeat(11, 1, 1)
    for i in range(11):
        T[i, 2, 3] = FWD_DISPLACEMENT_M * i / 10.0
    return T


def build_meta(stem: str, src_fps: float, video_path: Path) -> dict:
    return {
        "K": torch.tensor([[450.0, 0.0, OUT_W / 2],
                           [0.0, 450.0, OUT_H / 2],
                           [0.0, 0.0, 1.0]], dtype=torch.float32),   # placeholder (ignored under --treat-as-nv)
        "E_rig_front": torch.eye(4, dtype=torch.float32),
        "src_fps": int(round(src_fps)),
        "stride": 1,
        "chunk": 900,
        "uuid": stem,
        "clip_id": "clip_000000",
        "window_start": 0,
        "anchor_displacement_m": FWD_DISPLACEMENT_M,
        "frame_indices": torch.arange(CLIP_FRAMES, dtype=torch.int64),
        "source_video": str(video_path),
        "note": "nexar dashcam -> treat-as-nv; depth=DepthAnythingV2; pose=synthetic const-velocity",
    }


def stem_for(nexar_root: Path, video_path: Path) -> str:
    rel = video_path.relative_to(nexar_root).with_suffix("")
    return "nexar_" + "_".join(rel.parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nexar-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--num-clips", type=int, default=20, help="-1 = all (after include/shard filters)")
    ap.add_argument("--include", type=str, default=None, help="only videos whose path contains this substring (e.g. 'test-')")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dump-frames", action="store_true", help="also save the 41 frames as PNGs in <clip>/frames/ (for MapAnything pose)")
    ap.add_argument("--pose-mode", choices=["synthetic", "mapanything"], default="synthetic",
                    help="synthetic=write const-velocity pose.pt; mapanything=skip (pose.pt written by MapAnything step)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos = sorted(Path(p) for p in glob.glob(str(args.nexar_root / "**" / "*.mp4"), recursive=True))
    if args.include:
        videos = [v for v in videos if args.include in str(v)]
    if not videos:
        raise SystemExit(f"no mp4 under {args.nexar_root} (include={args.include})")
    videos = videos[args.shard_index::args.num_shards]   # deterministic shard slice
    if args.num_clips > 0:
        videos = videos[: args.num_clips]
    print(f"converting {len(videos)} Nexar clips -> {args.out_root}", flush=True)

    device = torch.device(args.device)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.depth_model)
    model = AutoModelForDepthEstimation.from_pretrained(
        args.depth_model, torch_dtype=torch.float16
    ).to(device).eval()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    for i, vp in enumerate(videos, 1):
        stem = stem_for(args.nexar_root, vp)
        clip_dir = args.out_root / "chunk_900" / stem / "clip_000000"
        done_marker = (clip_dir / "front.mp4").exists() and (clip_dir / "meta.pt").exists()
        if done_marker and not args.overwrite:
            print(f"[{i}/{len(videos)}] skip existing {stem}", flush=True)
        else:
            if clip_dir.exists():
                import shutil
                shutil.rmtree(clip_dir)
            clip_dir.mkdir(parents=True, exist_ok=True)
            frames_u8, eff_fps = load_and_resize(vp)
            depth = estimate_depth(frames_u8, model, processor, device)
            _write_mp4(clip_dir / "front.mp4", frames_u8, max(1, int(round(eff_fps))))
            torch.save({"depth_sequence": depth,
                        "source_indices": torch.arange(CLIP_FRAMES)}, clip_dir / "front_depth.pt")
            if args.dump_frames:
                from PIL import Image
                fr_dir = clip_dir / "frames"; fr_dir.mkdir(exist_ok=True)
                for j, fr in enumerate(frames_u8):
                    Image.fromarray(fr.permute(1, 2, 0).numpy()).save(fr_dir / f"frame_{j:03d}.png")
            if args.pose_mode == "synthetic":
                torch.save({"T_anchor_front": synthetic_pose()}, clip_dir / "pose.pt")
            # pose_mode == "mapanything": pose.pt is written later by the MapAnything step
            torch.save(build_meta(stem, eff_fps, vp), clip_dir / "meta.pt")
            print(f"[{i}/{len(videos)}] {stem}  fps={eff_fps:.1f}", flush=True)
        manifest_lines.append(json.dumps({"chunk": 900, "uuid": stem, "clip_id": "clip_000000"}))

    (args.out_root / "manifest_clips.jsonl").write_text("\n".join(manifest_lines) + "\n")
    print(f"DONE: wrote manifest with {len(manifest_lines)} clips -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
