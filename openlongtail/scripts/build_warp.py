from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image
from openlongtail.data.depth_warp import forward_splat_warp, se3_inverse_torch
from openlongtail.models.wan_vae import load_wan21_vae, normalize_rgb_for_wan_vae
SENSOR_NAMES = ['camera_front_wide_120fov_undistorted', 'camera_cross_left_120fov_undistorted', 'camera_cross_right_120fov_undistorted', 'camera_rear_left_70fov_undistorted', 'camera_rear_right_70fov_undistorted', 'camera_rear_tele_30fov_undistorted']
FRONT_IDX = 0
TARGET_IDS = (1, 2, 3, 4, 5)

def resize_rgb(rgb_hw3: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(rgb_hw3).resize((w, h), Image.BILINEAR))

def load_front_rgb_clip(uuid: str, data_root: Path, abs_indices: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    sensor = SENSOR_NAMES[FRONT_IDX]
    mp4 = data_root / uuid / 'all_views_undistorted_simplecalib' / 'camera' / sensor / f'{uuid}.{sensor}.mp4'
    full = iio.imread(mp4, plugin='pyav')
    out = np.stack([resize_rgb(full[int(i)], out_h, out_w) for i in abs_indices], axis=0)
    return out

def load_depth_clip(uuid: str, data_root: Path, abs_indices: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    p = data_root / uuid / 'depthcrafter_cache' / 'fullseq_h_384_w_672' / 'front_depth.pt'
    pkg = torch.load(p, map_location='cpu', weights_only=False)
    seq = pkg['depth_sequence']
    src_indices = pkg['source_indices']
    src_to_pos = {int(s): pos for (pos, s) in enumerate(src_indices)}
    rows = [seq[src_to_pos[int(i)]] for i in abs_indices]
    depth = torch.stack(rows, dim=0).float()
    depth = F.interpolate(depth, size=(out_h, out_w), mode='bilinear', align_corners=False).squeeze(1)
    return depth

@torch.no_grad()
def vae_encode_warped(vae: object, warped_rgb_uint8: torch.Tensor, device: torch.device, dtype: torch.dtype=torch.bfloat16) -> torch.Tensor:
    if warped_rgb_uint8.dtype != torch.uint8:
        raise TypeError(f'expected uint8, got {warped_rgb_uint8.dtype}')
    rgb = warped_rgb_uint8.to(device=device, dtype=torch.float32)
    rgb_chw = rgb.permute(1, 0, 2, 3).contiguous()
    video = normalize_rgb_for_wan_vae(rgb_chw, source_range='raw_0_255')
    encoded = vae.encode([video])[0].detach().to('cpu', dtype=dtype)
    return encoded

def downsample_visibility_to_latent(vis_pix: torch.Tensor, t_lat: int=11, h_lat: int=60, w_lat: int=104) -> torch.Tensor:
    assert vis_pix.dim() == 4 and vis_pix.shape[1] == 1, vis_pix.shape
    t_rgb = vis_pix.shape[0]
    if t_rgb != 41:
        raise ValueError(f'expected 41 RGB frames, got {t_rgb}')
    spatial = F.interpolate(vis_pix.float(), size=(h_lat, w_lat), mode='bilinear', align_corners=False)
    chunks = [spatial[0:1]]
    for i in range(1, t_lat):
        lo = 4 * i - 3
        hi = 4 * i + 1
        chunks.append(spatial[lo:hi].max(dim=0, keepdim=True).values)
    out = torch.cat(chunks, dim=0)
    out = out.permute(1, 0, 2, 3).to(torch.float16)
    return out

def list_clips_for_uuid(uuid_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for path in sorted(uuid_dir.glob('clip_*.pt')):
        name = path.name
        if '_p4' in name or '_warp' in name:
            continue
        try:
            cid = int(name.split('_')[1].split('.')[0])
            out.append((cid, path))
        except Exception:
            continue
    return out

def build_one_clip(uuid: str, data_root: Path, cache_path: Path, vae: object, device: torch.device, splat_radius: int) -> dict:
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    K = cache['K'].float()
    E = cache['E'].float()
    front_indices = cache['frame_indices']['camera_front_wide_120fov_undistorted'].numpy()
    (H_t, W_t) = cache['output_size']
    front_rgb_np = load_front_rgb_clip(uuid, data_root, front_indices, H_t, W_t)
    depth_t = load_depth_clip(uuid, data_root, front_indices, H_t, W_t).to(device)
    front_t = torch.from_numpy(front_rgb_np).permute(0, 3, 1, 2).contiguous().to(device)
    K_dev = K.to(device)
    E_dev = E.to(device)
    E_front = E_dev[FRONT_IDX]
    warped_latents = []
    vis_latents = []
    vis_pix_pct = []
    for t_id in TARGET_IDS:
        T_rel = se3_inverse_torch(E_dev[t_id].unsqueeze(0)).squeeze(0) @ E_front
        (warped, vis_pix) = forward_splat_warp(front_t, depth_t, K_dev[FRONT_IDX], K_dev[t_id], T_rel, out_h=H_t, out_w=W_t, splat_radius=splat_radius)
        z = vae_encode_warped(vae, warped, device=device, dtype=torch.bfloat16)
        warped_latents.append(z)
        vis_lat = downsample_visibility_to_latent(vis_pix.cpu()).to(torch.float16)
        vis_latents.append(vis_lat)
        vis_pix_pct.append(float(vis_pix.float().mean().cpu()))
    payload = {'warped_target_latents': torch.stack(warped_latents, dim=0).contiguous(), 'warped_target_visibility': torch.stack(vis_latents, dim=0).contiguous(), 'warped_target_visibility_pixel_pct': torch.tensor(vis_pix_pct, dtype=torch.float32), 'splat_radius': int(splat_radius), 'depth_source': 'depthcrafter_fullseq_h_384_w_672', 'target_view_ids': list(TARGET_IDS), 'uuid': uuid, 'clip_id': int(cache['clip_id']), 'sidecar_version': 'v0'}
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-cache-root', type=Path, default=Path('cache/latent_cache'))
    parser.add_argument('--data-root', type=Path, default=Path('data/by_uuid'))
    parser.add_argument('--vae-path', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--splat-radius', type=int, default=1)
    parser.add_argument('--max-clips', type=int, default=10, help='max total clips to process; -1 for all')
    parser.add_argument('--max-uuids', type=int, default=-1, help='max uuids to scan; -1 for all')
    parser.add_argument('--uuid-stride', type=int, default=1, help='take every N-th UUID (for parallel sharding)')
    parser.add_argument('--uuid-offset', type=int, default=0, help='start at this offset within the stride')
    parser.add_argument('--uuid', type=str, default=None, help='process exactly one UUID by name (V4 backfill mode); takes precedence over stride/offset')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--manifest', type=Path, default=None, help='optional jsonl manifest of (uuid, clip_id, status, elapsed_sec)')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'loading VAE from {args.vae_path}')
    vae = load_wan21_vae(args.vae_path, dtype=torch.bfloat16, device=device)
    per_uuid_root = args.latent_cache_root / 'per_uuid'
    if args.uuid:
        target = per_uuid_root / args.uuid
        if not target.is_dir():
            print(f'[error] --uuid {args.uuid} not found at {target}')
            return
        uuid_dirs = [target]
    else:
        uuid_dirs = sorted([p for p in per_uuid_root.iterdir() if p.is_dir()])
        if args.uuid_stride > 1 or args.uuid_offset > 0:
            uuid_dirs = uuid_dirs[args.uuid_offset::args.uuid_stride]
            print(f'sharding: stride={args.uuid_stride} offset={args.uuid_offset} → {len(uuid_dirs)} UUIDs assigned')
        if args.max_uuids > 0:
            uuid_dirs = uuid_dirs[:args.max_uuids]
    n_done = 0
    t_total = time.time()
    for uuid_dir in uuid_dirs:
        uuid = uuid_dir.name
        depth_p = args.data_root / uuid / 'depthcrafter_cache' / 'fullseq_h_384_w_672' / 'front_depth.pt'
        if not depth_p.exists():
            print(f'[skip] {uuid}: missing depth at {depth_p}')
            continue
        if not depth_p.exists():
            print(f'[skip] {uuid}: no depth cache')
            continue
        clips = list_clips_for_uuid(uuid_dir)
        for (clip_id, cache_path) in clips:
            sidecar_path = uuid_dir / f'clip_{clip_id:06d}_warp.pt'
            if sidecar_path.exists() and (not args.overwrite):
                continue
            t0 = time.time()
            try:
                payload = build_one_clip(uuid, args.data_root, cache_path, vae, device, args.splat_radius)
                tmp = sidecar_path.with_suffix('.pt.tmp')
                torch.save(payload, tmp)
                tmp.rename(sidecar_path)
                elapsed = time.time() - t0
                vis_pct = payload['warped_target_visibility_pixel_pct'].tolist()
                size_mb = sidecar_path.stat().st_size / 1024 ** 2
                print(f"[ok] {uuid}/clip_{clip_id:06d}: {elapsed:.1f}s {size_mb:.1f}MB  vis(per-target)={['%.1f' % (100 * v) for v in vis_pct]}")
                if args.manifest:
                    args.manifest.parent.mkdir(parents=True, exist_ok=True)
                    with args.manifest.open('a') as h:
                        h.write(json.dumps({'uuid': uuid, 'clip_id': clip_id, 'status': 'ok', 'elapsed_sec': round(elapsed, 2), 'size_mb': round(size_mb, 2), 'vis_pct': [round(100 * v, 1) for v in vis_pct]}) + '\n')
                n_done += 1
                if args.max_clips > 0 and n_done >= args.max_clips:
                    break
            except Exception as exc:
                print(f'[fail] {uuid}/clip_{clip_id:06d}: {exc}')
                if args.manifest:
                    with args.manifest.open('a') as h:
                        h.write(json.dumps({'uuid': uuid, 'clip_id': clip_id, 'status': 'fail', 'error': str(exc)[:200]}) + '\n')
        if args.max_clips > 0 and n_done >= args.max_clips:
            break
    print(f'\ndone: {n_done} clips in {(time.time() - t_total) / 60:.1f} min')
if __name__ == '__main__':
    main()
