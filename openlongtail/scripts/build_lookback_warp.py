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
from openlongtail.training.forward_ray_p61 import build_p61_anchor_camera_transforms
SENSOR_NAMES = ['camera_front_wide_120fov_undistorted', 'camera_cross_left_120fov_undistorted', 'camera_cross_right_120fov_undistorted', 'camera_rear_left_70fov_undistorted', 'camera_rear_right_70fov_undistorted', 'camera_rear_tele_30fov_undistorted']
FRONT_IDX = 0
REAR_TARGETS = (3, 4, 5)
LOOKBACK_OFFSET = {3: 4, 4: 4, 5: 6}

def rgb_t_to_latent_lf(rgb_t: int) -> int:
    if rgb_t == 0:
        return 0
    return (rgb_t + 3) // 4

def latent_lf_to_rgb_repr(latent_lf: int) -> int:
    if latent_lf == 0:
        return 0
    return 4 * latent_lf - 1

def resize_rgb(rgb_hw3: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(rgb_hw3).resize((w, h), Image.BILINEAR))

def load_front_full_rgb(uuid: str, data_root: Path) -> np.ndarray:
    sensor = SENSOR_NAMES[FRONT_IDX]
    mp4 = data_root / uuid / 'all_views_undistorted_simplecalib' / 'camera' / sensor / f'{uuid}.{sensor}.mp4'
    return iio.imread(mp4, plugin='pyav')

def load_full_depth(uuid: str, data_root: Path):
    p = data_root / uuid / 'depthcrafter_cache' / 'fullseq_h_384_w_672' / 'front_depth.pt'
    pkg = torch.load(p, map_location='cpu', weights_only=False)
    seq = pkg['depth_sequence']
    src_indices = pkg['source_indices']
    src_to_pos = {int(s): pos for (pos, s) in enumerate(src_indices)}
    return (seq, src_to_pos)

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
    return out.permute(1, 0, 2, 3).to(torch.float16)

def list_clips_for_uuid(uuid_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for path in sorted(uuid_dir.glob('clip_*.pt')):
        name = path.name
        if '_p4' in name or '_warp' in name or '_lookback' in name:
            continue
        try:
            cid = int(name.split('_')[1].split('.')[0])
            out.append((cid, path))
        except Exception:
            continue
    return out

@torch.no_grad()
def build_one_clip_adjacent(uuid: str, data_root: Path, cache_path: Path, p4_path: Path, front_full_rgb_cache: np.ndarray, depth_seq, depth_src_to_pos, vae, device: torch.device, splat_radius: int=1, vae_batch_size: int=8) -> dict:
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    p4 = torch.load(p4_path, map_location='cpu', weights_only=False)
    K = cache['K'].float()
    E = cache['E'].float()
    front_indices = cache['frame_indices']['camera_front_wide_120fov_undistorted'].numpy()
    (H_t, W_t) = cache['output_size']
    T_anchor_front = p4['T_anchor_front'].float()
    T_anchor_cam = build_p61_anchor_camera_transforms(E.unsqueeze(0), T_anchor_front.unsqueeze(0)).squeeze(0).to(device)
    K_dev = K.to(device)
    E_dev = E.to(device)
    warped_seqs_rgb: dict[int, torch.Tensor] = {}
    warped_seqs_vis: dict[int, torch.Tensor] = {}
    needed_source_lfs = set()
    for target_idx in REAR_TARGETS:
        K_off = LOOKBACK_OFFSET[target_idx]
        for rgb_t in range(41):
            target_lf = rgb_t_to_latent_lf(rgb_t)
            source_lf = max(0, target_lf - K_off)
            needed_source_lfs.add(source_lf)
    src_lf_to_rgb_dev: dict[int, torch.Tensor] = {}
    src_lf_to_depth_dev: dict[int, torch.Tensor] = {}
    for source_lf in sorted(needed_source_lfs):
        src_rgb_local = latent_lf_to_rgb_repr(source_lf)
        abs_idx = int(front_indices[src_rgb_local])
        rgb_full = front_full_rgb_cache[abs_idx]
        rgb = resize_rgb(rgb_full, H_t, W_t)
        depth = depth_seq[depth_src_to_pos[abs_idx]].float()
        depth_resized = F.interpolate(depth.unsqueeze(0), size=(H_t, W_t), mode='bilinear', align_corners=False).squeeze(0)
        src_lf_to_rgb_dev[source_lf] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(device)
        src_lf_to_depth_dev[source_lf] = depth_resized.to(device)
    vis_pix_total_per_target: dict[int, float] = {}
    for target_idx in REAR_TARGETS:
        K_off = LOOKBACK_OFFSET[target_idx]
        warped_rgb_frames: list[torch.Tensor] = []
        vis_pix_frames: list[torch.Tensor] = []
        for rgb_t in range(41):
            target_lf = rgb_t_to_latent_lf(rgb_t)
            source_lf = max(0, target_lf - K_off)
            T_src_anchor = T_anchor_cam[FRONT_IDX, source_lf]
            T_tgt_anchor = T_anchor_cam[target_idx, target_lf]
            T_src_to_tgt = se3_inverse_torch(T_tgt_anchor.unsqueeze(0)).squeeze(0) @ T_src_anchor
            (warped, vis) = forward_splat_warp(src_lf_to_rgb_dev[source_lf].unsqueeze(0), src_lf_to_depth_dev[source_lf], K_dev[FRONT_IDX], K_dev[target_idx], T_src_to_tgt, out_h=H_t, out_w=W_t, splat_radius=splat_radius)
            warped_rgb_frames.append(warped[0])
            vis_pix_frames.append(vis[0])
        warped_seq = torch.stack(warped_rgb_frames, dim=0)
        vis_seq = torch.stack(vis_pix_frames, dim=0)
        warped_seqs_rgb[target_idx] = warped_seq
        warped_seqs_vis[target_idx] = vis_seq
        vis_pix_total_per_target[target_idx] = float(vis_seq.float().mean().cpu())
    videos_to_encode = []
    for target_idx in REAR_TARGETS:
        rgb = warped_seqs_rgb[target_idx].float()
        chw = rgb.permute(1, 0, 2, 3).contiguous()
        videos_to_encode.append(normalize_rgb_for_wan_vae(chw, source_range='raw_0_255'))
    all_z: list[torch.Tensor] = []
    for i in range(0, len(videos_to_encode), vae_batch_size):
        chunk = videos_to_encode[i:i + vae_batch_size]
        encoded_chunk = vae.encode(chunk)
        for z in encoded_chunk:
            all_z.append(z.detach().to('cpu', dtype=torch.bfloat16))
    warped_adj = torch.zeros((5, 16, 11, 60, 104), dtype=torch.bfloat16)
    vis_adj = torch.zeros((5, 1, 11, 60, 104), dtype=torch.float16)
    for (k, target_idx) in enumerate(REAR_TARGETS):
        slot = target_idx - 1
        warped_adj[slot] = all_z[k]
        vis_adj[slot] = downsample_visibility_to_latent(warped_seqs_vis[target_idx].cpu())
    payload = {'adjacent_warped_target_latents': warped_adj.contiguous(), 'adjacent_warped_target_visibility': vis_adj.contiguous(), 'adjacent_warped_target_visibility_pixel_pct': torch.tensor([0.0, 0.0, vis_pix_total_per_target[3], vis_pix_total_per_target[4], vis_pix_total_per_target[5]], dtype=torch.float32), 'adjacent_lookback_offset_per_slot': torch.tensor([0, 0, LOOKBACK_OFFSET[3], LOOKBACK_OFFSET[4], LOOKBACK_OFFSET[5]], dtype=torch.long), 'splat_radius': int(splat_radius), 'depth_source': 'depthcrafter_fullseq_h_384_w_672', 'target_view_ids': [1, 2, 3, 4, 5], 'uuid': uuid, 'clip_id': int(cache['clip_id']), 'adj_sidecar_version': 'v0'}
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-cache-root', type=Path, default=Path('cache/latent_cache'))
    parser.add_argument('--data-root', type=Path, default=Path('data/by_uuid'))
    parser.add_argument('--vae-path', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--splat-radius', type=int, default=1)
    parser.add_argument('--vae-batch-size', type=int, default=8, help='batch multiple rear-target videos through VAE encode at once (H200 fits ≥8)')
    parser.add_argument('--max-clips', type=int, default=10)
    parser.add_argument('--max-uuids', type=int, default=-1)
    parser.add_argument('--uuid-stride', type=int, default=1)
    parser.add_argument('--uuid-offset', type=int, default=0)
    parser.add_argument('--uuid', type=str, default=None, help='process exactly one UUID by name (V4 backfill mode); takes precedence over stride/offset')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--manifest', type=Path, default=None)
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
            print(f'sharding: stride={args.uuid_stride} offset={args.uuid_offset} → {len(uuid_dirs)} UUIDs')
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
        try:
            front_full_rgb = load_front_full_rgb(uuid, args.data_root)
            (depth_seq, depth_src_to_pos) = load_full_depth(uuid, args.data_root)
        except Exception as exc:
            print(f'[skip] {uuid}: failed to load front/depth: {exc}')
            continue
        clips = list_clips_for_uuid(uuid_dir)
        for (clip_id, cache_path) in clips:
            sidecar_path = uuid_dir / f'clip_{clip_id:06d}_lookback.pt'
            p4_path = uuid_dir / f'clip_{clip_id:06d}_p4.pt'
            if not p4_path.exists():
                print(f'[skip] {uuid}/clip_{clip_id:06d}: missing p4 sidecar')
                continue
            if sidecar_path.exists() and (not args.overwrite):
                continue
            t0 = time.time()
            try:
                payload = build_one_clip_adjacent(uuid, args.data_root, cache_path, p4_path, front_full_rgb, depth_seq, depth_src_to_pos, vae, device, splat_radius=args.splat_radius, vae_batch_size=args.vae_batch_size)
                tmp = sidecar_path.with_suffix('.pt.tmp')
                torch.save(payload, tmp)
                tmp.rename(sidecar_path)
                elapsed = time.time() - t0
                vis = payload['adjacent_warped_target_visibility_pixel_pct'].tolist()
                size_mb = sidecar_path.stat().st_size / 1024 ** 2
                print(f"[ok] {uuid}/clip_{clip_id:06d}: {elapsed:.1f}s {size_mb:.1f}MB  rear_vis={['%.1f' % (100 * v) for v in vis[2:]]}")
                if args.manifest:
                    args.manifest.parent.mkdir(parents=True, exist_ok=True)
                    with args.manifest.open('a') as h:
                        h.write(json.dumps({'uuid': uuid, 'clip_id': clip_id, 'status': 'ok', 'elapsed_sec': round(elapsed, 2), 'size_mb': round(size_mb, 2), 'rear_vis_pct': [round(100 * v, 1) for v in vis[2:]]}) + '\n')
                n_done += 1
                if args.max_clips > 0 and n_done >= args.max_clips:
                    break
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f'[fail] {uuid}/clip_{clip_id:06d}: {exc}')
                if args.manifest:
                    with args.manifest.open('a') as h:
                        h.write(json.dumps({'uuid': uuid, 'clip_id': clip_id, 'status': 'fail', 'error': str(exc)[:200]}) + '\n')
        if args.max_clips > 0 and n_done >= args.max_clips:
            break
    print(f'\ndone: {n_done} clips in {(time.time() - t_total) / 60:.1f} min')
if __name__ == '__main__':
    main()
