from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from openlongtail.data.depth_warp import forward_splat_warp, se3_inverse_torch
from openlongtail.data.rig_parquet import SENSOR_ORDER
from openlongtail.data.text_emb_cache import load_text_embedding
from openlongtail.inference.teacher_p3 import _decode_targets
from openlongtail.inference.teacher_warp import InferenceBundleWarp, inference_single_target
from openlongtail.models.wan_vae import load_wan21_vae
from openlongtail.scripts.build_warp import downsample_visibility_to_latent, vae_encode_warped
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _to_uint8, _write_mp4
from openlongtail.scripts.inference_smoke import CONFIGS, _build_dit
from openlongtail.training.schedulers import FlowMatchScheduler
CROSS_TARGET_VIEWS: tuple[int, int] = (1, 2)
VIEW_NAMES: dict[int, str] = {1: 'cross_left', 2: 'cross_right'}

def collect_clip_dirs(test_data_root: Path, max_clips: int, shard_index: int, num_shards: int) -> list[Path]:
    if num_shards <= 0:
        raise ValueError(f'num_shards must be > 0, got {num_shards}')
    if not 0 <= shard_index < num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards}), got {shard_index}')
    manifest = test_data_root / 'manifest_clips.jsonl'
    clips: list[Path] = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            clips.append(test_data_root / f"chunk_{int(item['chunk']):03d}" / str(item['uuid']) / str(item['clip_id']))
    else:
        clips = sorted(test_data_root.glob('chunk_*/*/clip_*'))
    required = ('front.mp4', 'front_depth.pt', 'pose.pt', 'meta.pt')
    existing = [clip for clip in clips if all(((clip / name).exists() for name in required))]
    sharded = existing[shard_index::num_shards]
    if max_clips > 0:
        sharded = sharded[:max_clips]
    return sharded

def reconstruct_e_all(k_front: torch.Tensor, e_front: torch.Tensor, *, reference_k: torch.Tensor, reference_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if k_front.shape != (3, 3):
        raise ValueError(f'expected k_front (3,3), got {tuple(k_front.shape)}')
    if e_front.shape != (4, 4):
        raise ValueError(f'expected e_front (4,4), got {tuple(e_front.shape)}')
    if reference_k.shape != (6, 3, 3):
        raise ValueError(f'expected reference_k (6,3,3), got {tuple(reference_k.shape)}')
    if reference_e.shape != (6, 4, 4):
        raise ValueError(f'expected reference_e (6,4,4), got {tuple(reference_e.shape)}')
    k_all = k_front.float().unsqueeze(0).expand(6, 3, 3).clone()
    e_all = torch.empty_like(reference_e.float())
    e_all[0] = e_front.float()
    ref_front = reference_e[0].float()
    for view_id in range(1, 6):
        ref_front_to_target = torch.linalg.inv(reference_e[view_id].float()) @ ref_front
        e_all[view_id] = e_front.float() @ torch.linalg.inv(ref_front_to_target)
    return (k_all, e_all)

def _read_front_mp4(path: Path) -> torch.Tensor:
    import imageio.v3 as iio
    frames = iio.imread(path, plugin='pyav')
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'expected RGB video at {path}, got shape {frames.shape}')
    rgb = torch.from_numpy(frames[:41]).permute(0, 3, 1, 2).contiguous()
    if rgb.shape[0] != 41:
        raise ValueError(f'expected 41 frames in {path}, got {rgb.shape[0]}')
    return rgb.to(torch.uint8)

def _load_depth(path: Path, out_h: int, out_w: int) -> torch.Tensor:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    depth = payload['depth_sequence'].float()
    if depth.ndim == 4 and depth.shape[1] == 1:
        pass
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)
    else:
        raise ValueError(f'unexpected depth_sequence shape {tuple(depth.shape)} in {path}')
    if float(depth.max()) < 5.0:
        depth = 1.0 / depth.clamp(min=0.01, max=1.0)
        depth = depth.clamp(max=100.0)
    depth = depth[:41]
    if depth.shape[0] != 41:
        raise ValueError(f'expected 41 depth frames in {path}, got {depth.shape[0]}')
    return F.interpolate(depth, size=(out_h, out_w), mode='bilinear', align_corners=False).squeeze(1)

def _find_reference_cache(latent_cache_root: Path) -> Path:
    per_uuid = latent_cache_root / 'per_uuid'
    for path in sorted(per_uuid.glob('*/clip_000000.pt')):
        return path
    raise FileNotFoundError(f'no reference latent cache found under {per_uuid}')

def _load_reference_calibration(latent_cache_root: Path, reference_cache: Path | None) -> tuple[torch.Tensor, torch.Tensor, Path]:
    path = reference_cache or _find_reference_cache(latent_cache_root)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return (payload['K'].float(), payload['E'].float(), path)

def _make_testdata_warp_provider(front_rgb: torch.Tensor, front_depth: torch.Tensor, k_all: torch.Tensor, e_all: torch.Tensor, vae: object, device: torch.device, splat_radius: int) -> tuple[Any, dict[str, Any]]:
    front_t = front_rgb.to(device=device)
    depth_t = front_depth.to(device=device)
    k_dev = k_all.to(device=device, dtype=torch.float32)
    e_dev = e_all.to(device=device, dtype=torch.float32)
    (h, w) = (int(front_rgb.shape[-2]), int(front_rgb.shape[-1]))
    warped_by_view: dict[int, torch.Tensor] = {}
    vis_by_view: dict[int, torch.Tensor] = {}
    summary: dict[str, Any] = {}
    for view_id in CROSS_TARGET_VIEWS:
        t_rel = se3_inverse_torch(e_dev[view_id].unsqueeze(0)).squeeze(0) @ e_dev[0]
        (warped_rgb, vis_pix) = forward_splat_warp(front_t, depth_t, k_dev[0], k_dev[view_id], t_rel, out_h=h, out_w=w, splat_radius=splat_radius)
        warped_by_view[view_id] = vae_encode_warped(vae, warped_rgb, device=device, dtype=torch.bfloat16).to(device=device)
        vis_h_lat = vis_pix.shape[-2] // 8
        vis_w_lat = vis_pix.shape[-1] // 8
        vis_by_view[view_id] = downsample_visibility_to_latent(vis_pix.detach().cpu(), h_lat=vis_h_lat, w_lat=vis_w_lat).to(device=device, dtype=torch.float16)
        summary[str(view_id)] = {'view_name': VIEW_NAMES[view_id], 'pixel_visibility_pct': float(vis_pix.float().mean().detach().cpu().item() * 100.0)}

    def provider(target_view: int) -> tuple[torch.Tensor, torch.Tensor]:
        if int(target_view) not in warped_by_view:
            shape = next(iter(warped_by_view.values())).shape
            z = torch.zeros(shape, device=device, dtype=torch.bfloat16)
            v = torch.zeros((1, shape[1], shape[2], shape[3]), device=device, dtype=torch.float16)
            return (z.unsqueeze(0), v.unsqueeze(0))
        return (warped_by_view[int(target_view)].unsqueeze(0), vis_by_view[int(target_view)].unsqueeze(0))
    return (provider, summary)

def _load_clip_tensors(clip_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    front = _read_front_mp4(clip_dir / 'front.mp4')
    depth = _load_depth(clip_dir / 'front_depth.pt', int(front.shape[-2]), int(front.shape[-1]))
    pose = torch.load(clip_dir / 'pose.pt', map_location='cpu', weights_only=False)['T_anchor_front'].float()
    meta = torch.load(clip_dir / 'meta.pt', map_location='cpu', weights_only=False)
    return (front, depth, pose, meta['K'].float(), meta)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--config', choices=sorted(CONFIGS), default='openlongtail_1p3b')
    parser.add_argument('--wan21-vace-dir', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B'))
    parser.add_argument('--latent-cache-root', type=Path, default=Path('cache/latent_t41_stride4_v1'))
    parser.add_argument('--reference-cache', type=Path, default=None)
    parser.add_argument('--max-clips', type=int, default=-1)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--start-sigma', type=float, default=1.0)
    parser.add_argument('--guide-scale', type=float, default=3.5)
    parser.add_argument('--shared-noise-alpha', type=float, default=0.5)
    parser.add_argument('--splat-radius', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260510)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--treat-as-nv', action='store_true', help='Treat input as if captured with PAI camera rig: override k_all and e_all with PAI reference per-view K and E. Use for cross-dataset inputs like Waymo where target views must follow NV rig layout.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = CONFIGS[args.config]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = collect_clip_dirs(args.test_data_root, args.max_clips, args.shard_index, args.num_shards)
    (reference_k, reference_e, reference_cache) = _load_reference_calibration(args.latent_cache_root, args.reference_cache)
    (dit, load_info) = _build_dit(args.config, device, args.wan21_vace_dir, args.checkpoint_dir)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    text_encoder = CachedTextEncoder(text_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=null_mask.unsqueeze(0).to(device=device), null_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), null_mask=null_mask.unsqueeze(0).to(device=device))
    root_summary: dict[str, Any] = {'test_data_root': str(args.test_data_root), 'output_dir': str(output_dir), 'checkpoint_dir': str(args.checkpoint_dir), 'reference_cache': str(reference_cache), 'target_views': list(CROSS_TARGET_VIEWS), 'target_view_names': {str(k): v for (k, v) in VIEW_NAMES.items()}, 'sensors': list(SENSOR_ORDER), 'num_clips': len(clips), 'load_info': load_info, 'clips': []}
    (output_dir / 'config_used.json').write_text(json.dumps(_jsonable(vars(args)), indent=2, sort_keys=True))
    for (idx, clip_dir) in enumerate(clips, start=1):
        rel_key = clip_dir.relative_to(args.test_data_root)
        clip_out = output_dir / rel_key
        done_path = clip_out / 'clip_done.json'
        if done_path.exists() and (not args.overwrite):
            root_summary['clips'].append({'clip': str(rel_key), 'status': 'skipped_existing'})
            continue
        clip_out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f'[{idx}/{len(clips)}] {rel_key}', flush=True)
        (front, depth, pose, k_front, meta) = _load_clip_tensors(clip_dir)
        if args.treat_as_nv:
            k_all = reference_k.clone()
            e_all = reference_e.clone()
        else:
            (k_all, e_all) = reconstruct_e_all(k_front, meta['E_rig_front'].float(), reference_k=reference_k, reference_e=reference_e)
        (warp_provider, warp_summary) = _make_testdata_warp_provider(front, depth, k_all, e_all, vae, device, args.splat_radius)
        bundle = InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=FlowMatchScheduler(shift=config.model.sample_shift), warp_provider=warp_provider)
        from openlongtail.inference.teacher_p3 import _encode_front
        z_front = _encode_front(vae, front.to(device=device)).to(device=device)
        batch_shape = z_front.unsqueeze(0).shape
        generator = torch.Generator(device=device).manual_seed(args.seed + idx + args.shard_index * 100000)
        shared = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
        generated: dict[int, torch.Tensor] = {}
        pred_latents: list[torch.Tensor] = []
        for view_id in CROSS_TARGET_VIEWS:
            private = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
            alpha = float(args.shared_noise_alpha)
            target_noise = alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * private
            latent = inference_single_target(z_front, k_all.to(device=device), e_all.to(device=device), pose.to(device=device), str(rel_key), bundle, view_id, condition_latents_by_view=generated, target_noise=target_noise, num_steps=args.num_steps, guide_scale=args.guide_scale, start_sigma=args.start_sigma, generator=generator)
            generated[view_id] = latent
            pred_latents.append(latent)
        pred = _decode_targets(vae, torch.stack(pred_latents, dim=0)).detach().cpu()
        pred_tchw = pred.permute(0, 2, 1, 3, 4).contiguous()
        fps = int(meta.get('src_fps', config.data.target_fps))
        outputs = {'front_input': _write_mp4(clip_out / 'front_input.mp4', front, fps), 'pred': {}}
        for (local_idx, view_id) in enumerate(CROSS_TARGET_VIEWS):
            outputs['pred'][str(view_id)] = _write_mp4(clip_out / f'pred_{VIEW_NAMES[view_id]}.mp4', pred_tchw[local_idx], fps)
        elapsed = time.time() - started
        clip_summary = {'clip': str(rel_key), 'status': 'ok', 'elapsed_sec': elapsed, 'outputs': outputs, 'warp_summary': warp_summary, 'meta': {'chunk': int(meta.get('chunk', -1)), 'uuid': str(meta.get('uuid', '')), 'clip_id': str(meta.get('clip_id', '')), 'window_start': int(meta.get('window_start', -1)), 'anchor_displacement_m': float(meta.get('anchor_displacement_m', 0.0))}}
        done_path.write_text(json.dumps(_jsonable(clip_summary), indent=2, sort_keys=True))
        root_summary['clips'].append(clip_summary)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (output_dir / 'summary.json').write_text(json.dumps(_jsonable(root_summary), indent=2, sort_keys=True))
    print(f'DONE: {len(clips)} clips -> {output_dir}', flush=True)
if __name__ == '__main__':
    main()
