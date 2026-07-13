from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
import torch
from openlongtail.data.text_emb_cache import load_text_embedding
from openlongtail.inference.teacher_p3 import _decode_targets, _encode_front
from openlongtail.inference.teacher_warp import InferenceBundleWarp, inference_single_target, make_sidecar_warp_provider
from openlongtail.models.wan_vae import load_wan21_vae
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _make_comparison_grid, _write_mp4
from openlongtail.scripts.inference_smoke import CONFIGS, _build_dit
from openlongtail.training.schedulers import FlowMatchScheduler
ALL_TARGET_VIEWS: tuple[int, ...] = (1, 2, 3, 4, 5)
REAR_VIEWS: frozenset[int] = frozenset({3, 4, 5})
VIEW_NAMES: dict[int, str] = {1: 'cross_left', 2: 'cross_right', 3: 'rear_left', 4: 'rear_right', 5: 'rear_tele'}

def _guide_for_view(view_id: int, cross_guide: float, rear_guide: float) -> float:
    return rear_guide if view_id in REAR_VIEWS else cross_guide

def collect_v3_clips(root: Path, max_clips: int, shard_index: int, num_shards: int, first_clip_per_uuid: bool=False) -> list[tuple[str, int, Path]]:
    clips: list[tuple[str, int, Path]] = []
    for uuid_dir in sorted((root / 'per_uuid').iterdir()):
        if not uuid_dir.is_dir():
            continue
        uuid_clips: list[tuple[str, int, Path]] = []
        for cp in sorted(uuid_dir.glob('clip_*.pt')):
            stem = cp.stem
            if '_' in stem.replace('clip_', '', 1):
                continue
            try:
                cid = int(stem.replace('clip_', ''))
            except ValueError:
                continue
            uuid_clips.append((uuid_dir.name, cid, cp))
            if first_clip_per_uuid:
                break
        clips.extend(uuid_clips)
    sharded = clips[shard_index::num_shards]
    if max_clips > 0:
        sharded = sharded[:max_clips]
    return sharded

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--v3-root', type=Path, required=True, help='latent-cache dir (with per_uuid/ + text_cache/)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--config', choices=sorted(CONFIGS), default='openlongtail_1p3b')
    p.add_argument('--wan21-vace-dir', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B'))
    p.add_argument('--max-clips', type=int, default=5)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--first-clip-per-uuid', action='store_true', help='keep only the lowest-numbered clip per UUID (distinct scenes)')
    p.add_argument('--num-steps', type=int, default=50)
    p.add_argument('--start-sigma', type=float, default=1.0)
    p.add_argument('--cross-guide', type=float, default=3.5, help='CFG scale for cross views 1/2 (default 3.5)')
    p.add_argument('--rear-guide', type=float, default=7.0, help='CFG scale for rear views 3/4/5 (default 7.0)')
    p.add_argument('--guide-scale', type=float, default=None, help='DEPRECATED single-scale override; if set, applies to ALL views (disables per-view cross/rear guidance).')
    p.add_argument('--shared-noise-alpha', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=20260512)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def _load_text_for_uuid(v3_root: Path, uuid: str, config: Any, device: torch.device) -> tuple[CachedTextEncoder, str]:
    text_pkg = torch.load(v3_root / 'text_cache' / 'per_uuid' / f'{uuid}.pt', map_location='cpu', weights_only=False)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    text_emb = text_pkg.get('text_emb', text_pkg.get('emb'))
    text_mask = text_pkg.get('text_mask', text_pkg.get('mask', text_pkg.get('attention_mask')))
    if text_emb is None or text_mask is None:
        raise KeyError(f'text_pkg missing emb/mask, keys={list(text_pkg.keys())}')
    return (CachedTextEncoder(text_emb=text_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=text_mask.unsqueeze(0).to(device=device), null_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), null_mask=null_mask.unsqueeze(0).to(device=device)), str(text_pkg.get('prompt', '')))

def _decode_latent_to_video(vae, z: torch.Tensor) -> torch.Tensor:
    out = _decode_targets(vae, z.unsqueeze(0))[0]
    return out

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = CONFIGS[args.config]
    clips = collect_v3_clips(args.v3_root, args.max_clips, args.shard_index, args.num_shards, first_clip_per_uuid=args.first_clip_per_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (dit, load_info) = _build_dit(args.config, device, args.wan21_vace_dir, args.checkpoint_dir)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    root_summary: dict[str, Any] = {'v3_root': str(args.v3_root), 'checkpoint_dir': str(args.checkpoint_dir), 'target_views': list(ALL_TARGET_VIEWS), 'num_clips': len(clips), 'load_info': load_info, 'clips': []}
    (args.output_dir / 'config_used.json').write_text(json.dumps(_jsonable(vars(args)), indent=2, sort_keys=True))
    for (idx, (uuid, cid, clip_pt)) in enumerate(clips, start=1):
        clip_id = f'clip_{cid:06d}'
        clip_out = args.output_dir / uuid / clip_id
        done_path = clip_out / 'clip_done.json'
        if done_path.exists() and (not args.overwrite):
            root_summary['clips'].append({'clip': f'{uuid}/{clip_id}', 'status': 'skipped'})
            continue
        clip_out.mkdir(parents=True, exist_ok=True)
        print(f'[{idx}/{len(clips)}] {uuid}/{clip_id}', flush=True)
        started = time.time()
        latent_pkg = torch.load(clip_pt, map_location='cpu', weights_only=False)
        p4_path = clip_pt.with_name(f'{clip_id}_p4.pt')
        sidecar_path = clip_pt.with_name(f'{clip_id}_warp.pt')
        adj_sidecar_path = clip_pt.with_name(f'{clip_id}_lookback.pt')
        if not sidecar_path.exists():
            print(f'  SKIP: missing {sidecar_path}', flush=True)
            continue
        pose = torch.load(p4_path, map_location='cpu', weights_only=False)['T_anchor_front'].float()
        warp_provider = make_sidecar_warp_provider(str(sidecar_path), str(adj_sidecar_path) if adj_sidecar_path.exists() else None, device=device)
        (text_encoder, prompt) = _load_text_for_uuid(args.v3_root, uuid, config, device)
        bundle = InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=FlowMatchScheduler(shift=config.model.sample_shift), warp_provider=warp_provider)
        z_all = latent_pkg['z_all'].to(device=device, dtype=torch.bfloat16)
        z_front = z_all[0]
        k_all = latent_pkg['K'].float()
        e_all = latent_pkg['E'].float()
        batch_shape = z_front.unsqueeze(0).shape
        generator = torch.Generator(device=device).manual_seed(args.seed + idx + args.shard_index * 100000)
        shared = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
        generated: dict[int, torch.Tensor] = {}
        pred_latents: list[torch.Tensor] = []
        for view_id in ALL_TARGET_VIEWS:
            private = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
            alpha = float(args.shared_noise_alpha)
            target_noise = alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * private
            guide = args.guide_scale if args.guide_scale is not None else _guide_for_view(view_id, args.cross_guide, args.rear_guide)
            latent = inference_single_target(z_front, k_all.to(device=device), e_all.to(device=device), pose.to(device=device), prompt, bundle, view_id, condition_latents_by_view=generated, target_noise=target_noise, num_steps=args.num_steps, guide_scale=guide, start_sigma=args.start_sigma, generator=generator)
            generated[view_id] = latent
            pred_latents.append(latent)
        pred = _decode_targets(vae, torch.stack(pred_latents, dim=0)).detach().cpu()
        gt = _decode_targets(vae, z_all[1:6].detach()).detach().cpu()
        front_dec = _decode_targets(vae, z_all[0:1].detach()).detach().cpu()
        pred_tchw = pred.permute(0, 2, 1, 3, 4).contiguous()
        gt_tchw = gt.permute(0, 2, 1, 3, 4).contiguous()
        front_tchw = front_dec[0].permute(1, 0, 2, 3).contiguous()
        outputs: dict[str, Any] = {'front_input': _write_mp4(clip_out / 'front_input.mp4', front_tchw, fps=16), 'gt': {}, 'pred': {}}
        for (local_idx, view_id) in enumerate(ALL_TARGET_VIEWS):
            outputs['gt'][str(view_id)] = _write_mp4(clip_out / f'gt_{VIEW_NAMES[view_id]}.mp4', gt_tchw[local_idx], fps=16)
            outputs['pred'][str(view_id)] = _write_mp4(clip_out / f'pred_{VIEW_NAMES[view_id]}.mp4', pred_tchw[local_idx], fps=16)
        grid = _make_comparison_grid(front_tchw, gt_tchw, pred_tchw, list(ALL_TARGET_VIEWS))
        outputs['comparison_grid'] = _write_mp4(clip_out / 'comparison_grid.mp4', grid, fps=16)
        elapsed = time.time() - started
        clip_summary = {'clip': f'{uuid}/{clip_id}', 'status': 'ok', 'elapsed_sec': elapsed, 'outputs': outputs, 'prompt': prompt}
        done_path.write_text(json.dumps(_jsonable(clip_summary), indent=2, sort_keys=True))
        root_summary['clips'].append(clip_summary)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (args.output_dir / 'summary.json').write_text(json.dumps(_jsonable(root_summary), indent=2, sort_keys=True))
    print(f'DONE: {len(clips)} clips -> {args.output_dir}', flush=True)
if __name__ == '__main__':
    main()
