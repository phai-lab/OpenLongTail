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
from openlongtail.inference.teacher_warp import InferenceBundleWarp, inference_single_target
from openlongtail.models.wan_vae import load_wan21_vae
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _write_mp4
from openlongtail.scripts.inference_smoke import CONFIGS, _build_dit
from openlongtail.scripts.inference_testdata_cross import _load_clip_tensors, _load_reference_calibration, collect_clip_dirs, reconstruct_e_all
from openlongtail.scripts.inference_register import build_live_warp_provider
from openlongtail.training.schedulers import FlowMatchScheduler
ALL_TARGET_VIEWS: tuple[int, ...] = (1, 2, 3, 4, 5)
REAR_VIEWS: frozenset[int] = frozenset({3, 4, 5})
VIEW_NAMES: dict[int, str] = {1: 'cross_left', 2: 'cross_right', 3: 'rear_left', 4: 'rear_right', 5: 'rear_tele'}

def _guide_for_view(view_id: int, cross_guide: float, rear_guide: float) -> float:
    return rear_guide if view_id in REAR_VIEWS else cross_guide

class _RuntimeUMT5Encoder:

    def __init__(self, device: str) -> None:
        self._device = device
        self._t5 = None
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure(self) -> None:
        if self._t5 is None:
            from openlongtail.scripts.precompute_text_emb import _load_t5
            print('[caption] loading UMT5 text encoder for caption embedding ...', flush=True)
            self._t5 = _load_t5(self._device)

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = prompt.strip()
        if key in self._cache:
            return self._cache[key]
        self._ensure()
        from openlongtail.scripts.precompute_text_emb import _encode_prompt
        (emb, mask) = _encode_prompt(self._t5, key, self._device)
        self._cache[key] = (emb, mask)
        return (emb, mask)

def _resolve_caption(caption_cache: Path | None, uuid: str, override: str | None) -> tuple[str, str]:
    if override:
        return (override.strip(), 'override')
    if caption_cache is not None and uuid:
        txt = caption_cache / 'per_uuid' / f'{uuid}.txt'
        if txt.exists():
            text = txt.read_text().strip()
            if text:
                return (text, 'cache')
    return ('', 'none(null)')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Dashcam 5-view inference')
    p.add_argument('--test-data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--config', choices=sorted(CONFIGS), default='openlongtail_1p3b')
    p.add_argument('--wan21-vace-dir', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B'))
    p.add_argument('--latent-cache-root', type=Path, default=Path('data/clips'), help='source of the PAI reference K/E calibration (for --treat-as-nv / rig reconstruct)')
    p.add_argument('--reference-cache', type=Path, default=None)
    p.add_argument('--caption-cache', type=Path, default=None, help='dir with per_uuid/<uuid>.txt one-sentence captions (Qwen-derived); UMT5-encoded here as the cond text so CFG engages')
    p.add_argument('--caption', type=str, default=None, help='override caption applied to ALL clips (instead of --caption-cache)')
    p.add_argument('--cross-guide', type=float, default=3.5)
    p.add_argument('--rear-guide', type=float, default=7.0)
    p.add_argument('--num-steps', type=int, default=50)
    p.add_argument('--start-sigma', type=float, default=1.0)
    p.add_argument('--shared-noise-alpha', type=float, default=0.5)
    p.add_argument('--splat-radius', type=int, default=1)
    p.add_argument('--max-clips', type=int, default=-1)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--seed', type=int, default=20260710)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--treat-as-nv', action='store_true', help='override K/E with the PAI reference rig; required for cross-camera dashcam sources (Nexar/Waymo). Matches the baseline dashcam harness.')
    return p.parse_args()

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
    scheduler = FlowMatchScheduler(shift=config.model.sample_shift)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    null_e = null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    null_m = null_mask.unsqueeze(0).to(device=device)
    caption_encoder = _RuntimeUMT5Encoder(args.device)
    root_summary: dict[str, Any] = {'arm': 'D', 'test_data_root': str(args.test_data_root), 'output_dir': str(output_dir), 'checkpoint_dir': str(args.checkpoint_dir), 'reference_cache': str(reference_cache), 'warp': 'live(same-time cross + lookback rear, visibility-max merge)', 'cross_guide': args.cross_guide, 'rear_guide': args.rear_guide, 'target_views': list(ALL_TARGET_VIEWS), 'num_clips': len(clips), 'load_info': load_info, 'clips': []}
    (output_dir / 'config_used.json').write_text(json.dumps(_jsonable(vars(args)), indent=2, sort_keys=True))
    for (idx, clip_dir) in enumerate(clips, start=1):
        rel_key = clip_dir.relative_to(args.test_data_root)
        clip_out = output_dir / rel_key
        done_path = clip_out / 'clip_done.json'
        if done_path.exists() and (not args.overwrite):
            root_summary['clips'].append({'clip': str(rel_key), 'status': 'skipped'})
            continue
        clip_out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f'[{idx}/{len(clips)}] {rel_key} warp=live', flush=True)
        (front, depth, pose, k_front, meta) = _load_clip_tensors(clip_dir)
        uuid = str(meta.get('uuid', ''))
        if args.treat_as_nv:
            k_all = reference_k.clone()
            e_all = reference_e.clone()
        else:
            (k_all, e_all) = reconstruct_e_all(k_front, meta['E_rig_front'].float(), reference_k=reference_k, reference_e=reference_e)
        k_all = k_all.to(device)
        e_all = e_all.to(device)
        pose = pose.to(device)
        (warp_provider, warp_summary) = build_live_warp_provider(front, depth, k_all, e_all, pose, vae, device, splat_radius=args.splat_radius)
        (caption, cap_src) = _resolve_caption(args.caption_cache, uuid, args.caption)
        if caption:
            (cap_emb, cap_mask) = caption_encoder.encode(caption)
            text_encoder = CachedTextEncoder(text_emb=cap_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=cap_mask.unsqueeze(0).to(device=device), null_emb=null_e, null_mask=null_m)
        else:
            print(f'  WARNING: no caption for uuid={uuid} (src={cap_src}); CFG rides on null (supply --caption-cache or --caption for the rear-view fix)', flush=True)
            text_encoder = CachedTextEncoder(text_emb=null_e.clone(), text_mask=null_m.clone(), null_emb=null_e, null_mask=null_m)
        bundle = InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=scheduler, warp_provider=warp_provider)
        z_front = _encode_front(vae, front.to(device=device)).to(device=device)
        batch_shape = z_front.unsqueeze(0).shape
        generator = torch.Generator(device=device).manual_seed(args.seed + idx + args.shard_index * 100000)
        shared = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
        generated: dict[int, torch.Tensor] = {}
        pred_latents: list[torch.Tensor] = []
        for view_id in ALL_TARGET_VIEWS:
            private = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
            alpha = float(args.shared_noise_alpha)
            target_noise = alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * private
            guide = _guide_for_view(view_id, args.cross_guide, args.rear_guide)
            latent = inference_single_target(z_front, k_all, e_all, pose, caption or str(rel_key), bundle, view_id, condition_latents_by_view=generated, target_noise=target_noise, num_steps=args.num_steps, guide_scale=guide, start_sigma=args.start_sigma, generator=generator)
            generated[view_id] = latent
            pred_latents.append(latent)
        pred = _decode_targets(vae, torch.stack(pred_latents, dim=0)).detach().cpu()
        pred_tchw = pred.permute(0, 2, 1, 3, 4).contiguous()
        fps = int(meta.get('src_fps', getattr(config.data, 'target_fps', 16)))
        outputs: dict[str, Any] = {'front_input': _write_mp4(clip_out / 'front_input.mp4', front, fps), 'pred': {}}
        rows = [front]
        for (li, vid) in enumerate(ALL_TARGET_VIEWS):
            outputs['pred'][str(vid)] = _write_mp4(clip_out / f'pred_{VIEW_NAMES[vid]}.mp4', pred_tchw[li], fps)
            rows.append(pred_tchw[li])
        outputs['pred_grid'] = _write_mp4(clip_out / 'pred_grid.mp4', torch.cat(rows, dim=-1), fps)
        cfg_active = bool(caption) and (not torch.equal(text_encoder.text_emb.float(), text_encoder.null_emb.float()))
        rear_cov = {str(v): float(warp_summary.get(str(v), {}).get('merged_vis_lat_pct', 0.0)) for v in REAR_VIEWS}
        clip_summary = {'clip': str(rel_key), 'status': 'ok', 'elapsed_sec': time.time() - started, 'uuid': uuid, 'caption': caption, 'caption_source': cap_src, 'cfg_active': cfg_active, 'cross_guide': args.cross_guide, 'rear_guide': args.rear_guide, 'rear_merged_vis_lat_pct': rear_cov, 'warp_summary': warp_summary, 'outputs': outputs}
        done_path.write_text(json.dumps(_jsonable(clip_summary), indent=2, sort_keys=True))
        root_summary['clips'].append(clip_summary)
        print(f'  caption[{cap_src}]={caption[:80]!r} cfg_active={cfg_active} rear_cov(3/4/5)={[round(rear_cov[str(v)], 2) for v in REAR_VIEWS]}%', flush=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (output_dir / f'summary_shard{args.shard_index}.json').write_text(json.dumps(_jsonable(root_summary), indent=2, sort_keys=True))
    print(f'DONE: {len(clips)} dashcam clips -> {output_dir}', flush=True)
if __name__ == '__main__':
    main()
