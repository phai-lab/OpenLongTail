from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
import torch
from openlongtail.configs.cfg_p61_vace import P61_CONDITION_ENCODER_LAYERS, P61_SEMANTIC_QUERIES, P61_SYNC_TEMPORAL_WINDOW
from openlongtail.configs.openlongtail_vace import GEO_HEAD_DIM, GEO_PROJECTION_TEMPERATURE, GRAPH_GATE_INIT_BIAS, OPENLONGTAIL_1P3B_CONFIG, OPENLONGTAIL_BASE_CONFIG
from openlongtail.data.rig_parquet import SENSOR_ORDER
from openlongtail.data.text_emb_cache import load_text_embedding
from openlongtail.inference.teacher_warp import InferenceBundleWarp, inference_multiview, make_sidecar_warp_provider, make_zero_warp_provider
from openlongtail.models.dit_vace import DiTVACEWarp
from openlongtail.models.wan21_vace_backbone import default_wan21_vace_dir, load_wan21_vace_expert
from openlongtail.models.wan21_vace_lora import inject_lora_into_wan21_vace_expert
from openlongtail.models.wan_vae import load_wan21_vae
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _load_one_sample, _make_comparison_grid, _write_mp4
from openlongtail.scripts.inference_p61_smoke import TARGET_VIEWS, _build_dataset, _front_pose_summary
from openlongtail.training.checkpoint_warp import load_warp_checkpoint
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train import ray_shared_modules
CONFIGS = {'openlongtail_1p3b': OPENLONGTAIL_1P3B_CONFIG, 'openlongtail_base': OPENLONGTAIL_BASE_CONFIG}

def _checkpoint_ready(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.is_dir() and all(((checkpoint_dir / name).exists() for name in ('metadata.json', 'lora.pt', 'shared_modules.pt')))

def _step_from_checkpoint_dir(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.name.startswith('step_'):
        return -1
    try:
        return int(checkpoint_dir.name[len('step_'):])
    except ValueError:
        return -1

def _latest_ready_checkpoint(checkpoint_root: Path, min_step: int=0) -> Path | None:
    if not checkpoint_root.exists():
        return None
    candidates = []
    for path in checkpoint_root.glob('step_*'):
        step = _step_from_checkpoint_dir(path)
        if step >= int(min_step) and _checkpoint_ready(path):
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]

def wait_for_latest_checkpoint(checkpoint_root: Path, min_step: int, timeout_sec: int, poll_sec: int=30) -> Path:
    deadline = time.time() + float(timeout_sec)
    while True:
        ckpt = _latest_ready_checkpoint(checkpoint_root, min_step=min_step)
        if ckpt is not None:
            return ckpt
        if time.time() >= deadline:
            raise TimeoutError(f'checkpoint >= step {min_step} did not become ready under {checkpoint_root}')
        time.sleep(poll_sec)

def _build_dit(config_name: str, device: torch.device, wan21_vace_dir: Path | None, checkpoint_dir: Path) -> tuple[DiTVACEWarp, dict[str, Any]]:
    config = CONFIGS[config_name]
    is_fulltune = (Path(checkpoint_dir) / 'backbone.pt').exists()
    expert = load_wan21_vace_expert(wan21_vace_dir or default_wan21_vace_dir(), dtype=torch.bfloat16, device=device)
    if not is_fulltune:
        inject_lora_into_wan21_vace_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    dit = DiTVACEWarp(expert, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks, sync_temporal_window=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers=P61_CONDITION_ENCODER_LAYERS, semantic_queries=P61_SEMANTIC_QUERIES, graph_gate_init_bias=GRAPH_GATE_INIT_BIAS, geo_head_dim=GEO_HEAD_DIM, geo_projection_temperature=GEO_PROJECTION_TEMPERATURE).to(device=device, dtype=torch.bfloat16)
    payload = load_warp_checkpoint(checkpoint_dir)
    missing: list[str] = []
    unexpected: list[str] = []
    if is_fulltune:
        if 'backbone' not in payload:
            raise FileNotFoundError(f'expected backbone.pt in fulltune ckpt: {checkpoint_dir}')
        (missing, unexpected) = dit.load_state_dict(payload['backbone'], strict=False)
    else:
        if 'lora' not in payload:
            raise FileNotFoundError(f'expected lora.pt in checkpoint: {checkpoint_dir}')
        (missing, unexpected) = dit.load_state_dict(payload['lora'], strict=False)
    s_missing: list[str] = []
    s_unexpected: list[str] = []
    if 'shared_modules' in payload:
        (s_missing, s_unexpected) = ray_shared_modules(dit).load_state_dict(payload['shared_modules'], strict=False)
    dit.eval().requires_grad_(False)
    return (dit, {'mode': 'checkpoint', 'is_fulltune': bool(is_fulltune), 'checkpoint_dir': str(checkpoint_dir), 'checkpoint_metadata': payload.get('metadata', {}), 'missing_keys': list(missing), 'unexpected_keys': list(unexpected), 'missing_shared_keys': list(s_missing), 'unexpected_shared_keys': list(s_unexpected)})

def _resolve_sidecar_paths(uuid: str, clip_id: int, latent_cache_root: Path) -> tuple[Path, Path]:
    base = latent_cache_root / 'per_uuid' / uuid
    return (base / f'clip_{clip_id:06d}_warp.pt', base / f'clip_{clip_id:06d}_lookback.pt')

def _load_bundle(config_name: str, sample: dict[str, Any], device: torch.device, wan21_vace_dir: Path | None, checkpoint_dir: Path, use_sidecar: bool, latent_cache_root: Path, latent_shape: tuple[int, int, int]) -> tuple[InferenceBundleWarp, dict[str, Any]]:
    config = CONFIGS[config_name]
    (dit, load_info) = _build_dit(config_name, device, wan21_vace_dir, checkpoint_dir)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    text_encoder = CachedTextEncoder(text_emb=sample['text_emb'].unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=sample['text_mask'].unsqueeze(0).to(device=device), null_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), null_mask=null_mask.unsqueeze(0).to(device=device))
    if use_sidecar:
        uuid = str(sample['uuid'])
        clip_id = int(sample.get('clip_id', 0))
        (v0_path, v1_path) = _resolve_sidecar_paths(uuid, clip_id, latent_cache_root)
        if not v0_path.exists():
            raise FileNotFoundError(f'warp sidecar missing: {v0_path}')
        v1_path_or_none = v1_path if v1_path.exists() else None
        warp_provider = make_sidecar_warp_provider(str(v0_path), str(v1_path_or_none) if v1_path_or_none else None, device=device)
        load_info['warp_source'] = 'sidecar'
        load_info['sidecar_v0'] = str(v0_path)
        load_info['sidecar_v1'] = str(v1_path_or_none) if v1_path_or_none else None
    else:
        warp_provider = make_zero_warp_provider(latent_shape, device=device)
        load_info['warp_source'] = 'zero'
    return (InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=FlowMatchScheduler(shift=config.model.sample_shift), warp_provider=warp_provider), load_info)

def _ensure_no_requested_outputs(output_dir: Path) -> None:
    requested = [output_dir / 'front_input.mp4', output_dir / 'comparison_grid.mp4', output_dir / 'config_used.json', output_dir / 'summary.json', *[output_dir / f'gt_view_{v}.mp4' for v in TARGET_VIEWS], *[output_dir / f'pred_view_{v}.mp4' for v in TARGET_VIEWS]]
    existing = [str(p) for p in requested if p.exists()]
    if existing:
        raise FileExistsError('refusing to overwrite existing smoke outputs: ' + ', '.join(existing))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=sorted(CONFIGS), default='openlongtail_1p3b')
    p.add_argument('--checkpoint-dir', type=Path, default=None)
    p.add_argument('--checkpoint-root', type=Path, default=None)
    p.add_argument('--min-checkpoint-step', type=int, default=0)
    p.add_argument('--wait-timeout-sec', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--wan21-vace-dir', type=Path, default=None)
    p.add_argument('--latent-cache-root', type=Path, default=Path('cache/latent_cache'))
    p.add_argument('--uuid', type=str, default=None)
    p.add_argument('--num-steps', type=int, default=40)
    p.add_argument('--start-sigma', type=float, default=1.0)
    p.add_argument('--guide-scale', type=float, default=3.5)
    p.add_argument('--shared-noise-alpha', type=float, default=0.5)
    p.add_argument('--no-sidecar', action='store_true', help='Use a zero warp_provider instead of loading sidecars (for sanity-only)')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=20260430)
    return p.parse_args()

def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        if args.wait_timeout_sec > 0:
            deadline = time.time() + float(args.wait_timeout_sec)
            while not _checkpoint_ready(args.checkpoint_dir):
                if time.time() >= deadline:
                    raise TimeoutError(f'checkpoint did not become ready: {args.checkpoint_dir}')
                time.sleep(30)
        if not _checkpoint_ready(args.checkpoint_dir):
            raise FileNotFoundError(f'checkpoint is not ready: {args.checkpoint_dir}')
        return args.checkpoint_dir
    if args.checkpoint_root is not None:
        timeout = int(args.wait_timeout_sec)
        if timeout > 0:
            return wait_for_latest_checkpoint(args.checkpoint_root, args.min_checkpoint_step, timeout)
        ckpt = _latest_ready_checkpoint(args.checkpoint_root, args.min_checkpoint_step)
        if ckpt is None:
            raise FileNotFoundError(f'no checkpoint >= step {args.min_checkpoint_step} under {args.checkpoint_root}')
        return ckpt
    raise ValueError('provide --checkpoint-dir or --checkpoint-root')

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = _resolve_checkpoint(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_no_requested_outputs(output_dir)
    device = torch.device(args.device)
    dataset = _build_dataset(args.config)
    (dataset_idx, sample, motion) = _load_one_sample(dataset, args.uuid)
    uuid = str(sample['uuid'])
    config = CONFIGS[args.config]
    (output_dir / 'config_used.json').write_text(json.dumps({'args': _jsonable(vars(args)), 'resolved_checkpoint_dir': str(checkpoint_dir), 'config_name': args.config, 'config': _jsonable(config), 'selected_dataset_idx': dataset_idx, 'selected_uuid': uuid, 'target_views': TARGET_VIEWS, 'sensors': SENSOR_ORDER}, indent=2, sort_keys=True))
    latent_shape = (11, sample['rgb'].shape[-2] // 8, sample['rgb'].shape[-1] // 8)
    (bundle, load_info) = _load_bundle(args.config, sample, device, args.wan21_vace_dir, checkpoint_dir, use_sidecar=not args.no_sidecar, latent_cache_root=args.latent_cache_root, latent_shape=latent_shape)
    front = sample['rgb'][0]
    gt_targets = sample['rgb'][TARGET_VIEWS]
    K_all = sample['K'].to(device=device)
    E_all = sample['E'].to(device=device)
    T_anchor_front = sample['T_anchor_front'].to(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    pred = inference_multiview(front.to(device), K_all, E_all, T_anchor_front, uuid, bundle, num_steps=args.num_steps, guide_scale=args.guide_scale, start_sigma=args.start_sigma, shared_noise_alpha=args.shared_noise_alpha, generator=generator)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
    pred_tchw = pred.detach().cpu().permute(0, 2, 1, 3, 4).contiguous()
    fps = config.data.target_fps
    video_outputs: dict[str, Any] = {'front_input': _write_mp4(output_dir / 'front_input.mp4', front, fps), 'gt': {}, 'pred': {}}
    for (local_idx, view_id) in enumerate(TARGET_VIEWS):
        video_outputs['gt'][str(view_id)] = _write_mp4(output_dir / f'gt_view_{view_id}.mp4', gt_targets[local_idx], fps)
        video_outputs['pred'][str(view_id)] = _write_mp4(output_dir / f'pred_view_{view_id}.mp4', pred_tchw[local_idx], fps)
    video_outputs['comparison_grid'] = _write_mp4(output_dir / 'comparison_grid.mp4', _make_comparison_grid(front, gt_targets, pred_tchw, TARGET_VIEWS), fps)
    summary = {'uuid': uuid, 'dataset_idx': dataset_idx, 'checkpoint_dir': str(checkpoint_dir), 'config': args.config, 'target_views': TARGET_VIEWS, 'trajectory_source': 'dataset', 'training_family': 'depth_warp_condition_v1', 'num_steps': args.num_steps, 'start_sigma': args.start_sigma, 'guide_scale': args.guide_scale, 'shared_noise_alpha': args.shared_noise_alpha, 'front_motion_score': motion, 'condition_preserved': bundle.last_condition_preserved, 'pred_shape': list(pred.shape), 'elapsed_sec': elapsed_sec, 'gpu_peak_memory_gb': peak_gb, 'load_info': load_info, 'outputs': video_outputs}
    summary.update(_front_pose_summary(sample['T_anchor_front']))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
