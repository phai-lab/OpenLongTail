from __future__ import annotations
import argparse
import json
import random
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any
import torch
from openlongtail.data.multiview_dataset import RayMultiViewDataConfig, RaySixCamDataset
from openlongtail.data.rig_parquet import SENSOR_ORDER
from openlongtail.data.text_emb_cache import load_text_embedding
from openlongtail.inference.teacher_p3 import P3InferenceBundle, encode_target_first_frame_latents, inference_p3, validate_p3_target_views
from openlongtail.models.dit_p3 import DiTP3
from openlongtail.models.wan22_backbone import load_wan22_expert
from openlongtail.models.wan22_lora import inject_lora_into_wan22_expert
from openlongtail.models.wan_vae import load_wan21_vae, precompute_blank_image_cond_latent
from openlongtail.training.checkpoint import load_checkpoint
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray import CONFIGS
VIEW_LABELS = ('front_wide', 'cross_left', 'cross_right', 'rear_left', 'rear_right', 'rear_tele')

class CachedTextEncoder:

    def __init__(self, text_emb: torch.Tensor, text_mask: torch.Tensor, null_emb: torch.Tensor, null_mask: torch.Tensor) -> None:
        self.text_emb = text_emb
        self.text_mask = text_mask
        self.null_emb = null_emb
        self.null_mask = null_mask

    def encode_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt
        return (self.text_emb, self.text_mask)

    def null_cached(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.null_emb, self.null_mask)

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

def _to_uint8(frames: torch.Tensor) -> torch.Tensor:
    x = frames.detach().float().cpu()
    if x.numel() == 0:
        raise ValueError('expected non-empty frame tensor')
    if float(x.max()) > 2.0:
        x = x.clamp(0.0, 255.0) / 255.0
    elif float(x.min()) < 0.0:
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        x = x.clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8)

def _draw_label(frame_chw: torch.Tensor, label: str) -> torch.Tensor:
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame_chw.permute(1, 2, 0).numpy())
    draw = ImageDraw.Draw(image)
    pad = 5
    bbox = draw.textbbox((pad, pad), label)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255))
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()

def _write_mp4(path: Path, video_tchw: torch.Tensor, fps: int) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _to_uint8(video_tchw).permute(0, 2, 3, 1).numpy()
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {'path': str(path), 'frames': int(frames.shape[0]), 'format': 'mp4'}

def _make_grid_frame(front: torch.Tensor, gt_targets: torch.Tensor, pred_targets: torch.Tensor, target_views: list[int]) -> torch.Tensor:
    front_u8 = _draw_label(_to_uint8(front), 'FRONT input')
    top = torch.cat([front_u8, front_u8], dim=2)
    rows = [top]
    gt_u8 = _to_uint8(gt_targets)
    pred_u8 = _to_uint8(pred_targets)
    for (local_idx, view_id) in enumerate(target_views):
        view_name = VIEW_LABELS[view_id]
        rows.append(torch.cat([_draw_label(gt_u8[local_idx], f'GT {view_id} {view_name}'), _draw_label(pred_u8[local_idx], f'PRED {view_id} {view_name}')], dim=2))
    return torch.cat(rows, dim=1)

def _make_comparison_grid(front: torch.Tensor, gt_targets: torch.Tensor, pred_targets_tchw: torch.Tensor, target_views: list[int]) -> torch.Tensor:
    if front.ndim != 4 or gt_targets.ndim != 5 or pred_targets_tchw.ndim != 5:
        raise ValueError(f'expected front (T,3,H,W), gt (3,T,3,H,W), pred (3,T,3,H,W), got {tuple(front.shape)}, {tuple(gt_targets.shape)}, {tuple(pred_targets_tchw.shape)}')
    frames = [_make_grid_frame(front[frame_idx], gt_targets[:, frame_idx], pred_targets_tchw[:, frame_idx], target_views) for frame_idx in range(front.shape[0])]
    return torch.stack(frames, dim=0)

def front_motion_score(front_rgb: torch.Tensor) -> float:
    if front_rgb.ndim != 4:
        raise ValueError(f'expected front_rgb shape (T, C, H, W), got {tuple(front_rgb.shape)}')
    if front_rgb.shape[0] < 2:
        return 0.0
    return float((front_rgb[1:].float() - front_rgb[:-1].float()).abs().mean().item())

def _build_dataset(config_name: str) -> RaySixCamDataset:
    config = CONFIGS[config_name]
    data_cfg = replace(config.data, text_drop_prob=0.0, max_items=None, num_workers=0)
    return RaySixCamDataset(RayMultiViewDataConfig(data_root=data_cfg.data_root, text_emb_cache_root=data_cfg.text_emb_cache_root, uuid_allowlist_json=data_cfg.uuid_allowlist_json, clip_length=data_cfg.clip_length, output_size=data_cfg.output_size, target_fps=data_cfg.target_fps, clip_anchor_seconds=data_cfg.clip_anchor_seconds, clip_jitter_seconds=data_cfg.clip_jitter_seconds, use_undistorted_simplecalib=data_cfg.use_undistorted_simplecalib, use_offline_extrinsics=data_cfg.use_offline_extrinsics, text_drop_prob=data_cfg.text_drop_prob, max_items=data_cfg.max_items, num_workers=0))

def _dataset_index_for_uuid(dataset: RaySixCamDataset, uuid: str) -> int:
    for (idx, uuid_dir) in enumerate(dataset.uuid_dirs):
        if uuid_dir.name == uuid:
            return idx
    raise KeyError(f'uuid {uuid!r} was not found in dataset')

def _load_one_sample(dataset: RaySixCamDataset, uuid: str | None, max_scan: int=20) -> tuple[int, dict[str, Any], float]:
    if uuid is not None:
        idx = _dataset_index_for_uuid(dataset, uuid)
        sample = dataset[idx]
        return (idx, sample, front_motion_score(sample['rgb'][0]))
    limit = min(len(dataset), max_scan)
    best: tuple[int, dict[str, Any], float] | None = None
    for idx in range(limit):
        sample = dataset[idx]
        motion = front_motion_score(sample['rgb'][0])
        if best is None or motion > best[2]:
            best = (idx, sample, motion)
        if motion > 0.01:
            return (idx, sample, motion)
    if best is None:
        raise FileNotFoundError('dataset is empty')
    return best

def _p3_shared_modules(dit: DiTP3) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({'plucker_mlp': dit.plucker_mlp, 'cam_id_embed': dit.cam_id_embed, 'role_embed': dit.role_embed, 'cross_view': dit.cross_view})

def _build_p3_dit(expert_dir: Path, checkpoint_state: dict[str, torch.Tensor] | None, shared_state: dict[str, torch.Tensor] | None, config_name: str, device: torch.device) -> tuple[DiTP3, dict[str, Any]]:
    config = CONFIGS[config_name]
    expert = load_wan22_expert(expert_dir, dtype=torch.bfloat16, device=device)
    inject_lora_into_wan22_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    blank = precompute_blank_image_cond_latent(vae, output_size=config.data.output_size, clip_length=config.data.clip_length, device=device, dtype=torch.bfloat16)
    if hasattr(vae, 'model'):
        vae.model.to('cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    dit = DiTP3(expert, blank_image_cond_latent=blank, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks).to(device=device, dtype=torch.bfloat16)
    if checkpoint_state is None:
        missing: list[str] = []
        unexpected: list[str] = []
        lora_source = 'frozen_base'
    else:
        (missing_keys, unexpected_keys) = dit.load_state_dict(checkpoint_state, strict=False)
        missing = list(missing_keys)
        unexpected = list(unexpected_keys)
        lora_source = 'checkpoint_lora'
    shared_missing: list[str] = []
    shared_unexpected: list[str] = []
    if shared_state is not None:
        (shared_missing, shared_unexpected) = _p3_shared_modules(dit).load_state_dict(shared_state, strict=False)
    dit.eval().requires_grad_(False)
    return (dit, {'lora_source': lora_source, 'missing_lora_keys': missing, 'unexpected_lora_keys': unexpected, 'missing_shared_keys': list(shared_missing), 'unexpected_shared_keys': list(shared_unexpected)})

def _load_bundle(checkpoint_dir: Path, config_name: str, sample: dict[str, Any], device: torch.device, *, use_frozen_high_base: bool=False) -> tuple[P3InferenceBundle, dict[str, Any]]:
    config = CONFIGS[config_name]
    payload = load_checkpoint(checkpoint_dir)
    if 'low_lora' not in payload:
        raise FileNotFoundError(f'expected low_lora.pt in checkpoint: {checkpoint_dir}')
    (low_dit, low_load_info) = _build_p3_dit(config.checkpoints.wan22_low_dir, payload['low_lora'], payload.get('shared_modules'), config_name, device)
    high_dit = None
    high_load_info: dict[str, Any] | None = None
    high_source = 'none'
    if 'high_lora' in payload:
        (high_dit, high_load_info) = _build_p3_dit(config.checkpoints.wan22_high_dir, payload['high_lora'], payload.get('shared_modules'), config_name, device)
        high_source = 'checkpoint_lora'
    elif use_frozen_high_base:
        (high_dit, high_load_info) = _build_p3_dit(config.checkpoints.wan22_high_dir, None, payload.get('shared_modules'), config_name, device)
        high_source = 'frozen_high_base'
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    text_encoder = CachedTextEncoder(text_emb=sample['text_emb'].unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=sample['text_mask'].unsqueeze(0).to(device=device), null_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), null_mask=null_mask.unsqueeze(0).to(device=device))
    bundle = P3InferenceBundle(vae=vae, low_dit=low_dit, high_dit=high_dit, text_encoder=text_encoder, scheduler=FlowMatchScheduler())
    return (bundle, {'checkpoint_metadata': payload.get('metadata', {}), 'has_low_lora': True, 'has_high_lora': 'high_lora' in payload, 'has_shared_modules': 'shared_modules' in payload, 'use_frozen_high_base_requested': use_frozen_high_base, 'high_source': high_source, 'low_load_info': low_load_info, 'high_load_info': high_load_info})

def _ensure_no_requested_outputs(output_dir: Path, target_views: list[int]) -> None:
    requested = [output_dir / 'front_input.mp4', output_dir / 'comparison_grid.mp4', output_dir / 'config_used.json', output_dir / 'summary.json', *[output_dir / f'gt_view_{view_id}.mp4' for view_id in target_views], *[output_dir / f'pred_view_{view_id}.mp4' for view_id in target_views]]
    existing = [str(path) for path in requested if path.exists()]
    if existing:
        raise FileExistsError('refusing to overwrite existing smoke outputs: ' + ', '.join(existing))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), default='latent_cache_vdrop')
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--uuid', type=str, default=None)
    parser.add_argument('--target-views', nargs=3, type=int, default=[1, 3, 5])
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--start-sigma', type=float, default=0.9)
    parser.add_argument('--guide-scale', type=float, default=3.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=20260429)
    parser.add_argument('--use-frozen-high-base', action='store_true', help='Use the original frozen high expert for sigma above the MoE boundary when no high_lora checkpoint is present.')
    parser.add_argument('--condition-target-first-frame', action='store_true', help="Clamp each target view's first latent frame to that view's GT first-frame condition during denoising.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    target_views = validate_p3_target_views(args.target_views)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_no_requested_outputs(output_dir, target_views)
    device = torch.device(args.device)
    dataset = _build_dataset(args.config)
    (dataset_idx, sample, motion) = _load_one_sample(dataset, args.uuid)
    uuid = str(sample['uuid'])
    config = CONFIGS[args.config]
    config_used = {'args': _jsonable(vars(args)), 'config_name': args.config, 'config': _jsonable(config), 'selected_dataset_idx': dataset_idx, 'selected_uuid': uuid, 'target_views': target_views, 'sensors': SENSOR_ORDER}
    (output_dir / 'config_used.json').write_text(json.dumps(config_used, indent=2, sort_keys=True))
    (bundle, load_info) = _load_bundle(args.checkpoint_dir, args.config, sample, device, use_frozen_high_base=args.use_frozen_high_base)
    front = sample['rgb'][0]
    gt_targets = sample['rgb'][target_views]
    K_all = sample['K'].to(device=device)
    E_all = sample['E'].to(device=device)
    target_first_latents = encode_target_first_frame_latents(bundle.vae, gt_targets.to(device=device)) if args.condition_target_first_frame else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    pred = inference_p3(front.to(device), K_all, E_all, uuid, target_views, bundle, num_steps=args.num_steps, guide_scale=args.guide_scale, start_sigma=args.start_sigma, high_switch_sigma=config.model.moe_boundary_sigma, use_high_expert=bundle.high_dit is not None, target_first_latents=target_first_latents)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
    pred_tchw = pred.detach().cpu().permute(0, 2, 1, 3, 4).contiguous()
    fps = config.data.target_fps
    video_outputs: dict[str, Any] = {'front_input': _write_mp4(output_dir / 'front_input.mp4', front, fps), 'gt': {}, 'pred': {}}
    for (local_idx, view_id) in enumerate(target_views):
        video_outputs['gt'][str(view_id)] = _write_mp4(output_dir / f'gt_view_{view_id}.mp4', gt_targets[local_idx], fps)
        video_outputs['pred'][str(view_id)] = _write_mp4(output_dir / f'pred_view_{view_id}.mp4', pred_tchw[local_idx], fps)
    grid = _make_comparison_grid(front, gt_targets, pred_tchw, target_views)
    video_outputs['comparison_grid'] = _write_mp4(output_dir / 'comparison_grid.mp4', grid, fps)
    summary = {'uuid': uuid, 'dataset_idx': dataset_idx, 'checkpoint_dir': str(args.checkpoint_dir), 'config': args.config, 'target_views': target_views, 'stream_view_ids': [0, *target_views], 'stream_role_ids': [0, 1, 1, 1], 'num_steps': args.num_steps, 'start_sigma': args.start_sigma, 'guide_scale': args.guide_scale, 'condition_target_first_frame': args.condition_target_first_frame, 'target_first_latent_condition_shape': list(target_first_latents.shape) if target_first_latents is not None else None, 'use_frozen_high_base': args.use_frozen_high_base, 'high_p3_checkpoint_available': load_info['has_high_lora'], 'high_p3_checkpoint_missing': not load_info['has_high_lora'], 'high_p3_expert_available': bundle.high_dit is not None, 'high_p3_frozen_base_used': load_info['high_source'] == 'frozen_high_base', 'full_sigma_mode': load_info['high_source'] if args.start_sigma >= 1.0 else 'low_sigma_only', 'full_sigma_quality_validated': bool(bundle.high_dit is not None and args.start_sigma >= 1.0), 'front_motion_score': motion, 'front_preserved': bundle.last_front_preserved, 'pred_shape': list(pred.shape), 'elapsed_sec': elapsed_sec, 'gpu_peak_memory_gb': peak_gb, 'load_info': load_info, 'outputs': video_outputs}
    if bundle.high_dit is None:
        summary['warning'] = 'low-only inference was used; full sigma=1.0 quality is not validated'
    elif load_info['high_source'] == 'frozen_high_base':
        summary['warning'] = 'frozen high-base baseline was used above the MoE boundary; trained Stage B high_lora quality is not validated'
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
