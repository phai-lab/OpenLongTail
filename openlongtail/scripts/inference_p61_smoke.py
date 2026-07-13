from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
import torch
from openlongtail.configs.cfg_p61_vace import P61_VACE_1P3B_CONFIG, P61_VACE_CONFIG
from openlongtail.data.multiview_dataset import RayMultiViewDataConfig, RaySixCamDataset
from openlongtail.data.rig_parquet import SENSOR_ORDER
from openlongtail.data.text_emb_cache import load_text_embedding
from openlongtail.data.transforms import se3_log_rotation_angle
from openlongtail.inference.teacher_p61 import P61InferenceBundle, inference_p61
from openlongtail.models.dit_p61_vace import DiTP61VACE
from openlongtail.models.wan21_vace_backbone import default_wan21_vace_dir, load_wan21_vace_expert
from openlongtail.models.wan21_vace_lora import inject_lora_into_wan21_vace_expert
from openlongtail.models.wan_vae import load_wan21_vae
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _load_one_sample, _make_comparison_grid, _write_mp4
from openlongtail.training.checkpoint_p61 import load_checkpoint_p61
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray_p61 import ray_shared_modules_p61
CONFIGS = {'p61_vace': P61_VACE_CONFIG, 'p61_vace_1p3b': P61_VACE_1P3B_CONFIG}
TARGET_VIEWS = [1, 2, 3, 4, 5]

def _build_dataset(config_name: str) -> RaySixCamDataset:
    config = CONFIGS[config_name]
    data_cfg = config.data
    return RaySixCamDataset(RayMultiViewDataConfig(data_root=data_cfg.data_root, text_emb_cache_root=data_cfg.text_emb_cache_root, uuid_allowlist_json=data_cfg.uuid_allowlist_json, clip_length=data_cfg.clip_length, output_size=data_cfg.output_size, target_fps=data_cfg.target_fps, clip_anchor_seconds=data_cfg.clip_anchor_seconds, clip_jitter_seconds=data_cfg.clip_jitter_seconds, use_undistorted_simplecalib=data_cfg.use_undistorted_simplecalib, use_offline_extrinsics=data_cfg.use_offline_extrinsics, text_drop_prob=0.0, max_items=None, num_workers=0, include_p4_front_pose=True))

def _checkpoint_ready(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.is_dir() and all(((checkpoint_dir / name).exists() for name in ('metadata.json', 'p61_lora.pt', 'shared_modules.pt')))

def _step_from_checkpoint_dir(checkpoint_dir: Path) -> int:
    prefix = 'step_'
    if not checkpoint_dir.name.startswith(prefix):
        return -1
    try:
        return int(checkpoint_dir.name[len(prefix):])
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
        checkpoint = _latest_ready_checkpoint(checkpoint_root, min_step=min_step)
        if checkpoint is not None:
            return checkpoint
        if time.time() >= deadline:
            raise TimeoutError(f'checkpoint >= step {min_step} did not become ready under {checkpoint_root}')
        time.sleep(poll_sec)

def _zero_vanilla_p61_conditioning(dit: DiTP61VACE) -> None:
    modules = [dit.plucker_mlp, dit.cam_id_embed, dit.role_embed, dit.condition_type_embed, dit.availability_embed, dit.trajectory_mlp, dit.condition_encoder, dit.semantic_resampler, dit.graph_memory]
    for module in modules:
        for param in module.parameters():
            param.data.zero_()

def _build_dit(config_name: str, device: torch.device, wan21_vace_dir: Path | None, checkpoint_dir: Path | None, vanilla_base: bool, zero_shot_conditioning: bool) -> tuple[DiTP61VACE, dict[str, Any]]:
    config = CONFIGS[config_name]
    expert = load_wan21_vace_expert(wan21_vace_dir or default_wan21_vace_dir(), dtype=torch.bfloat16, device=device)
    inject_lora_into_wan21_vace_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    dit = DiTP61VACE(expert, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks).to(device=device, dtype=torch.bfloat16)
    if vanilla_base:
        _zero_vanilla_p61_conditioning(dit)
        load_info: dict[str, Any] = {'mode': 'vanilla_base', 'checkpoint_dir': None, 'has_p61_lora': False, 'has_shared_modules': False, 'p61_conditioning_zeroed': True, 'front_vace_target_condition': True}
    elif zero_shot_conditioning:
        load_info = {'mode': 'zero_shot_conditioning', 'checkpoint_dir': None, 'has_p61_lora': False, 'has_shared_modules': False, 'p61_conditioning_zeroed': False, 'front_vace_target_condition': True}
    else:
        if checkpoint_dir is None:
            raise ValueError('checkpoint_dir is required unless --vanilla-base or --zero-shot-conditioning is set')
        payload = load_checkpoint_p61(checkpoint_dir)
        if 'p61_lora' not in payload:
            raise FileNotFoundError(f'expected p61_lora.pt in checkpoint: {checkpoint_dir}')
        (missing, unexpected) = dit.load_state_dict(payload['p61_lora'], strict=False)
        shared_missing: list[str] = []
        shared_unexpected: list[str] = []
        if 'shared_modules' in payload:
            (shared_missing, shared_unexpected) = ray_shared_modules_p61(dit).load_state_dict(payload['shared_modules'], strict=False)
        load_info = {'mode': 'checkpoint', 'checkpoint_dir': str(checkpoint_dir), 'checkpoint_metadata': payload.get('metadata', {}), 'has_p61_lora': True, 'has_shared_modules': 'shared_modules' in payload, 'missing_p61_lora_keys': list(missing), 'unexpected_p61_lora_keys': list(unexpected), 'missing_shared_keys': list(shared_missing), 'unexpected_shared_keys': list(shared_unexpected)}
    dit.eval().requires_grad_(False)
    return (dit, load_info)

def _load_bundle(config_name: str, sample: dict[str, Any], device: torch.device, wan21_vace_dir: Path | None, checkpoint_dir: Path | None, vanilla_base: bool, zero_shot_conditioning: bool) -> tuple[P61InferenceBundle, dict[str, Any]]:
    config = CONFIGS[config_name]
    (dit, load_info) = _build_dit(config_name, device, wan21_vace_dir, checkpoint_dir, vanilla_base, zero_shot_conditioning)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    text_encoder = CachedTextEncoder(text_emb=sample['text_emb'].unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=sample['text_mask'].unsqueeze(0).to(device=device), null_emb=null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), null_mask=null_mask.unsqueeze(0).to(device=device))
    return (P61InferenceBundle(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=FlowMatchScheduler(shift=config.model.sample_shift), front_vace_target_condition=bool(vanilla_base or zero_shot_conditioning)), load_info)

def _front_pose_summary(T_anchor_front: torch.Tensor) -> dict[str, float]:
    T = T_anchor_front.detach().float().cpu()
    trans = T[:, :3, 3].norm(dim=-1)
    angles = se3_log_rotation_angle(T)
    jumps = (T[1:, :3, 3] - T[:-1, :3, 3]).norm(dim=-1) if T.shape[0] > 1 else torch.zeros(1)
    return {'T_anchor_front_translation_norm_min': float(trans.min().item()), 'T_anchor_front_translation_norm_max': float(trans.max().item()), 'T_anchor_front_rotation_angle_min': float(angles.min().item()), 'T_anchor_front_rotation_angle_max': float(angles.max().item()), 'T_anchor_front_frame_to_frame_jump_max': float(jumps.max().item())}

def _ensure_no_requested_outputs(output_dir: Path) -> None:
    requested = [output_dir / 'front_input.mp4', output_dir / 'comparison_grid.mp4', output_dir / 'config_used.json', output_dir / 'summary.json', *[output_dir / f'gt_view_{view_id}.mp4' for view_id in TARGET_VIEWS], *[output_dir / f'pred_view_{view_id}.mp4' for view_id in TARGET_VIEWS]]
    existing = [str(path) for path in requested if path.exists()]
    if existing:
        raise FileExistsError('refusing to overwrite existing smoke outputs: ' + ', '.join(existing))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), default='p61_vace_1p3b')
    parser.add_argument('--checkpoint-dir', type=Path, default=None)
    parser.add_argument('--checkpoint-root', type=Path, default=None)
    parser.add_argument('--min-checkpoint-step', type=int, default=0)
    parser.add_argument('--wait-timeout-sec', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wan21-vace-dir', type=Path, default=None)
    parser.add_argument('--uuid', type=str, default=None)
    parser.add_argument('--trajectory-source', choices=('dataset', 'identity', 'planner'), default='dataset')
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--start-sigma', type=float, default=1.0)
    parser.add_argument('--guide-scale', type=float, default=3.5)
    parser.add_argument('--shared-noise-alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=20260430)
    parser.add_argument('--vanilla-base', action='store_true')
    parser.add_argument('--zero-shot-conditioning', action='store_true', help='Run base Wan2.1-VACE with randomly initialized conditioning modules enabled and no checkpoint.')
    return parser.parse_args()

def _resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.vanilla_base and args.zero_shot_conditioning:
        raise ValueError('--vanilla-base and --zero-shot-conditioning are mutually exclusive')
    if args.vanilla_base or args.zero_shot_conditioning:
        return None
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
        checkpoint = _latest_ready_checkpoint(args.checkpoint_root, args.min_checkpoint_step)
        if checkpoint is None:
            raise FileNotFoundError(f'no checkpoint >= step {args.min_checkpoint_step} under {args.checkpoint_root}')
        return checkpoint
    raise ValueError('provide --checkpoint-dir or --checkpoint-root, unless --vanilla-base is set')

def main() -> None:
    args = parse_args()
    if args.trajectory_source != 'dataset':
        raise NotImplementedError(f'trajectory source {args.trajectory_source!r} is not implemented')
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
    (output_dir / 'config_used.json').write_text(json.dumps({'args': _jsonable(vars(args)), 'resolved_checkpoint_dir': str(checkpoint_dir) if checkpoint_dir is not None else None, 'config_name': args.config, 'config': _jsonable(config), 'selected_dataset_idx': dataset_idx, 'selected_uuid': uuid, 'target_views': TARGET_VIEWS, 'sensors': SENSOR_ORDER}, indent=2, sort_keys=True))
    (bundle, load_info) = _load_bundle(args.config, sample, device, args.wan21_vace_dir, checkpoint_dir, args.vanilla_base, args.zero_shot_conditioning)
    front = sample['rgb'][0]
    gt_targets = sample['rgb'][TARGET_VIEWS]
    K_all = sample['K'].to(device=device)
    E_all = sample['E'].to(device=device)
    T_anchor_front = sample['T_anchor_front'].to(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    pred = inference_p61(front.to(device), K_all, E_all, T_anchor_front, uuid, bundle, num_steps=args.num_steps, guide_scale=args.guide_scale, start_sigma=args.start_sigma, shared_noise_alpha=args.shared_noise_alpha, generator=generator)
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
    summary = {'uuid': uuid, 'dataset_idx': dataset_idx, 'checkpoint_dir': str(checkpoint_dir) if checkpoint_dir is not None else None, 'config': args.config, 'target_views': TARGET_VIEWS, 'trajectory_source': args.trajectory_source, 'training_family': 'p61_vace_graph_autoregressive_single_target', 'vanilla_base': bool(args.vanilla_base), 'zero_shot_conditioning': bool(args.zero_shot_conditioning), 'num_steps': args.num_steps, 'start_sigma': args.start_sigma, 'guide_scale': args.guide_scale, 'shared_noise_alpha': args.shared_noise_alpha, 'front_motion_score': motion, 'condition_preserved': bundle.last_condition_preserved, 'pred_shape': list(pred.shape), 'elapsed_sec': elapsed_sec, 'gpu_peak_memory_gb': peak_gb, 'load_info': load_info, 'outputs': video_outputs}
    summary.update(_front_pose_summary(sample['T_anchor_front']))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
