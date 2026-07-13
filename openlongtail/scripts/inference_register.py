from __future__ import annotations
import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from openlongtail.data.depth_warp import forward_splat_warp, se3_inverse_torch
from openlongtail.inference.teacher_p3 import _decode_targets, _encode_front, _ensure_batched_mask, _ensure_batched_text, sigmas_descending_from_start
from openlongtail.inference.teacher_p61 import _reset_condition_inputs, _resolve_geometry_override, build_p61_inference_inputs
from openlongtail.inference.teacher_warp import InferenceBundleWarp, make_sidecar_warp_provider
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX
from openlongtail.scripts.build_lookback_warp import LOOKBACK_OFFSET, latent_lf_to_rgb_repr, rgb_t_to_latent_lf
from openlongtail.scripts.build_warp import FRONT_IDX, SENSOR_NAMES, downsample_visibility_to_latent, load_depth_clip, load_front_rgb_clip, vae_encode_warped
from openlongtail.scripts.inference_p3_smoke import CachedTextEncoder, _jsonable, _make_comparison_grid, _write_mp4
from openlongtail.training.forward_ray_p61 import build_p61_anchor_camera_transforms, build_p61_dynamic_plucker_streams
from openlongtail.training.forward_ray_p62 import build_p62_target_pose_features, build_p62_time_window_relative_pose_features
from openlongtail.training.schedulers import FlowMatchScheduler
ALL_TARGET_VIEWS: tuple[int, ...] = (1, 2, 3, 4, 5)
CROSS_VIEWS: tuple[int, ...] = (1, 2)
REAR_VIEWS: tuple[int, ...] = (3, 4, 5)
VIEW_NAMES: dict[int, str] = {1: 'cross_left', 2: 'cross_right', 3: 'rear_left', 4: 'rear_right', 5: 'rear_tele'}
NUM_STYLE_AXES: int = 3
FRONT_SENSOR: str = SENSOR_NAMES[FRONT_IDX]

def _config_unknown_style_ids() -> tuple[int, ...]:
    try:
        from openlongtail.configs.openlongtail_style_vace import STYLE_UNKNOWN_INDICES
        return tuple((int(i) for i in STYLE_UNKNOWN_INDICES))
    except Exception:
        return (5, 3, 4)

def resolve_unknown_style_ids(dit: torch.nn.Module | None, num_axes: int=NUM_STYLE_AXES) -> tuple[int, ...]:
    conditioner = getattr(dit, 'style_conditioner', None) if dit is not None else None
    if conditioner is not None:
        ui = getattr(conditioner, 'unknown_indices', None)
        if ui is not None and len(ui) == num_axes:
            return tuple((int(i) for i in ui))
    ui = _config_unknown_style_ids()
    if len(ui) != num_axes:
        raise ValueError(f'unknown-index count {len(ui)} != num_axes {num_axes}')
    return ui

def load_style_ids_for_uuid(style_cache_root: Path | None, uuid: str, unknown_ids: tuple[int, ...], num_axes: int=NUM_STYLE_AXES) -> tuple[torch.Tensor, str]:
    from openlongtail.data.style_code_mixin import _coerce_style_ids
    unknown = torch.tensor(unknown_ids, dtype=torch.long)
    if style_cache_root is None or not uuid:
        return (unknown, 'unknown(no-cache)')
    path = Path(style_cache_root) / 'per_uuid' / f'{uuid}.pt'
    if not path.exists():
        return (unknown, 'unknown(missing)')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    ids = _coerce_style_ids(payload, num_axes)
    if ids is None:
        return (unknown, 'unknown(malformed)')
    return (ids.to(torch.long), 'cache')

def _warp_same_time(front_rgb: torch.Tensor, front_depth: torch.Tensor, k_all: torch.Tensor, e_all: torch.Tensor, view_id: int, out_h: int, out_w: int, splat_radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    k = k_all.to(device=device, dtype=torch.float32)
    e = e_all.to(device=device, dtype=torch.float32)
    t_rel = se3_inverse_torch(e[view_id].unsqueeze(0)).squeeze(0) @ e[FRONT_IDX]
    return forward_splat_warp(front_rgb.to(device), front_depth.to(device), k[FRONT_IDX], k[view_id], t_rel, out_h=out_h, out_w=out_w, splat_radius=splat_radius)

def _warp_lookback(front_rgb: torch.Tensor, front_depth: torch.Tensor, k_all: torch.Tensor, T_anchor_cam: torch.Tensor, view_id: int, k_off: int, out_h: int, out_w: int, splat_radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    k = k_all.to(device=device, dtype=torch.float32)
    T_cam = T_anchor_cam.to(device=device, dtype=torch.float32)
    front = front_rgb.to(device)
    depth = front_depth.to(device)
    warped_frames: list[torch.Tensor] = []
    vis_frames: list[torch.Tensor] = []
    for rgb_t in range(front.shape[0]):
        target_lf = rgb_t_to_latent_lf(rgb_t)
        source_lf = max(0, target_lf - int(k_off))
        src_local = latent_lf_to_rgb_repr(source_lf)
        src_local = min(src_local, front.shape[0] - 1)
        T_src = T_cam[FRONT_IDX, source_lf]
        T_tgt = T_cam[view_id, target_lf]
        T_src_to_tgt = se3_inverse_torch(T_tgt.unsqueeze(0)).squeeze(0) @ T_src
        (warped, vis) = forward_splat_warp(front[src_local:src_local + 1], depth[src_local:src_local + 1], k[FRONT_IDX], k[view_id], T_src_to_tgt, out_h=out_h, out_w=out_w, splat_radius=splat_radius)
        warped_frames.append(warped[0])
        vis_frames.append(vis[0])
    return (torch.stack(warped_frames, dim=0), torch.stack(vis_frames, dim=0))

def build_live_warp_provider(front_rgb: torch.Tensor, front_depth: torch.Tensor, k_all: torch.Tensor, e_all: torch.Tensor, T_anchor_front: torch.Tensor, vae: object | None, device: torch.device, splat_radius: int=1, cross_views: tuple[int, ...]=CROSS_VIEWS, rear_views: tuple[int, ...]=REAR_VIEWS, lookback_offset: dict[int, int] | None=None) -> tuple[Callable[[int], tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    lookback_offset = dict(LOOKBACK_OFFSET if lookback_offset is None else lookback_offset)
    out_h = int(front_rgb.shape[-2])
    out_w = int(front_rgb.shape[-1])
    (h_lat, w_lat) = (out_h // 8, out_w // 8)
    T_anchor_cam = build_p61_anchor_camera_transforms(e_all.to(device).float().unsqueeze(0), T_anchor_front.to(device).float().unsqueeze(0)).squeeze(0)
    warped_by_view: dict[int, torch.Tensor] = {}
    vis_by_view: dict[int, torch.Tensor] = {}
    summary: dict[str, Any] = {}

    def _vis_lat(vis_pix: torch.Tensor) -> torch.Tensor:
        return downsample_visibility_to_latent(vis_pix.cpu().float(), t_lat=11, h_lat=h_lat, w_lat=w_lat)

    def _encode(warped_rgb: torch.Tensor) -> torch.Tensor:
        if vae is None:
            return torch.zeros(16, 11, h_lat, w_lat, dtype=torch.bfloat16)
        return vae_encode_warped(vae, warped_rgb.to(torch.uint8), device=device, dtype=torch.bfloat16)
    for view_id in ALL_TARGET_VIEWS:
        (st_rgb, st_vis) = _warp_same_time(front_rgb, front_depth, k_all, e_all, view_id, out_h, out_w, splat_radius, device)
        st_vis_lat = _vis_lat(st_vis)
        st_z = _encode(st_rgb)
        view_summary: dict[str, Any] = {'view_name': VIEW_NAMES[view_id], 'same_time_vis_pix_pct': float(st_vis.float().mean().item() * 100.0), 'same_time_vis_lat_pct': float(st_vis_lat.float().mean().item() * 100.0)}
        if view_id in rear_views:
            k_off = int(lookback_offset.get(view_id, 0))
            (lb_rgb, lb_vis) = _warp_lookback(front_rgb, front_depth, k_all, T_anchor_cam, view_id, k_off, out_h, out_w, splat_radius, device)
            lb_vis_lat = _vis_lat(lb_vis)
            lb_z = _encode(lb_rgb)
            use_lb = lb_vis_lat.float() > st_vis_lat.float()
            merged_z = torch.where(use_lb.expand_as(st_z), lb_z, st_z)
            merged_vis = torch.maximum(st_vis_lat.float(), lb_vis_lat.float()).to(torch.float16)
            view_summary.update(lookback_offset=k_off, lookback_vis_pix_pct=float(lb_vis.float().mean().item() * 100.0), lookback_vis_lat_pct=float(lb_vis_lat.float().mean().item() * 100.0), merged_vis_lat_pct=float(merged_vis.float().mean().item() * 100.0), merge_uses_lookback_pct=float(use_lb.float().mean().item() * 100.0))
            warped_by_view[view_id] = merged_z.to(device=device, dtype=torch.bfloat16)
            vis_by_view[view_id] = merged_vis.to(device=device)
        else:
            view_summary['merged_vis_lat_pct'] = view_summary['same_time_vis_lat_pct']
            warped_by_view[view_id] = st_z.to(device=device, dtype=torch.bfloat16)
            vis_by_view[view_id] = st_vis_lat.to(device=device, dtype=torch.float16)
        summary[str(view_id)] = view_summary

    def provider(target_view: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = int(target_view)
        z = warped_by_view[v].unsqueeze(0)
        vis = vis_by_view[v].unsqueeze(0)
        return (z, vis)
    return (provider, summary)

def _forward_param_names(dit: torch.nn.Module) -> set[str]:
    try:
        sig = inspect.signature(dit.forward)
    except (TypeError, ValueError):
        return set()
    return {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

def _dit_forward(dit: torch.nn.Module, canonical: dict[str, Any], style_ids: torch.Tensor | None) -> torch.Tensor:
    params = _forward_param_names(dit)
    if not params:
        out = dit(canonical['z_backbone'], canonical['sigma'], canonical['text_emb'], canonical['text_mask'], canonical['backbone_plucker'], canonical['backbone_view_ids'], canonical['backbone_role_ids'], canonical['T_anchor_front'], canonical['condition_latents'], canonical['condition_plucker'], canonical['condition_view_ids'], canonical['condition_type_ids'], canonical['condition_available_mask'], canonical['relative_pose_features'], canonical['target_pose_features'], warped_target_latent=canonical['warped_target_latent'], warped_target_visibility=canonical['warped_target_visibility'], **{'style_ids': style_ids} if style_ids is not None else {})
        return out
    call: dict[str, Any] = {}
    for (key, value) in canonical.items():
        if key in params:
            call[key] = value
        elif key == 'text_emb' and 'caption_emb' in params:
            call['caption_emb'] = value
        elif key == 'text_mask' and 'caption_mask' in params:
            call['caption_mask'] = value
    if style_ids is not None and 'style_ids' in params:
        call['style_ids'] = style_ids
    return dit(**call)

@torch.no_grad()
def inference_register_single_target(z_front: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, T_anchor_front: torch.Tensor, text_prompt: str, bundle: InferenceBundleWarp, target_view: int, cond_style_ids: torch.Tensor, uncond_style_ids: torch.Tensor, guide_scale: float, condition_latents_by_view: dict[int, torch.Tensor] | None=None, target_noise: torch.Tensor | None=None, num_steps: int=50, start_sigma: float=1.0, generator: torch.Generator | None=None, geometry_mode: str='correct') -> torch.Tensor:
    device = z_front.device
    inputs = build_p61_inference_inputs(z_front, target_view, condition_latents_by_view=condition_latents_by_view, target_noise=target_noise, generator=generator)
    z_backbone = inputs.z_backbone
    reference_backbone = z_backbone.clone()
    reference_conditions = inputs.condition_latents.clone()
    latent_frames = z_backbone.shape[3]
    h_tok = z_backbone.shape[-2] // 2
    w_tok = z_backbone.shape[-1] // 2
    K_b = K_all.to(device=device, dtype=torch.float32).unsqueeze(0)
    E_b = E_all.to(device=device, dtype=torch.float32).unsqueeze(0)
    T_b = T_anchor_front.to(device=device, dtype=torch.float32).unsqueeze(0)
    geometry = _resolve_geometry_override(inputs, target_view, geometry_mode)
    backbone_plucker = build_p61_dynamic_plucker_streams(K_b, E_b, T_b, geometry.backbone_plucker_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_backbone.dtype)
    condition_plucker = build_p61_dynamic_plucker_streams(K_b, E_b, T_b, geometry.condition_plucker_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_backbone.dtype)
    if geometry.zero_plucker:
        backbone_plucker = torch.zeros_like(backbone_plucker)
        condition_plucker = torch.zeros_like(condition_plucker)
    T_anchor_cam = build_p61_anchor_camera_transforms(E_b, T_b)
    if geometry.zero_relative_pose:
        relative_pose_features = z_backbone.new_zeros(T_anchor_cam.shape[0], T_anchor_cam.shape[2], 3, T_anchor_cam.shape[2], 6)
        target_pose_features = z_backbone.new_zeros(T_anchor_cam.shape[0], T_anchor_cam.shape[2], 6)
    else:
        relative_pose_features = build_p62_time_window_relative_pose_features(T_anchor_cam, geometry.relative_pose_target_view, geometry.condition_plucker_view_ids, dtype=z_backbone.dtype)
        target_pose_features = build_p62_target_pose_features(T_anchor_cam, geometry.relative_pose_target_view, dtype=z_backbone.dtype)
    (warp_t, vis_t) = bundle.warp_provider(target_view)
    warp_t = warp_t.to(device=device, dtype=torch.bfloat16)
    vis_t = vis_t.to(device=device, dtype=torch.float16)
    if warp_t.shape[0] != z_backbone.shape[0]:
        warp_t = warp_t.expand(z_backbone.shape[0], *warp_t.shape[1:]).contiguous()
        vis_t = vis_t.expand(z_backbone.shape[0], *vis_t.shape[1:]).contiguous()
    (text_emb, text_mask) = bundle.text_encoder.encode_cached(text_prompt)
    (null_emb, null_mask) = bundle.text_encoder.null_cached()
    text_emb = _ensure_batched_text(text_emb, 'text_emb').to(device=device)
    text_mask = _ensure_batched_mask(text_mask, 'text_mask').to(device=device)
    null_emb = _ensure_batched_text(null_emb, 'null_emb').to(device=device)
    null_mask = _ensure_batched_mask(null_mask, 'null_mask').to(device=device)
    batch = z_backbone.shape[0]
    cond_style = cond_style_ids.to(device=device, dtype=torch.long).view(1, -1)
    uncond_style = uncond_style_ids.to(device=device, dtype=torch.long).view(1, -1)
    if batch > 1:
        cond_style = cond_style.expand(batch, -1).contiguous()
        uncond_style = uncond_style.expand(batch, -1).contiguous()
    sigmas = sigmas_descending_from_start(bundle.scheduler, num_steps, start_sigma, device=device, dtype=z_backbone.dtype)

    def _canonical(t_emb: torch.Tensor, t_mask: torch.Tensor) -> dict[str, Any]:
        return dict(z_backbone=z_backbone, sigma=sigma, text_emb=t_emb, text_mask=t_mask, backbone_plucker=backbone_plucker, backbone_view_ids=geometry.dit_backbone_view_ids, backbone_role_ids=inputs.backbone_role_ids, T_anchor_front=T_b, condition_latents=inputs.condition_latents, condition_plucker=condition_plucker, condition_view_ids=geometry.dit_condition_view_ids, condition_type_ids=inputs.condition_type_ids, condition_available_mask=inputs.condition_available_mask, relative_pose_features=relative_pose_features, target_pose_features=target_pose_features, warped_target_latent=warp_t, warped_target_visibility=vis_t)
    for (sigma_value, sigma_next) in zip(sigmas[:-1], sigmas[1:]):
        sigma = sigma_value.view(1)
        _reset_condition_inputs(inputs, reference_backbone, reference_conditions)
        v_cond = _dit_forward(bundle.dit, _canonical(text_emb, text_mask), cond_style)
        v_uncond = _dit_forward(bundle.dit, _canonical(null_emb, null_mask), uncond_style)
        v_pred = v_uncond + guide_scale * (v_cond - v_uncond)
        z_backbone[:, P61_TARGET_STREAM_INDEX] = z_backbone[:, P61_TARGET_STREAM_INDEX] + (sigma_next - sigma_value).view(1, 1, 1, 1, 1) * v_pred[:, P61_TARGET_STREAM_INDEX]
        _reset_condition_inputs(inputs, reference_backbone, reference_conditions)
    bundle.last_condition_preserved = bool(torch.equal(z_backbone[:, 0], reference_backbone[:, 0]) and torch.equal(inputs.condition_latents, reference_conditions))
    return z_backbone[0, P61_TARGET_STREAM_INDEX]

def _guide_for_view(view_id: int, cross_guide: float, rear_guide: float) -> float:
    return rear_guide if view_id in REAR_VIEWS else cross_guide

def generate_five_views(z_front: torch.Tensor, k_all: torch.Tensor, e_all: torch.Tensor, pose: torch.Tensor, prompt: str, bundle: InferenceBundleWarp, cond_style_ids: torch.Tensor, uncond_style_ids: torch.Tensor, cross_guide: float, rear_guide: float, num_steps: int, start_sigma: float, shared_noise_alpha: float, generator: torch.Generator) -> list[torch.Tensor]:
    device = z_front.device
    batch_shape = z_front.unsqueeze(0).shape
    shared = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
    alpha = float(shared_noise_alpha)
    generated: dict[int, torch.Tensor] = {}
    preds: list[torch.Tensor] = []
    for view_id in ALL_TARGET_VIEWS:
        private = torch.randn(batch_shape, device=device, dtype=z_front.dtype, generator=generator)
        target_noise = alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * private
        latent = inference_register_single_target(z_front, k_all, e_all, pose, prompt, bundle, view_id, cond_style_ids=cond_style_ids, uncond_style_ids=uncond_style_ids, guide_scale=_guide_for_view(view_id, cross_guide, rear_guide), condition_latents_by_view=generated, target_noise=target_noise, num_steps=num_steps, start_sigma=start_sigma, generator=generator)
        generated[view_id] = latent
        preds.append(latent)
    return preds

def _resolve_config(config_name: str) -> Any:
    for (modname, attrs) in [('openlongtail.configs.openlongtail_register_vace', ('OPENLONGTAIL_REGISTER_1P3B_CONFIG', 'OPENLONGTAIL_REGISTER_CONFIG'))]:
        try:
            mod = __import__(modname, fromlist=['*'])
        except Exception:
            continue
        for attr in (config_name.upper() + '_CONFIG',) + attrs:
            cfg = getattr(mod, attr, None)
            if cfg is not None:
                return cfg
    from openlongtail.scripts.inference_smoke import CONFIGS
    return CONFIGS['openlongtail_1p3b']

def _build_dit_register(config_name: str, device: torch.device, wan21_vace_dir: Path | None, checkpoint_dir: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    from openlongtail.configs.cfg_p61_vace import P61_CONDITION_ENCODER_LAYERS, P61_SEMANTIC_QUERIES, P61_SYNC_TEMPORAL_WINDOW
    from openlongtail.configs.openlongtail_vace import GEO_HEAD_DIM, GEO_PROJECTION_TEMPERATURE, GRAPH_GATE_INIT_BIAS
    from openlongtail.models.dit_register_vace import DiTVACERegister
    from openlongtail.models.wan21_vace_backbone import default_wan21_vace_dir, load_wan21_vace_expert
    from openlongtail.models.wan21_vace_lora import inject_lora_into_wan21_vace_expert
    from openlongtail.training.checkpoint_warp import load_warp_checkpoint
    from openlongtail.training.train import ray_shared_modules
    config = _resolve_config(config_name)
    checkpoint_dir = Path(checkpoint_dir)
    is_fulltune = (checkpoint_dir / 'backbone.pt').exists()
    expert = load_wan21_vace_expert(wan21_vace_dir or default_wan21_vace_dir(), dtype=torch.bfloat16, device=device)
    if not is_fulltune:
        inject_lora_into_wan21_vace_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    dit = DiTVACERegister(expert, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks, sync_temporal_window=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers=P61_CONDITION_ENCODER_LAYERS, semantic_queries=P61_SEMANTIC_QUERIES, graph_gate_init_bias=GRAPH_GATE_INIT_BIAS, geo_head_dim=GEO_HEAD_DIM, geo_projection_temperature=GEO_PROJECTION_TEMPERATURE).to(device=device, dtype=torch.bfloat16)
    payload = load_warp_checkpoint(checkpoint_dir)
    load_info: dict[str, Any] = {'mode': 'checkpoint', 'is_fulltune': bool(is_fulltune), 'checkpoint_dir': str(checkpoint_dir), 'checkpoint_metadata': payload.get('metadata', {})}
    if is_fulltune:
        if 'backbone' not in payload:
            raise FileNotFoundError(f'expected backbone.pt in fulltune ckpt: {checkpoint_dir}')
        (m, u) = dit.load_state_dict(payload['backbone'], strict=False)
    else:
        if 'lora' not in payload:
            raise FileNotFoundError(f'expected lora.pt in checkpoint: {checkpoint_dir}')
        (m, u) = dit.load_state_dict(payload['lora'], strict=False)
    load_info['missing_keys'] = list(m)
    load_info['unexpected_keys'] = list(u)
    if 'shared_modules' in payload:
        (sm, su) = ray_shared_modules(dit).load_state_dict(payload['shared_modules'], strict=False)
        load_info['missing_shared_keys'] = list(sm)
        load_info['unexpected_shared_keys'] = list(su)
    handled = {'metadata.json', 'lora.pt', 'shared_modules.pt', 'optimizer.pt', 'backbone.pt'}
    extra_loaded: list[str] = []
    for extra in sorted(checkpoint_dir.glob('*.pt')):
        if extra.name in handled:
            continue
        try:
            sd = torch.load(extra, map_location='cpu', weights_only=True)
        except Exception:
            continue
        if isinstance(sd, dict) and sd and all((isinstance(v, torch.Tensor) for v in sd.values())):
            dit.load_state_dict(sd, strict=False)
            extra_loaded.append(extra.name)
    if extra_loaded:
        load_info['extra_module_files'] = extra_loaded
    dit.eval().requires_grad_(False)
    return (dit, load_info)

def _p4_path(clip_pt: Path) -> Path:
    return clip_pt.with_name(f'{clip_pt.stem}_p4.pt')

def _front_mp4_exists(data_root: Path, uuid: str) -> bool:
    return (data_root / uuid / 'all_views_undistorted_simplecalib' / 'camera' / FRONT_SENSOR / f'{uuid}.{FRONT_SENSOR}.mp4').exists()

def _depth_exists(data_root: Path, uuid: str) -> bool:
    return (data_root / uuid / 'depthcrafter_cache' / 'fullseq_h_384_w_672' / 'front_depth.pt').exists()

def load_nv_clip_inputs(clip_pt: Path, data_root_override: Path | None=None) -> dict[str, Any]:
    cache = torch.load(clip_pt, map_location='cpu', weights_only=False)
    uuid = str(cache['uuid'])
    clip_id = int(cache['clip_id'])
    (out_h, out_w) = (int(x) for x in cache['output_size'])
    front_indices = cache['frame_indices'][FRONT_SENSOR].numpy()
    data_root = Path(data_root_override) if data_root_override is not None else Path(cache['data_root'])
    front_np = load_front_rgb_clip(uuid, data_root, front_indices, out_h, out_w)
    front = torch.from_numpy(np.ascontiguousarray(front_np)).permute(0, 3, 1, 2).contiguous().to(torch.uint8)
    depth = load_depth_clip(uuid, data_root, front_indices, out_h, out_w)
    pose = torch.load(_p4_path(clip_pt), map_location='cpu', weights_only=False)['T_anchor_front'].float()
    return {'uuid': uuid, 'clip_id': clip_id, 'output_size': (out_h, out_w), 'front_rgb': front, 'front_depth': depth, 'K': cache['K'].float(), 'E': cache['E'].float(), 'T_anchor_front': pose, 'z_all': cache['z_all'], 'data_root': str(data_root)}

def collect_v3_clips(root: Path, max_clips: int, shard_index: int, num_shards: int) -> list[tuple[str, int, Path]]:
    clips: list[tuple[str, int, Path]] = []
    for uuid_dir in sorted((root / 'per_uuid').iterdir()):
        if not uuid_dir.is_dir():
            continue
        for cp in sorted(uuid_dir.glob('clip_*.pt')):
            stem = cp.stem
            if '_' in stem.replace('clip_', '', 1):
                continue
            try:
                cid = int(stem.replace('clip_', ''))
            except ValueError:
                continue
            clips.append((uuid_dir.name, cid, cp))
    sharded = clips[shard_index::num_shards]
    if max_clips > 0:
        sharded = sharded[:max_clips]
    return sharded

def _load_caption_encoder(v3_root: Path, uuid: str, config: Any, device: torch.device) -> tuple[CachedTextEncoder, str]:
    from openlongtail.data.text_emb_cache import load_text_embedding
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    null_e = null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    null_m = null_mask.unsqueeze(0).to(device=device)
    text_path = Path(v3_root) / 'text_cache' / 'per_uuid' / f'{uuid}.pt'
    if text_path.exists():
        pkg = torch.load(text_path, map_location='cpu', weights_only=False)
        emb = pkg.get('text_emb', pkg.get('emb'))
        mask = pkg.get('text_mask', pkg.get('mask', pkg.get('attention_mask')))
        prompt = str(pkg.get('prompt', ''))
        if emb is not None and mask is not None:
            return (CachedTextEncoder(text_emb=emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=mask.unsqueeze(0).to(device=device), null_emb=null_e, null_mask=null_m), prompt)
    return (CachedTextEncoder(text_emb=null_e.clone(), text_mask=null_m.clone(), null_emb=null_e, null_mask=null_m), '')

def _load_txt_caption_encoder(caption_cache: Path, uuid: str, config: Any, device: torch.device, umt5: Any) -> tuple[CachedTextEncoder, str]:
    from openlongtail.data.text_emb_cache import load_text_embedding
    from openlongtail.scripts.inference import _resolve_caption
    (null_emb, null_mask) = load_text_embedding(config.data.text_emb_cache_root, 'null')
    null_e = null_emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    null_m = null_mask.unsqueeze(0).to(device=device)
    (caption, _src) = _resolve_caption(caption_cache, uuid, None)
    if caption:
        (emb, mask) = umt5.encode(caption)
        return (CachedTextEncoder(text_emb=emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16), text_mask=mask.unsqueeze(0).to(device=device), null_emb=null_e, null_mask=null_m), caption)
    print(f'  WARNING: no caption for uuid={uuid}; CFG rides on null', flush=True)
    return (CachedTextEncoder(text_emb=null_e.clone(), text_mask=null_m.clone(), null_emb=null_e, null_mask=null_m), '')

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='5-view inference')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--v3-root', type=Path, default=None, help='latent-cache dir (NV latent-cache clips)')
    src.add_argument('--test-data-root', type=Path, default=None, help='front-dashcam TestData root (LTtestMay10 layout)')
    p.add_argument('--output-dir', type=Path, required=False)
    p.add_argument('--checkpoint-dir', type=Path, required=False)
    p.add_argument('--config', type=str, default='openlongtail_register_1p3b')
    p.add_argument('--wan21-vace-dir', type=Path, default=Path('checkpoints/Wan2.1-VACE-1.3B'))
    p.add_argument('--style-cache-root', type=Path, default=Path('data/clips/style_cache'))
    p.add_argument('--data-root', type=Path, default=None, help='override the raw RGB/depth data_root (default: from the clip)')
    p.add_argument('--caption-cache', type=Path, default=None, help='(--test-data-root only) dir with per_uuid/<uuid>.txt one-sentence captions (Qwen-derived); UMT5-encoded at runtime as the cond text so CFG engages. Mirrors inference --caption-cache exactly.')
    p.add_argument('--warp-mode', choices=('live', 'sidecar'), default='live')
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
    p.add_argument('--treat-as-nv', action='store_true', help='(--test-data-root only) override K/E with the PAI reference rig; required for cross-camera dashcam sources (Nexar/Waymo).')
    p.add_argument('--self-test', action='store_true', help='CPU-only wiring/warp self-test (no GPU/ckpt)')
    return p.parse_args(argv)

def _run_v3(args: argparse.Namespace, dit: torch.nn.Module, vae: object, config: Any, device: torch.device) -> None:
    unknown_ids = resolve_unknown_style_ids(dit)
    uncond_style = torch.tensor(unknown_ids, dtype=torch.long)
    clips = collect_v3_clips(args.v3_root, args.max_clips, args.shard_index, args.num_shards)
    scheduler = FlowMatchScheduler(shift=config.model.sample_shift)
    root_summary: dict[str, Any] = {'source': 'v3', 'num_clips': len(clips), 'warp_mode': args.warp_mode, 'clips': []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config_used.json').write_text(json.dumps(_jsonable(vars(args)), indent=2, sort_keys=True))
    for (idx, (uuid, cid, clip_pt)) in enumerate(clips, start=1):
        clip_id = f'clip_{cid:06d}'
        clip_out = args.output_dir / uuid / clip_id
        done = clip_out / 'clip_done.json'
        if done.exists() and (not args.overwrite):
            root_summary['clips'].append({'clip': f'{uuid}/{clip_id}', 'status': 'skipped'})
            continue
        clip_out.mkdir(parents=True, exist_ok=True)
        print(f'[{idx}/{len(clips)}] {uuid}/{clip_id} warp={args.warp_mode}', flush=True)
        started = time.time()
        cache = torch.load(clip_pt, map_location='cpu', weights_only=False)
        z_all = cache['z_all'].to(device=device, dtype=torch.bfloat16)
        z_front = z_all[0]
        k_all = cache['K'].float().to(device)
        e_all = cache['E'].float().to(device)
        pose = torch.load(_p4_path(clip_pt), map_location='cpu', weights_only=False)['T_anchor_front'].float().to(device)
        warp_summary: dict[str, Any] = {}
        if args.warp_mode == 'sidecar':
            sidecar = clip_pt.with_name(f'{clip_id}_warp.pt')
            adj_sidecar = clip_pt.with_name(f'{clip_id}_lookback.pt')
            if not sidecar.exists():
                root_summary['clips'].append({'clip': f'{uuid}/{clip_id}', 'status': 'missing_sidecar'})
                continue
            warp_provider = make_sidecar_warp_provider(str(sidecar), str(adj_sidecar) if adj_sidecar.exists() else None, device=device)
        else:
            inp = load_nv_clip_inputs(clip_pt, args.data_root)
            (warp_provider, warp_summary) = build_live_warp_provider(inp['front_rgb'], inp['front_depth'], k_all, e_all, pose, vae, device, splat_radius=args.splat_radius)
        (text_encoder, prompt) = _load_caption_encoder(args.v3_root, uuid, config, device)
        (cond_style, style_src) = load_style_ids_for_uuid(args.style_cache_root, uuid, unknown_ids)
        bundle = InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=scheduler, warp_provider=warp_provider)
        generator = torch.Generator(device=device).manual_seed(args.seed + idx + args.shard_index * 100000)
        preds = generate_five_views(z_front, k_all, e_all, pose, prompt, bundle, cond_style, uncond_style, args.cross_guide, args.rear_guide, args.num_steps, args.start_sigma, args.shared_noise_alpha, generator)
        pred = _decode_targets(vae, torch.stack(preds, dim=0)).detach().cpu()
        gt = _decode_targets(vae, z_all[1:6].detach()).detach().cpu()
        front_dec = _decode_targets(vae, z_all[0:1].detach()).detach().cpu()
        pred_tchw = pred.permute(0, 2, 1, 3, 4).contiguous()
        gt_tchw = gt.permute(0, 2, 1, 3, 4).contiguous()
        front_tchw = front_dec[0].permute(1, 0, 2, 3).contiguous()
        outputs: dict[str, Any] = {'front_input': _write_mp4(clip_out / 'front_input.mp4', front_tchw, 16), 'gt': {}, 'pred': {}}
        for (li, vid) in enumerate(ALL_TARGET_VIEWS):
            outputs['gt'][str(vid)] = _write_mp4(clip_out / f'gt_{VIEW_NAMES[vid]}.mp4', gt_tchw[li], 16)
            outputs['pred'][str(vid)] = _write_mp4(clip_out / f'pred_{VIEW_NAMES[vid]}.mp4', pred_tchw[li], 16)
        outputs['comparison_grid'] = _write_mp4(clip_out / 'comparison_grid.mp4', _make_comparison_grid(front_tchw, gt_tchw, pred_tchw, list(ALL_TARGET_VIEWS)), 16)
        clip_summary = {'clip': f'{uuid}/{clip_id}', 'status': 'ok', 'elapsed_sec': time.time() - started, 'prompt': prompt, 'style_ids': cond_style.tolist(), 'style_source': style_src, 'uncond_style_ids': uncond_style.tolist(), 'warp_summary': warp_summary, 'outputs': outputs}
        done.write_text(json.dumps(_jsonable(clip_summary), indent=2, sort_keys=True))
        root_summary['clips'].append(clip_summary)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (args.output_dir / f'summary_shard{args.shard_index}.json').write_text(json.dumps(_jsonable(root_summary), indent=2, sort_keys=True))
    print(f'DONE: {len(clips)} v3 clips -> {args.output_dir}', flush=True)

def _run_testdata(args: argparse.Namespace, dit: torch.nn.Module, vae: object, config: Any, device: torch.device) -> None:
    from openlongtail.scripts.inference_testdata_cross import _load_clip_tensors, _load_reference_calibration, collect_clip_dirs, reconstruct_e_all
    if args.warp_mode == 'sidecar':
        raise ValueError('--warp-mode sidecar is not available for --test-data-root (dashcam clips have no NV sidecar sidecars); use --warp-mode live.')
    unknown_ids = resolve_unknown_style_ids(dit)
    uncond_style = torch.tensor(unknown_ids, dtype=torch.long)
    clips = collect_clip_dirs(args.test_data_root, args.max_clips, args.shard_index, args.num_shards)
    (reference_k, reference_e, _) = _load_reference_calibration(Path('data/clips'), None)
    scheduler = FlowMatchScheduler(shift=config.model.sample_shift)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary: dict[str, Any] = {'source': 'testdata', 'num_clips': len(clips), 'warp_mode': 'live', 'clips': []}
    (args.output_dir / 'config_used.json').write_text(json.dumps(_jsonable(vars(args)), indent=2, sort_keys=True))
    _umt5_enc = None
    if args.caption_cache is not None:
        from openlongtail.scripts.inference import _RuntimeUMT5Encoder
        _umt5_enc = _RuntimeUMT5Encoder(str(device))
    for (idx, clip_dir) in enumerate(clips, start=1):
        rel = clip_dir.relative_to(args.test_data_root)
        clip_out = args.output_dir / rel
        done = clip_out / 'clip_done.json'
        if done.exists() and (not args.overwrite):
            root_summary['clips'].append({'clip': str(rel), 'status': 'skipped'})
            continue
        clip_out.mkdir(parents=True, exist_ok=True)
        print(f'[{idx}/{len(clips)}] {rel} warp=live', flush=True)
        started = time.time()
        (front, depth, pose, k_front, meta) = _load_clip_tensors(clip_dir)
        if args.treat_as_nv:
            (k_all, e_all) = (reference_k.clone(), reference_e.clone())
        else:
            (k_all, e_all) = reconstruct_e_all(k_front, meta['E_rig_front'].float(), reference_k=reference_k, reference_e=reference_e)
        k_all = k_all.to(device)
        e_all = e_all.to(device)
        pose = pose.to(device)
        (warp_provider, warp_summary) = build_live_warp_provider(front, depth, k_all, e_all, pose, vae, device, splat_radius=args.splat_radius)
        z_front = _encode_front(vae, front.to(device)).to(device)
        uuid = str(meta.get('uuid', ''))
        if args.caption_cache is not None:
            (text_encoder, prompt) = _load_txt_caption_encoder(args.caption_cache, uuid, config, device, _umt5_enc)
        else:
            (text_encoder, prompt) = _load_caption_encoder(args.v3_root or Path('data/clips'), uuid, config, device)
        (cond_style, style_src) = load_style_ids_for_uuid(args.style_cache_root, uuid, unknown_ids)
        bundle = InferenceBundleWarp(vae=vae, dit=dit, text_encoder=text_encoder, scheduler=scheduler, warp_provider=warp_provider)
        generator = torch.Generator(device=device).manual_seed(args.seed + idx + args.shard_index * 100000)
        preds = generate_five_views(z_front, k_all, e_all, pose, prompt or str(rel), bundle, cond_style, uncond_style, args.cross_guide, args.rear_guide, args.num_steps, args.start_sigma, args.shared_noise_alpha, generator)
        pred = _decode_targets(vae, torch.stack(preds, dim=0)).detach().cpu()
        pred_tchw = pred.permute(0, 2, 1, 3, 4).contiguous()
        fps = int(meta.get('src_fps', getattr(config.data, 'target_fps', 16)))
        outputs: dict[str, Any] = {'front_input': _write_mp4(clip_out / 'front_input.mp4', front, fps), 'pred': {}}
        rows = [front]
        for (li, vid) in enumerate(ALL_TARGET_VIEWS):
            outputs['pred'][str(vid)] = _write_mp4(clip_out / f'pred_{VIEW_NAMES[vid]}.mp4', pred_tchw[li], fps)
            rows.append(pred_tchw[li])
        outputs['pred_grid'] = _write_mp4(clip_out / 'pred_grid.mp4', torch.cat(rows, dim=-1), fps)
        clip_summary = {'clip': str(rel), 'status': 'ok', 'elapsed_sec': time.time() - started, 'prompt': prompt, 'style_ids': cond_style.tolist(), 'style_source': style_src, 'warp_summary': warp_summary, 'outputs': outputs}
        done.write_text(json.dumps(_jsonable(clip_summary), indent=2, sort_keys=True))
        root_summary['clips'].append(clip_summary)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (args.output_dir / f'summary_shard{args.shard_index}.json').write_text(json.dumps(_jsonable(root_summary), indent=2, sort_keys=True))
    print(f'DONE: {len(clips)} testdata clips -> {args.output_dir}', flush=True)

def main(argv: list[str] | None=None) -> None:
    args = parse_args(argv)
    if args.self_test:
        raise SystemExit(self_test(args.v3_root, args.data_root))
    if args.output_dir is None or args.checkpoint_dir is None:
        raise ValueError('--output-dir and --checkpoint-dir are required for a real run')
    if args.v3_root is None and args.test_data_root is None:
        raise ValueError('provide --v3-root or --test-data-root')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = _resolve_config(args.config)
    from openlongtail.models.wan_vae import load_wan21_vae
    (dit, load_info) = _build_dit_register(args.config, device, args.wan21_vace_dir, args.checkpoint_dir)
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    print('load_info:', json.dumps(_jsonable(load_info), sort_keys=True)[:2000], flush=True)
    if args.v3_root is not None:
        _run_v3(args, dit, vae, config, device)
    else:
        _run_testdata(args, dit, vae, config, device)

def _find_live_test_clip(v3_root: Path, data_root_override: Path | None, max_scan: int=400) -> Path | None:
    per_uuid = v3_root / 'per_uuid'
    if not per_uuid.is_dir():
        return None
    scanned = 0
    for uuid_dir in sorted(per_uuid.iterdir()):
        if not uuid_dir.is_dir():
            continue
        scanned += 1
        if scanned > max_scan:
            break
        uuid = uuid_dir.name
        clip0 = uuid_dir / 'clip_000000.pt'
        if not clip0.exists() or not _p4_path(clip0).exists():
            continue
        try:
            cache = torch.load(clip0, map_location='cpu', weights_only=False)
        except Exception:
            continue
        data_root = Path(data_root_override) if data_root_override is not None else Path(cache.get('data_root', ''))
        if _front_mp4_exists(data_root, uuid) and _depth_exists(data_root, uuid):
            return clip0
    return None

def self_test(v3_root: Path | None, data_root: Path | None) -> int:
    torch.manual_seed(0)
    device = torch.device('cpu')
    v3_root = v3_root or Path('data/clips')
    style_root = Path('data/clips/style_cache')
    print('=' * 72)
    print('SELF-TEST (CPU)')
    print('=' * 72)
    import openlongtail.scripts.inference_register as _self
    print('[ok] module imports on CPU (no GPU / no model / no config needed)')
    clip_pt = _find_live_test_clip(v3_root, data_root)
    if clip_pt is None:
        print('[FAIL] could not find an NV clip with raw front RGB + depth + _p4 under', v3_root)
        return 1
    inp = load_nv_clip_inputs(clip_pt, data_root)
    print(f"[ok] loaded NV clip {inp['uuid']}/clip_{inp['clip_id']:06d}  front={tuple(inp['front_rgb'].shape)} depth={tuple(inp['front_depth'].shape)} E={tuple(inp['E'].shape)} pose={tuple(inp['T_anchor_front'].shape)}")
    (provider, summary) = build_live_warp_provider(inp['front_rgb'], inp['front_depth'], inp['K'], inp['E'], inp['T_anchor_front'], vae=None, device=device, splat_radius=1)
    print('\n  per-view REAR coverage (latent grid): same-time  vs  live-lookback  -> merged')
    measured: dict[int, dict[str, float]] = {}
    for vid in REAR_VIEWS:
        s = summary[str(vid)]
        (st, lb, mg) = (s['same_time_vis_lat_pct'], s['lookback_vis_lat_pct'], s['merged_vis_lat_pct'])
        measured[vid] = {'same_time': st, 'lookback': lb, 'merged': mg}
        print(f"    view {vid} ({VIEW_NAMES[vid]:<10}): same={st:6.2f}%   lookback={lb:6.2f}%   merged={mg:6.2f}%   (lookback offset K={s['lookback_offset']})")
    for vid in CROSS_VIEWS:
        s = summary[str(vid)]
        print(f"    view {vid} ({VIEW_NAMES[vid]:<10}): same={s['same_time_vis_lat_pct']:6.2f}%   (cross=same-time)")
    (_, vis3) = provider(3)
    vis3_mean_pct = float(vis3.float().mean().item() * 100.0)
    print(f'\n  provider(3) -> visibility tensor {tuple(vis3.shape)} mean={vis3_mean_pct:.2f}%')
    v3 = measured[3]
    if not (vis3_mean_pct > 1.0 and v3['lookback'] > 1.0 and (v3['lookback'] > v3['same_time'] + 1.0)):
        print('\n[FAIL] rear provider(3) live-lookback coverage did not clear the >0 & >> same-time bar')
        return 1
    for vid in REAR_VIEWS:
        assert measured[vid]['lookback'] >= measured[vid]['same_time'] - 1e-06, f'view {vid} lookback < same-time'
    if measured[5]['lookback'] < 1.0:
        print('  note: rear_tele (view 5, 30-FOV) coverage is inherently low even offline; it needs a longer baseline (this is expected, not a failure).')
    print('\n[ok] LIVE lookback gives non-trivial REAR coverage (provider(3) vis > 0) where same-time is ~0 -> 0%-rear-coverage bug eliminated')
    unknown_ids = resolve_unknown_style_ids(None)
    (cond_style, src) = load_style_ids_for_uuid(style_root, inp['uuid'], unknown_ids)
    if torch.equal(cond_style, torch.tensor(unknown_ids, dtype=torch.long)):
        cond_style = torch.tensor([0, 0, 0], dtype=torch.long)
        src = f'{src}->forced[0,0,0]'
    uncond_style = torch.tensor(unknown_ids, dtype=torch.long)
    print(f'\n  style cond={cond_style.tolist()} (src={src})  uncond(UNKNOWN)={uncond_style.tolist()}')
    assert not torch.equal(cond_style, uncond_style), 'cond/uncond style_ids must differ'
    calls: list[dict[str, Any]] = []

    class _RecordingDiT(torch.nn.Module):

        def forward(self, z_backbone, sigma, text_emb, text_mask, backbone_plucker, backbone_view_ids, backbone_role_ids, T_anchor_front, condition_latents, condition_plucker, condition_view_ids, condition_type_ids, condition_available_mask, relative_pose_features, target_pose_features, warped_target_latent, warped_target_visibility, style_ids=None):
            calls.append({'style_ids': None if style_ids is None else style_ids.clone(), 'text_mean': float(text_emb.float().mean().item())})
            return torch.zeros_like(z_backbone)
    emb_dim = 16
    cap_e = torch.full((1, 4, emb_dim), 0.7)
    null_e = torch.zeros((1, 4, emb_dim))
    msk = torch.ones((1, 4), dtype=torch.long)
    text_encoder = CachedTextEncoder(text_emb=cap_e, text_mask=msk, null_emb=null_e, null_mask=msk)
    z_front = inp['z_all'][0].to(torch.float32)
    zero_provider = lambda vid: (torch.zeros(1, 16, 11, 60, 104), torch.zeros(1, 1, 11, 60, 104))
    bundle = InferenceBundleWarp(vae=None, dit=_RecordingDiT(), text_encoder=text_encoder, scheduler=FlowMatchScheduler(shift=5.0), warp_provider=zero_provider)
    gen = torch.Generator(device='cpu').manual_seed(0)
    _ = inference_register_single_target(z_front, inp['K'], inp['E'], inp['T_anchor_front'], 'caption', bundle, target_view=3, cond_style_ids=cond_style, uncond_style_ids=uncond_style, guide_scale=7.0, num_steps=2, generator=gen)
    assert len(calls) >= 2, f'expected >=2 DiT calls (cond+uncond), got {len(calls)}'
    (c0, c1) = (calls[0], calls[1])
    assert c0['style_ids'] is not None and c1['style_ids'] is not None, 'style_ids not threaded to DiT'
    assert torch.equal(c0['style_ids'][0], cond_style.to(torch.long)), 'cond call got wrong style_ids'
    assert torch.equal(c1['style_ids'][0], uncond_style.to(torch.long)), 'uncond call got wrong style_ids'
    assert not torch.equal(c0['style_ids'], c1['style_ids']), 'cond/uncond style_ids identical at DiT'
    assert abs(c0['text_mean'] - c1['text_mean']) > 1e-06, 'cond/uncond caption identical at DiT'
    print(f"[ok] STYLE-CFG wiring: DiT call#0 style={c0['style_ids'][0].tolist()} (caption), call#1 style={c1['style_ids'][0].tolist()} (null) -> cond != uncond")
    print('\n' + '=' * 72)
    print('SELF-TEST PASSED')
    print(f"  clip: {inp['uuid']}/clip_{inp['clip_id']:06d}")
    print('  measured REAR coverage (latent %, live-lookback vs same-time):')
    for vid in REAR_VIEWS:
        m = measured[vid]
        print(f"    view {vid} {VIEW_NAMES[vid]:<10}: lookback={m['lookback']:.2f}%  same-time={m['same_time']:.2f}%  merged={m['merged']:.2f}%")
    print('=' * 72)
    return 0
if __name__ == '__main__':
    main()
