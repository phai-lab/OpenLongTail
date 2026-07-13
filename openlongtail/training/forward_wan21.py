from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from openlongtail.data.transforms import se3_inverse, se3_log_rotation_angle
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import _prepare_z_all_for_p3, _resolve_batch_device, build_p3_streams_from_latents, resolve_p3_target_views
from openlongtail.training.schedulers import FlowMatchScheduler
FRONT_IDX = 0

@dataclass
class RayTrainingStepWan21Output:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int]]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def sample_sigma_unified(batch: int, scheduler: FlowMatchScheduler, device: torch.device | str) -> torch.Tensor:
    if batch <= 0:
        raise ValueError(f'expected batch > 0, got {batch}')
    t = torch.rand(batch, device=device)
    return scheduler.sigmas_for(t)

def _base_metrics(loss: torch.Tensor, sigma: torch.Tensor, target_views: list[int], stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor, training_method: str) -> dict[str, torch.Tensor | str | list[int]]:
    return {'loss': loss.detach(), 'stage': 'unified', 'training_method': training_method, 'target_views': target_views, 'stream_view_ids': [int(item) for item in stream_view_ids.detach().cpu().tolist()], 'stream_role_ids': [int(item) for item in stream_role_ids.detach().cpu().tolist()], 'sigma_mean': sigma.detach().float().mean(), 'sigma_min': sigma.detach().float().min(), 'sigma_max': sigma.detach().float().max()}

def training_step_ray_p3_wan21(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, num_targets: int=3) -> RayTrainingStepWan21Output:
    resolved_device = _resolve_batch_device(batch, device)
    target_views = resolve_p3_target_views(resolved_device, num_targets=num_targets)
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, target_views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p3_streams_from_latents(z_all, target_views, sigma)
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E = batch['E'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    with torch.no_grad():
        plucker_all = compute_plucker_rays(K, E, h_tok=h_tok, w_tok=w_tok).to(dtype=z_all.dtype)
        plucker = plucker_all[:, streams.stream_view_ids]
        latent_frames = z_all.shape[3]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    dit = components.low_dit
    v_pred = dit(streams.z_streams, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, streams.stream_view_ids, streams.stream_role_ids)
    pred_targets = v_pred[:, streams.target_local_indices]
    loss = ((pred_targets - streams.v_target) ** 2).mean()
    metrics = _base_metrics(loss, sigma, target_views, streams.stream_view_ids, streams.stream_role_ids, 'p3_wan21_single_expert_unified')
    for (local_idx, global_view_idx) in enumerate(target_views):
        metrics[f'flow_view_{global_view_idx}'] = ((pred_targets[:, local_idx] - streams.v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepWan21Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)

def training_step_ray_p4_wan21(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, num_targets: int=3) -> RayTrainingStepWan21Output:
    resolved_device = _resolve_batch_device(batch, device)
    target_views = resolve_p3_target_views(resolved_device, num_targets=num_targets)
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, target_views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p3_streams_from_latents(z_all, target_views, sigma)
    if 'T_anchor_front' not in batch:
        raise KeyError("Wan21-P4 training requires batch['T_anchor_front']")
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E_rig = batch['E'].to(device=resolved_device, dtype=torch.float32)
    T_anchor_front = batch['T_anchor_front'].to(device=resolved_device, dtype=torch.float32)
    if T_anchor_front.ndim != 4 or T_anchor_front.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_front shape (B, T, 4, 4), got {tuple(T_anchor_front.shape)}')
    if T_anchor_front.shape[0] != z_all.shape[0]:
        raise ValueError(f'expected T_anchor_front batch {z_all.shape[0]}, got {T_anchor_front.shape[0]}')
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    latent_frames = z_all.shape[3]
    if T_anchor_front.shape[1] < latent_frames:
        raise ValueError(f'expected T_anchor_front temporal dim >= {latent_frames}, got {T_anchor_front.shape[1]}')
    with torch.no_grad():
        E_front = E_rig[:, FRONT_IDX:FRONT_IDX + 1]
        E_front_cam = se3_inverse(E_front) @ E_rig
        E_anchor_cam = T_anchor_front[:, :, None] @ E_front_cam[:, None]
        plucker_all = compute_plucker_rays(K, E_anchor_cam, h_tok=h_tok, w_tok=w_tok).to(dtype=z_all.dtype)
        plucker = plucker_all[:, streams.stream_view_ids]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    dit = components.low_dit
    v_pred = dit(streams.z_streams, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, streams.stream_view_ids, streams.stream_role_ids)
    pred_targets = v_pred[:, streams.target_local_indices]
    loss = ((pred_targets - streams.v_target) ** 2).mean()
    metrics = _base_metrics(loss, sigma, target_views, streams.stream_view_ids, streams.stream_role_ids, 'p4_wan21_front_pose_single_expert_unified')
    front_translation_norm = T_anchor_front[:, :, :3, 3].norm(dim=-1)
    metrics['front_translation_norm_mean'] = front_translation_norm.mean().detach()
    metrics['front_translation_norm_max'] = front_translation_norm.max().detach()
    metrics['front_rotation_angle_max'] = se3_log_rotation_angle(T_anchor_front).abs().max().detach()
    for (local_idx, global_view_idx) in enumerate(target_views):
        metrics[f'flow_view_{global_view_idx}'] = ((pred_targets[:, local_idx] - streams.v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepWan21Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
