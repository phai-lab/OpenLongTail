from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from openlongtail.data.transforms import se3_inverse, se3_log_rotation_angle
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX
from openlongtail.models.dit_p62_vace import DiTP62VACE
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import _prepare_z_all_for_p3, _resolve_batch_device
from openlongtail.training.forward_ray_p5 import FRONT_IDX
from openlongtail.training.forward_ray_p61 import P61_TARGET_VIEWS, build_p61_anchor_camera_transforms, build_p61_dynamic_plucker_streams, build_p61_streams_from_latents, canonical_p61_target_view, p61_graph_layout, resolve_p61_target_view
from openlongtail.training.forward_wan21 import sample_sigma_unified
from openlongtail.training.schedulers import FlowMatchScheduler
P62_TARGET_VIEWS = P61_TARGET_VIEWS
P62_POSE_TRANSLATION_SCALE = 25.0
P62_POSE_ROTATION_SCALE = torch.pi
P62_POSE_FEATURE_CLIP = 4.0

@dataclass
class RayTrainingStepP62Output:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int] | float | int]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))

def _matrix_to_quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f'expected rotation matrices with shape (...,3,3), got {tuple(rotation.shape)}')
    m = rotation.float()
    (m00, m01, m02) = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2])
    (m10, m11, m12) = (m[..., 1, 0], m[..., 1, 1], m[..., 1, 2])
    (m20, m21, m22) = (m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    q_abs = _sqrt_positive_part(torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_wxyz = torch.stack([torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1), torch.stack([m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20], dim=-1), torch.stack([m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21], dim=-1), torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1)], dim=-2)
    denom = 2.0 * q_abs[..., None].clamp_min(1e-08)
    candidates = quat_by_wxyz / denom
    index = q_abs.argmax(dim=-1)
    gather_index = index[..., None, None].expand(*index.shape, 1, 4)
    quat = torch.gather(candidates, -2, gather_index).squeeze(-2)
    quat = torch.nn.functional.normalize(quat, dim=-1, eps=1e-08)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)

def _stable_so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    quat = _matrix_to_quaternion_wxyz(rotation)
    xyz = quat[..., 1:]
    sin_half = xyz.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 0].clamp_min(1e-08))
    scale = torch.where(sin_half < 1e-06, torch.full_like(sin_half, 2.0), angle / sin_half.clamp_min(1e-08))
    return xyz * scale.unsqueeze(-1)

def normalize_p62_pose_features(features: torch.Tensor) -> torch.Tensor:
    trans = features[..., :3] / float(P62_POSE_TRANSLATION_SCALE)
    rot = features[..., 3:] / float(P62_POSE_ROTATION_SCALE)
    normalized = torch.cat([trans, rot], dim=-1)
    return torch.nan_to_num(normalized, nan=0.0, posinf=P62_POSE_FEATURE_CLIP, neginf=-P62_POSE_FEATURE_CLIP).clamp(-P62_POSE_FEATURE_CLIP, P62_POSE_FEATURE_CLIP)

def build_p62_time_window_relative_pose_features(T_anchor_cam: torch.Tensor, target_view: int, condition_view_ids: torch.Tensor, dtype: torch.dtype | None=None) -> torch.Tensor:
    from openlongtail.models.dit_p61_vace import P61_CONDITION_SLOTS
    if T_anchor_cam.ndim != 5 or T_anchor_cam.shape[1] != 6 or T_anchor_cam.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_cam shape (B,6,T,4,4), got {tuple(T_anchor_cam.shape)}')
    if condition_view_ids.shape != (P61_CONDITION_SLOTS,):
        raise ValueError(f'expected condition_view_ids shape ({P61_CONDITION_SLOTS},), got {tuple(condition_view_ids.shape)}')
    target = canonical_p61_target_view(target_view)
    T_target_inv = se3_inverse(T_anchor_cam[:, target])
    T_cond = T_anchor_cam[:, condition_view_ids.to(T_anchor_cam.device)]
    T_rel = T_target_inv[:, :, None, None] @ T_cond[:, None]
    trans = T_rel[..., :3, 3]
    rot = _stable_so3_log_vector(T_rel[..., :3, :3]).to(device=trans.device, dtype=trans.dtype)
    features = normalize_p62_pose_features(torch.cat([trans, rot], dim=-1))
    return features if dtype is None else features.to(dtype=dtype)

def build_p62_target_pose_features(T_anchor_cam: torch.Tensor, target_view: int, dtype: torch.dtype | None=None) -> torch.Tensor:
    if T_anchor_cam.ndim != 5 or T_anchor_cam.shape[1] != 6 or T_anchor_cam.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_cam shape (B,6,T,4,4), got {tuple(T_anchor_cam.shape)}')
    target = canonical_p61_target_view(target_view)
    T_target = T_anchor_cam[:, target]
    trans = T_target[..., :3, 3]
    rot = _stable_so3_log_vector(T_target[..., :3, :3]).to(device=trans.device, dtype=trans.dtype)
    features = normalize_p62_pose_features(torch.cat([trans, rot], dim=-1))
    return features if dtype is None else features.to(dtype=dtype)

def training_step_ray_p62(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, target_view: int | None=None, front_condition_noise_prob: float=0.05, front_condition_noise_min: float=0.005, front_condition_noise_max: float=0.03, neighbor_condition_noise_prob: float=0.3, neighbor_condition_noise_min: float=0.02, neighbor_condition_noise_max: float=0.15, rear_loss_weight: float=1.0) -> RayTrainingStepP62Output:
    resolved_device = _resolve_batch_device(batch, device)
    target = canonical_p61_target_view(target_view) if target_view is not None else resolve_p61_target_view(resolved_device)
    layout = p61_graph_layout(target, device=resolved_device)
    needed_views = sorted({target, *[int(view_id) for (view_id, available) in zip(layout.condition_view_ids.detach().cpu().tolist(), layout.condition_available_mask.detach().cpu().tolist()) if bool(available) and int(view_id) != FRONT_IDX]})
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, needed_views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p61_streams_from_latents(z_all, sigma, target, front_condition_noise_prob=front_condition_noise_prob, front_condition_noise_min=front_condition_noise_min, front_condition_noise_max=front_condition_noise_max, neighbor_condition_noise_prob=neighbor_condition_noise_prob, neighbor_condition_noise_min=neighbor_condition_noise_min, neighbor_condition_noise_max=neighbor_condition_noise_max)
    if 'T_anchor_front' not in batch:
        raise KeyError("P6.2 training requires batch['T_anchor_front']")
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E_rig = batch['E'].to(device=resolved_device, dtype=torch.float32)
    T_anchor_front = batch['T_anchor_front'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    latent_frames = z_all.shape[3]
    with torch.no_grad():
        backbone_plucker = build_p61_dynamic_plucker_streams(K, E_rig, T_anchor_front, streams.layout.backbone_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_all.dtype)
        condition_plucker = build_p61_dynamic_plucker_streams(K, E_rig, T_anchor_front, streams.layout.condition_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_all.dtype)
        T_anchor_cam = build_p61_anchor_camera_transforms(E_rig, T_anchor_front)
        relative_pose_features = build_p62_time_window_relative_pose_features(T_anchor_cam, target, streams.layout.condition_view_ids, dtype=z_all.dtype)
        target_pose_features = build_p62_target_pose_features(T_anchor_cam, target, dtype=z_all.dtype)
    dit = components.low_dit
    if dit is None:
        raise ValueError('P6.2 requires components.low_dit')
    v_pred = dit(streams.z_backbone, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), backbone_plucker, streams.layout.backbone_view_ids, streams.layout.backbone_role_ids, T_anchor_front, streams.condition_latents, condition_plucker, streams.layout.condition_view_ids, streams.layout.condition_type_ids, streams.layout.condition_available_mask, relative_pose_features, target_pose_features)
    pred_target = v_pred[:, P61_TARGET_STREAM_INDEX:P61_TARGET_STREAM_INDEX + 1]
    unweighted_loss = ((pred_target - streams.v_target) ** 2).mean()
    target_loss_weight = float(rear_loss_weight) if int(target) == 5 else 1.0
    loss = unweighted_loss * target_loss_weight
    front_translation_norm = T_anchor_front[:, :latent_frames, :3, 3].norm(dim=-1)
    metrics: dict[str, torch.Tensor | str | list[int] | float | int] = {'loss': loss.detach(), 'unweighted_loss': unweighted_loss.detach(), 'target_loss_weight': target_loss_weight, 'stage': 'unified', 'training_method': 'p62_strong_geometry_vace_graph_autoregressive_single_target', 'target_view': int(target), 'condition_view_ids': [int(item) for item in streams.layout.condition_view_ids.detach().cpu().tolist()], 'condition_available_mask': [int(item) for item in streams.layout.condition_available_mask.detach().cpu().tolist()], 'backbone_view_ids': [int(item) for item in streams.layout.backbone_view_ids.detach().cpu().tolist()], 'sigma_mean': sigma.detach().float().mean(), 'sigma_min': sigma.detach().float().min(), 'sigma_max': sigma.detach().float().max(), 'front_condition_noise_sigma_mean': streams.front_condition_noise_sigma.detach().float().mean(), 'front_condition_noise_sigma_max': streams.front_condition_noise_sigma.detach().float().max(), 'neighbor_condition_noise_sigma_mean': streams.neighbor_condition_noise_sigma.detach().float().mean(), 'neighbor_condition_noise_sigma_max': streams.neighbor_condition_noise_sigma.detach().float().max(), 'front_translation_norm_mean': front_translation_norm.mean().detach(), 'front_translation_norm_max': front_translation_norm.max().detach(), 'front_rotation_angle_max': se3_log_rotation_angle(T_anchor_front[:, :latent_frames]).abs().max().detach(), 'front_vace_target_condition': 1}
    if isinstance(dit, DiTP62VACE):
        metrics['sync_temporal_window'] = int(dit.sync_temporal_window)
        metrics['graph_gate_init_bias'] = float(dit.graph_gate_init_bias)
        metrics['geo_head_dim'] = int(dit.geo_head_dim)
    else:
        sync_window = getattr(dit, 'sync_temporal_window', None)
        if sync_window is not None:
            metrics['sync_temporal_window'] = int(sync_window)
    metrics[f'flow_view_{target}'] = unweighted_loss.detach()
    metrics[f'weighted_flow_view_{target}'] = loss.detach()
    return RayTrainingStepP62Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
