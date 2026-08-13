from __future__ import annotations
from dataclasses import dataclass, replace as dc_replace
from typing import Any
import torch
from openlongtail.data.transforms import se3_log_rotation_angle
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX, P61_TYPE_FRONT, P61_TYPE_PAD
from openlongtail.models.dit_vace import DiTVACEWarp
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import _prepare_z_all_for_p3, _resolve_batch_device
from openlongtail.training.forward_ray_p5 import FRONT_IDX
from openlongtail.training.forward_ray_p61 import P61GraphLayout, P61_TARGET_VIEWS, build_p61_anchor_camera_transforms, build_p61_dynamic_plucker_streams, build_p61_streams_from_latents, canonical_p61_target_view, p61_graph_layout, resolve_p61_target_view
from openlongtail.training.forward_ray_p62 import build_p62_target_pose_features, build_p62_time_window_relative_pose_features
from openlongtail.training.forward_wan21 import sample_sigma_unified
from openlongtail.training.schedulers import FlowMatchScheduler
WARP_TARGET_VIEWS = P61_TARGET_VIEWS

# --- component-ablation switches -------------------------------------------
# 'warp'  : VACE reactive slot gets the geometrically reprojected target latent
#           (the full method).
# 'naive' : reactive slot gets the front latent copied verbatim, visibility all
#           ones. The VACE conditioning pathway stays live and equally full, so
#           the delta against 'warp' isolates the REPROJECTION GEOMETRY rather
#           than measuring what happens when the backbone's context is disabled.
REACTIVE_MODES = ('warp', 'naive')
# 'proposed' : G(3)={0,1}, G(4)={0,2}, G(5)={0,3,4}
# 'star'     : G(i)={0} for every target -- the memory bank exists with an
#              identical parameter count and only ever attends to the front
#              view, so the delta isolates the graph topology itself.
GRAPH_MODES = ('proposed', 'star')

def star_graph_layout(layout: P61GraphLayout) -> P61GraphLayout:
    available = torch.zeros_like(layout.condition_available_mask)
    available[0] = True
    view_ids = torch.zeros_like(layout.condition_view_ids)
    view_ids[0] = layout.condition_view_ids[0]
    type_ids = torch.full_like(layout.condition_type_ids, P61_TYPE_PAD)
    type_ids[0] = P61_TYPE_FRONT
    return dc_replace(layout, condition_view_ids=view_ids, condition_type_ids=type_ids, condition_available_mask=available)

@dataclass
class RayTrainingStepOutputWarp:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int] | float | int]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def training_step_ray_warp(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, target_view: int | None=None, front_condition_noise_prob: float=0.05, front_condition_noise_min: float=0.005, front_condition_noise_max: float=0.03, neighbor_condition_noise_prob: float=0.3, neighbor_condition_noise_min: float=0.02, neighbor_condition_noise_max: float=0.15, rear_loss_weight: float=1.0, reactive_mode: str='warp', disable_plucker: bool=False, graph_mode: str='proposed') -> RayTrainingStepOutputWarp:
    if reactive_mode not in REACTIVE_MODES:
        raise ValueError(f'reactive_mode must be one of {REACTIVE_MODES}, got {reactive_mode!r}')
    if graph_mode not in GRAPH_MODES:
        raise ValueError(f'graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}')
    resolved_device = _resolve_batch_device(batch, device)
    target = canonical_p61_target_view(target_view) if target_view is not None else resolve_p61_target_view(resolved_device)
    layout = p61_graph_layout(target, device=resolved_device)
    if graph_mode == 'star':
        # masked before needed_views, so neighbour latents are never even loaded
        layout = star_graph_layout(layout)
    needed_views = sorted({target, *[int(view_id) for (view_id, available) in zip(layout.condition_view_ids.detach().cpu().tolist(), layout.condition_available_mask.detach().cpu().tolist()) if bool(available) and int(view_id) != FRONT_IDX]})
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, needed_views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p61_streams_from_latents(z_all, sigma, target, front_condition_noise_prob=front_condition_noise_prob, front_condition_noise_min=front_condition_noise_min, front_condition_noise_max=front_condition_noise_max, neighbor_condition_noise_prob=neighbor_condition_noise_prob, neighbor_condition_noise_min=neighbor_condition_noise_min, neighbor_condition_noise_max=neighbor_condition_noise_max, layout=layout)
    if 'T_anchor_front' not in batch:
        raise KeyError("P6.5 training requires batch['T_anchor_front']")
    if 'warped_target_latents' not in batch or 'warped_target_visibility' not in batch:
        raise KeyError("P6.5 training requires batch['warped_target_latents'] and batch['warped_target_visibility']")
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
        if disable_plucker:
            # Metric camera geometry only. View IDENTITY (cam_id/role/type
            # embeddings) is deliberately left intact -- without it the model
            # cannot know which of the five cameras to render and the baseline
            # would be degenerate rather than merely geometry-free.
            backbone_plucker = torch.zeros_like(backbone_plucker)
            condition_plucker = torch.zeros_like(condition_plucker)
            relative_pose_features = torch.zeros_like(relative_pose_features)
            target_pose_features = torch.zeros_like(target_pose_features)
    warped_all = batch['warped_target_latents'].to(device=resolved_device, dtype=z_all.dtype)
    vis_all = batch['warped_target_visibility'].to(device=resolved_device, dtype=z_all.dtype)
    if warped_all.shape[1] != 5 or vis_all.shape[1] != 5:
        raise ValueError(f'expected sidecar 5 target slots, got {tuple(warped_all.shape)}, {tuple(vis_all.shape)}')
    slot = int(target) - 1
    warped_target_latent = warped_all[:, slot]
    warped_target_visibility = vis_all[:, slot]
    if reactive_mode == 'naive':
        # Same slot, same shapes, fully populated -- only the reprojection is gone.
        warped_target_latent = z_all[:, FRONT_IDX].to(dtype=warped_target_latent.dtype)
        warped_target_visibility = torch.ones_like(warped_target_visibility)
    dit = components.low_dit
    if dit is None:
        raise ValueError('P6.5 requires components.low_dit')
    v_pred = dit(streams.z_backbone, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), backbone_plucker, streams.layout.backbone_view_ids, streams.layout.backbone_role_ids, T_anchor_front, streams.condition_latents, condition_plucker, streams.layout.condition_view_ids, streams.layout.condition_type_ids, streams.layout.condition_available_mask, relative_pose_features, target_pose_features, warped_target_latent=warped_target_latent, warped_target_visibility=warped_target_visibility)
    pred_target = v_pred[:, P61_TARGET_STREAM_INDEX:P61_TARGET_STREAM_INDEX + 1]
    unweighted_loss = ((pred_target - streams.v_target) ** 2).mean()
    target_loss_weight = float(rear_loss_weight) if int(target) == 5 else 1.0
    loss = unweighted_loss * target_loss_weight
    front_translation_norm = T_anchor_front[:, :latent_frames, :3, 3].norm(dim=-1)
    metrics: dict[str, torch.Tensor | str | list[int] | float | int] = {'loss': loss.detach(), 'unweighted_loss': unweighted_loss.detach(), 'target_loss_weight': target_loss_weight, 'stage': 'unified', 'training_method': 'depth_warp_condition_vace_graph_autoregressive_single_target', 'target_view': int(target), 'condition_view_ids': [int(item) for item in streams.layout.condition_view_ids.detach().cpu().tolist()], 'condition_available_mask': [int(item) for item in streams.layout.condition_available_mask.detach().cpu().tolist()], 'backbone_view_ids': [int(item) for item in streams.layout.backbone_view_ids.detach().cpu().tolist()], 'sigma_mean': sigma.detach().float().mean(), 'sigma_min': sigma.detach().float().min(), 'sigma_max': sigma.detach().float().max(), 'front_condition_noise_sigma_mean': streams.front_condition_noise_sigma.detach().float().mean(), 'front_condition_noise_sigma_max': streams.front_condition_noise_sigma.detach().float().max(), 'neighbor_condition_noise_sigma_mean': streams.neighbor_condition_noise_sigma.detach().float().mean(), 'neighbor_condition_noise_sigma_max': streams.neighbor_condition_noise_sigma.detach().float().max(), 'front_translation_norm_mean': front_translation_norm.mean().detach(), 'front_translation_norm_max': front_translation_norm.max().detach(), 'front_rotation_angle_max': se3_log_rotation_angle(T_anchor_front[:, :latent_frames]).abs().max().detach(), 'warp_visibility_mean': warped_target_visibility.float().mean().detach(), 'front_vace_target_condition': 0, 'warp_target_condition': 1}
    if isinstance(dit, DiTVACEWarp):
        metrics['sync_temporal_window'] = int(dit.sync_temporal_window)
        metrics['graph_gate_init_bias'] = float(dit.graph_gate_init_bias)
        metrics['geo_head_dim'] = int(dit.geo_head_dim)
        metrics['geo_projection_temperature'] = float(dit.geo_projection_temperature)
    # logged so each ablation arm is auditable from the training log alone:
    # plucker_abs_mean == 0 under --disable-plucker, warp_visibility_mean == 1
    # under --reactive-mode naive, condition_available_mask == [1,0,0] under star.
    metrics['plucker_abs_mean'] = backbone_plucker.detach().float().abs().mean()
    metrics['target_pose_abs_mean'] = target_pose_features.detach().float().abs().mean()
    metrics['ablation_reactive_mode'] = reactive_mode
    metrics['ablation_disable_plucker'] = int(bool(disable_plucker))
    metrics['ablation_graph_mode'] = graph_mode
    metrics[f'flow_view_{target}'] = unweighted_loss.detach()
    metrics[f'weighted_flow_view_{target}'] = loss.detach()
    return RayTrainingStepOutputWarp(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
