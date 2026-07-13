from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from openlongtail.models.dit_p5_vace import P5_VACE_CONTEXT_CHANNELS
from openlongtail.models.dit_p61_vace import P61_BACKBONE_STREAMS, P61_TARGET_STREAM_INDEX
from openlongtail.models.dit_p63_vace import DiTP63VACE

class DiTVACEWarp(DiTP63VACE):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True, graph_gate_init_bias: float=-1.4, geo_head_dim: int=16, geo_projection_temperature: float=4.0) -> None:
        super().__init__(expert, dim=dim, dim_attn=dim_attn, heads=heads, cross_view_blocks=cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature)

    def build_vace_context_warp(self, z_backbone: torch.Tensor, warped_target_latent: torch.Tensor, warped_target_visibility: torch.Tensor) -> torch.Tensor:
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        (batch, _, _, frames, height, width) = z_backbone.shape
        if warped_target_latent.shape != (batch, 16, frames, height, width):
            raise ValueError(f'unexpected warped_target_latent shape {tuple(warped_target_latent.shape)}; expected ({batch}, 16, {frames}, {height}, {width})')
        if warped_target_visibility.shape != (batch, 1, frames, height, width):
            raise ValueError(f'unexpected warped_target_visibility shape {tuple(warped_target_visibility.shape)}; expected ({batch}, 1, {frames}, {height}, {width})')
        inactive = torch.zeros_like(z_backbone)
        reactive = torch.zeros_like(z_backbone)
        mask = torch.zeros(batch, P61_BACKBONE_STREAMS, 64, frames, height, width, device=z_backbone.device, dtype=z_backbone.dtype)
        inactive[:, 0] = z_backbone[:, 0]
        reactive[:, P61_TARGET_STREAM_INDEX] = warped_target_latent.to(z_backbone.dtype)
        vis = warped_target_visibility.to(device=z_backbone.device, dtype=z_backbone.dtype)
        mask[:, P61_TARGET_STREAM_INDEX] = vis.expand(batch, 64, frames, height, width)
        return torch.cat([inactive, reactive, mask], dim=2)

    def forward(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, target_pose_features: torch.Tensor, warped_target_latent: torch.Tensor, warped_target_visibility: torch.Tensor) -> torch.Tensor:
        if warped_target_latent is None or warped_target_visibility is None:
            raise ValueError("forward requires warped_target_latent and warped_target_visibility.")
        vace_context = self.build_vace_context_warp(z_backbone, warped_target_latent, warped_target_visibility)
        return DiTP63VACE.forward(self, z_backbone=z_backbone, sigma=sigma, text_emb=text_emb, text_mask=text_mask, backbone_plucker=backbone_plucker, backbone_view_ids=backbone_view_ids, backbone_role_ids=backbone_role_ids, T_anchor_front=T_anchor_front, condition_latents=condition_latents, condition_plucker=condition_plucker, condition_view_ids=condition_view_ids, condition_type_ids=condition_type_ids, condition_available_mask=condition_available_mask, relative_pose_features=relative_pose_features, target_pose_features=target_pose_features, vace_context=vace_context)
