from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from openlongtail.configs.openlongtail_style_vace import STYLE_AXIS_CARDINALITIES, STYLE_REAR_ONLY, STYLE_REAR_TARGET_VIEWS
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX
from openlongtail.models.dit_vace import DiTVACEWarp

class StyleConditioner(nn.Module):

    def __init__(self, axis_cardinalities: Sequence[int]=STYLE_AXIS_CARDINALITIES, dim: int=1536) -> None:
        super().__init__()
        self.axis_cardinalities = tuple((int(c) for c in axis_cardinalities))
        if any((c < 2 for c in self.axis_cardinalities)):
            raise ValueError(f'each style axis needs >=2 categories (incl. unknown), got {self.axis_cardinalities}')
        self.unknown_indices = tuple((c - 1 for c in self.axis_cardinalities))
        self.dim = int(dim)
        self.mod_dim = 6 * self.dim
        self.embeds = nn.ModuleList([nn.Embedding(c, self.mod_dim) for c in self.axis_cardinalities])
        for emb in self.embeds:
            nn.init.zeros_(emb.weight)

    @property
    def num_axes(self) -> int:
        return len(self.axis_cardinalities)

    def unknown_style_ids(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.unknown_indices, dtype=torch.long, device=device).view(1, -1).expand(batch, -1)

    def forward(self, style_ids: torch.Tensor) -> torch.Tensor:
        if style_ids.ndim != 2 or style_ids.shape[1] != self.num_axes:
            raise ValueError(f'expected style_ids shape (B, {self.num_axes}), got {tuple(style_ids.shape)}')
        device = self.embeds[0].weight.device
        ids = style_ids.to(device=device, dtype=torch.long)
        modulation: torch.Tensor | None = None
        for (axis, emb) in enumerate(self.embeds):
            col = ids[:, axis].clamp(0, self.axis_cardinalities[axis] - 1)
            part = emb(col)
            modulation = part if modulation is None else modulation + part
        assert modulation is not None
        return modulation

class DiTVACEStyle(DiTVACEWarp):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True, graph_gate_init_bias: float=-1.4, geo_head_dim: int=16, geo_projection_temperature: float=4.0, style_axis_cardinalities: Sequence[int]=STYLE_AXIS_CARDINALITIES, style_rear_only: bool=STYLE_REAR_ONLY, style_rear_target_views: Sequence[int]=STYLE_REAR_TARGET_VIEWS) -> None:
        super().__init__(expert, dim=dim, dim_attn=dim_attn, heads=heads, cross_view_blocks=cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature)
        self.style_rear_only = bool(style_rear_only)
        self.style_rear_target_views = tuple((int(v) for v in style_rear_target_views))
        self.style_conditioner = StyleConditioner(axis_cardinalities=style_axis_cardinalities, dim=self.dim)
        self._pending_style_ids: torch.Tensor | None = None

    def _style_modulation(self, style_ids: torch.Tensor) -> torch.Tensor:
        return self.style_conditioner(style_ids)

    def _target_pose_modulation(self, target_pose_features: torch.Tensor, backbone_view_ids: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = super()._target_pose_modulation(target_pose_features, backbone_view_ids, batch, device, dtype)
        style_ids = self._pending_style_ids
        if style_ids is None:
            return base
        style_mod = self._style_modulation(style_ids.to(device)).view(-1, 6, self.dim)
        if style_mod.shape[0] == 1 and batch > 1:
            style_mod = style_mod.expand(batch, -1, -1)
        if style_mod.shape[0] != batch:
            raise ValueError(f'expected style_ids batch {batch}, got {tuple(style_ids.shape)}')
        gate = 1.0
        if self.style_rear_only:
            target_view_id = int(backbone_view_ids[P61_TARGET_STREAM_INDEX].item())
            gate = 1.0 if target_view_id in self.style_rear_target_views else 0.0
        return base + (style_mod * gate).to(dtype=dtype)

    def forward(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, target_pose_features: torch.Tensor, warped_target_latent: torch.Tensor, warped_target_visibility: torch.Tensor, style_ids: torch.Tensor | None=None) -> torch.Tensor:
        previous_style_ids = self._pending_style_ids
        self._pending_style_ids = style_ids
        try:
            return super().forward(z_backbone=z_backbone, sigma=sigma, text_emb=text_emb, text_mask=text_mask, backbone_plucker=backbone_plucker, backbone_view_ids=backbone_view_ids, backbone_role_ids=backbone_role_ids, T_anchor_front=T_anchor_front, condition_latents=condition_latents, condition_plucker=condition_plucker, condition_view_ids=condition_view_ids, condition_type_ids=condition_type_ids, condition_available_mask=condition_available_mask, relative_pose_features=relative_pose_features, target_pose_features=target_pose_features, warped_target_latent=warped_target_latent, warped_target_visibility=warped_target_visibility)
        finally:
            self._pending_style_ids = previous_style_ids
