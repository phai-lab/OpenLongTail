from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.configs.openlongtail_register_vace import REGISTER_CAPTION_DIM, REGISTER_STYLE_AXIS_CARDINALITIES, REGISTER_STYLE_NUM_REGISTERS, REGISTER_STYLE_REAR_ONLY, REGISTER_STYLE_REAR_TARGET_VIEWS
from openlongtail.models.dit_p5_vace import P5_VACE_CONTEXT_CHANNELS
from openlongtail.models.dit_p61_vace import P61_BACKBONE_STREAMS, P61_TARGET_STREAM_INDEX
from openlongtail.models.dit_style_vace import DiTVACEStyle

class StyleRegisterBank(nn.Module):

    def __init__(self, axis_cardinalities: Sequence[int]=REGISTER_STYLE_AXIS_CARDINALITIES, dim: int=1536, num_registers: int=REGISTER_STYLE_NUM_REGISTERS, caption_dim: int=REGISTER_CAPTION_DIM, heads: int=12) -> None:
        super().__init__()
        self.axis_cardinalities = tuple((int(c) for c in axis_cardinalities))
        if any((c < 2 for c in self.axis_cardinalities)):
            raise ValueError(f'each style axis needs >=2 categories (incl. unknown), got {self.axis_cardinalities}')
        self.unknown_indices = tuple((c - 1 for c in self.axis_cardinalities))
        self.dim = int(dim)
        self.num_registers = int(num_registers)
        self.caption_dim = int(caption_dim)
        if self.dim % int(heads) != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={self.dim}, heads={heads}')
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.registers = nn.Parameter(torch.randn(1, self.num_registers, self.dim) * 0.02)
        self.axis_embeds = nn.ModuleList([nn.Embedding(c, self.dim) for c in self.axis_cardinalities])
        for emb in self.axis_embeds:
            nn.init.normal_(emb.weight, std=0.02)
        self.caption_proj = nn.Linear(self.caption_dim, self.dim)
        self.caption_ln = nn.LayerNorm(self.dim)
        self.cap_to_q = nn.Linear(self.dim, self.dim)
        self.cap_to_k = nn.Linear(self.dim, self.dim)
        self.cap_to_v = nn.Linear(self.dim, self.dim)
        self.cap_to_o = nn.Linear(self.dim, self.dim)
        self.composer = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim), nn.GELU(approximate='tanh'), nn.Linear(self.dim, self.dim))
        nn.init.zeros_(self.composer[-1].weight)
        nn.init.zeros_(self.composer[-1].bias)

    @property
    def num_axes(self) -> int:
        return len(self.axis_cardinalities)

    def unknown_style_ids(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.unknown_indices, dtype=torch.long, device=device).view(1, -1).expand(batch, -1)

    def _discrete(self, style_ids: torch.Tensor, batch: int, device, dtype) -> torch.Tensor:
        if style_ids.ndim != 2 or style_ids.shape[1] != self.num_axes:
            raise ValueError(f'expected style_ids shape (B, {self.num_axes}), got {tuple(style_ids.shape)}')
        ids = style_ids.to(device=device, dtype=torch.long)
        out: torch.Tensor | None = None
        for (axis, emb) in enumerate(self.axis_embeds):
            col = ids[:, axis].clamp(0, self.axis_cardinalities[axis] - 1)
            part = emb(col)
            out = part if out is None else out + part
        assert out is not None
        if out.shape[0] == 1 and batch > 1:
            out = out.expand(batch, -1)
        return out.to(dtype=dtype)

    def _caption(self, query: torch.Tensor, caption_emb: torch.Tensor, caption_mask: torch.Tensor | None) -> torch.Tensor:
        (batch, klen, _) = query.shape
        if caption_emb.ndim != 3 or caption_emb.shape[-1] != self.caption_dim:
            raise ValueError(f'expected caption_emb shape (B, L, {self.caption_dim}), got {tuple(caption_emb.shape)}')
        cap = self.caption_ln(self.caption_proj(caption_emb.to(dtype=query.dtype)))
        length = cap.shape[1]
        q = self.cap_to_q(query).view(batch, klen, self.heads, self.head_dim).transpose(1, 2)
        k = self.cap_to_k(cap).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.cap_to_v(cap).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        row_has_any = None
        if caption_mask is not None:
            valid = caption_mask.to(device=query.device, dtype=torch.bool)
            if valid.shape != (batch, length):
                raise ValueError(f'expected caption_mask shape {(batch, length)}, got {tuple(valid.shape)}')
            row_has_any = valid.any(dim=1)
            safe_valid = valid.clone()
            safe_valid[~row_has_any] = True
            attn_mask = torch.zeros(batch, 1, 1, length, device=query.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~safe_valid[:, None, None, :], float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, klen, self.dim)
        out = self.cap_to_o(out)
        if row_has_any is not None:
            out = out * row_has_any.view(batch, 1, 1).to(dtype=out.dtype)
        return out

    def forward(self, style_ids: torch.Tensor | None, caption_emb: torch.Tensor | None=None, caption_mask: torch.Tensor | None=None, batch: int | None=None) -> torch.Tensor:
        device = self.registers.device
        dtype = self.registers.dtype
        if batch is None:
            if style_ids is not None:
                batch = int(style_ids.shape[0])
            elif caption_emb is not None:
                batch = int(caption_emb.shape[0])
            else:
                batch = 1
        base = self.registers.to(device=device, dtype=dtype).expand(batch, -1, -1)
        fused = base
        if style_ids is not None:
            fused = fused + self._discrete(style_ids, batch, device, dtype)[:, None, :]
        if caption_emb is not None:
            fused = fused + self._caption(fused, caption_emb.to(device=device), caption_mask)
        return self.composer(fused)

class StyleCrossAttention(nn.Module):

    def __init__(self, dim: int, heads: int, num_views: int=6, rear_target_views: Sequence[int]=REGISTER_STYLE_REAR_TARGET_VIEWS, view_gate_on_bias: float=3.0, view_gate_off_bias: float=-3.0, sigma_gate_init_bias: float=0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.num_views = int(num_views)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_o = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_o.weight)
        nn.init.zeros_(self.to_o.bias)
        rear = {int(v) for v in rear_target_views}
        view_gate = torch.full((self.num_views,), float(view_gate_off_bias))
        for v in rear:
            if 0 <= v < self.num_views:
                view_gate[v] = float(view_gate_on_bias)
        self.view_gate = nn.Parameter(view_gate)
        self.sigma_gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.sigma_gate[-1].weight)
        nn.init.constant_(self.sigma_gate[-1].bias, float(sigma_gate_init_bias))

    def _view_gate_scalar(self, target_view_id: int | None) -> torch.Tensor:
        gate = torch.sigmoid(self.view_gate)
        if target_view_id is None:
            return gate.mean()
        onehot = F.one_hot(torch.tensor(int(target_view_id), device=self.view_gate.device), self.num_views).to(dtype=gate.dtype)
        return (gate * onehot).sum()

    def _sigma_gate_scalar(self, sigma: torch.Tensor) -> torch.Tensor:
        w_dtype = self.sigma_gate[-1].weight.dtype
        val = torch.sigmoid(self.sigma_gate(sigma.float().view(-1, 1).to(dtype=w_dtype)))
        return val.view(-1, 1, 1)

    def forward(self, target_tokens: torch.Tensor, style_bank: torch.Tensor, sigma: torch.Tensor, target_view_id: int | None=None) -> torch.Tensor:
        if target_tokens.ndim != 3:
            raise ValueError(f'expected target_tokens shape (B,N,dim), got {tuple(target_tokens.shape)}')
        if style_bank.ndim != 3 or style_bank.shape[-1] != self.dim:
            raise ValueError(f'expected style_bank shape (B,K,{self.dim}), got {tuple(style_bank.shape)}')
        (batch, n_tok, _) = target_tokens.shape
        if style_bank.shape[0] != batch:
            style_bank = style_bank.expand(batch, -1, -1)
        klen = style_bank.shape[1]
        style_bank = style_bank.to(device=target_tokens.device, dtype=target_tokens.dtype)
        q = self.to_q(target_tokens).view(batch, n_tok, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(style_bank).view(batch, klen, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(style_bank).view(batch, klen, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, n_tok, self.dim)
        out = self.to_o(out)
        g_view = self._view_gate_scalar(target_view_id).to(dtype=out.dtype)
        g_sigma = self._sigma_gate_scalar(sigma.to(out.device)).to(dtype=out.dtype)
        return g_view * g_sigma * out

class DiTVACERegister(DiTVACEStyle):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True, graph_gate_init_bias: float=-1.4, geo_head_dim: int=16, geo_projection_temperature: float=4.0, style_axis_cardinalities: Sequence[int]=REGISTER_STYLE_AXIS_CARDINALITIES, style_rear_only: bool=REGISTER_STYLE_REAR_ONLY, style_rear_target_views: Sequence[int]=REGISTER_STYLE_REAR_TARGET_VIEWS, style_num_registers: int=REGISTER_STYLE_NUM_REGISTERS, caption_dim: int=REGISTER_CAPTION_DIM) -> None:
        super().__init__(expert, dim=dim, dim_attn=dim_attn, heads=heads, cross_view_blocks=cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature, style_axis_cardinalities=style_axis_cardinalities, style_rear_only=style_rear_only, style_rear_target_views=style_rear_target_views)
        self.style_bank = StyleRegisterBank(axis_cardinalities=style_axis_cardinalities, dim=self.dim, num_registers=style_num_registers, caption_dim=caption_dim, heads=heads)
        self.style_attn = nn.ModuleList([StyleCrossAttention(self.dim, heads, rear_target_views=self.style_rear_target_views) for _ in self.cross_view_blocks])

    def forward(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, target_pose_features: torch.Tensor, warped_target_latent: torch.Tensor, warped_target_visibility: torch.Tensor, style_ids: torch.Tensor | None=None, caption_emb: torch.Tensor | None=None, caption_mask: torch.Tensor | None=None) -> torch.Tensor:
        if style_ids is None and caption_emb is None:
            return super().forward(z_backbone=z_backbone, sigma=sigma, text_emb=text_emb, text_mask=text_mask, backbone_plucker=backbone_plucker, backbone_view_ids=backbone_view_ids, backbone_role_ids=backbone_role_ids, T_anchor_front=T_anchor_front, condition_latents=condition_latents, condition_plucker=condition_plucker, condition_view_ids=condition_view_ids, condition_type_ids=condition_type_ids, condition_available_mask=condition_available_mask, relative_pose_features=relative_pose_features, target_pose_features=target_pose_features, warped_target_latent=warped_target_latent, warped_target_visibility=warped_target_visibility, style_ids=None)
        vace_context = self.build_vace_context_warp(z_backbone, warped_target_latent, warped_target_visibility)
        style_bank = self.style_bank(style_ids, caption_emb=caption_emb, caption_mask=caption_mask, batch=z_backbone.shape[0])
        previous_style_ids = self._pending_style_ids
        self._pending_style_ids = style_ids
        try:
            return self._forward_coupled(z_backbone=z_backbone, sigma=sigma, text_emb=text_emb, backbone_plucker=backbone_plucker, backbone_view_ids=backbone_view_ids, backbone_role_ids=backbone_role_ids, T_anchor_front=T_anchor_front, condition_latents=condition_latents, condition_plucker=condition_plucker, condition_view_ids=condition_view_ids, condition_type_ids=condition_type_ids, condition_available_mask=condition_available_mask, relative_pose_features=relative_pose_features, target_pose_features=target_pose_features, vace_context=vace_context, style_bank=style_bank)
        finally:
            self._pending_style_ids = previous_style_ids

    def _forward_coupled(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, target_pose_features: torch.Tensor, vace_context: torch.Tensor, style_bank: torch.Tensor) -> torch.Tensor:
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        batch = z_backbone.shape[0]
        if backbone_view_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_view_ids shape (2,), got {tuple(backbone_view_ids.shape)}')
        if backbone_role_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_role_ids shape (2,), got {tuple(backbone_role_ids.shape)}')
        if vace_context.ndim != 6 or vace_context.shape[1] != P61_BACKBONE_STREAMS or vace_context.shape[2] != P5_VACE_CONTEXT_CHANNELS:
            raise ValueError(f'expected vace_context shape (B,2,96,T,H,W), got {tuple(vace_context.shape)}')
        target_view_id = int(backbone_view_ids[P61_TARGET_STREAM_INDEX].item())
        (dense_memory, semantic_memory) = self._condition_bank_memory(condition_latents, condition_plucker, condition_view_ids, condition_type_ids, condition_available_mask, T_anchor_front)
        (hidden, grid_sizes, seq_lens) = self._patchify_base(z_backbone)
        cond_emb = self._backbone_embedding(backbone_plucker.to(hidden.device, hidden.dtype), backbone_view_ids.to(hidden.device), backbone_role_ids.to(hidden.device), T_anchor_front.to(hidden.device), batch)
        hidden = hidden + cond_emb
        vace_hidden = self._patchify_vace_context(vace_context.to(hidden.device, hidden.dtype)) + cond_emb
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time(sigma.to(hidden.device, hidden.dtype), batch, hidden.device)
        pose_mod = self._target_pose_modulation(target_pose_features, backbone_view_ids, batch, hidden.device, e0.dtype)
        e0_view = e0.view(batch, P61_BACKBONE_STREAMS, 6, self.dim)
        e0_view[:, P61_TARGET_STREAM_INDEX] = e0_view[:, P61_TARGET_STREAM_INDEX] + pose_mod
        e0 = e0_view.reshape(batch * P61_BACKBONE_STREAMS, 6, self.dim)
        context = self._prepare_context(text_emb.to(hidden.device, hidden.dtype), batch)
        freqs = self.expert.freqs.to(hidden.device) if hasattr(self.expert, 'freqs') else None
        autocast_enabled = hidden.device.type == 'cuda'
        base_kwargs = {'e': e0, 'seq_lens': seq_lens, 'grid_sizes': grid_sizes, 'freqs': freqs, 'context': context, 'context_lens': None}
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hints = self._forward_vace_hints(vace_hidden, hidden, seq_len, base_kwargs)
        target_tokens_per_stream = hidden.shape[1]
        frames = backbone_plucker.shape[2]
        tokens_per_frame = target_tokens_per_stream // frames
        target_plucker = backbone_plucker[:, P61_TARGET_STREAM_INDEX]
        style_bank = style_bank.to(device=hidden.device, dtype=hidden.dtype)
        cva_idx = 0
        for (block_idx, block) in enumerate(self.expert.blocks):
            kwargs = dict(base_kwargs)
            kwargs['hints'] = hints
            kwargs['context_scale'] = 1.0
            with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(lambda x, b=block, kw=kwargs: b(x, **kw), hidden, use_reentrant=self.use_reentrant)
                else:
                    hidden = block(hidden, **kwargs)
            if block_idx in self.cross_view_blocks:
                target_hidden = hidden.view(batch, P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)[:, P61_TARGET_STREAM_INDEX]
                target_frames = target_hidden.reshape(batch, frames, tokens_per_frame, self.dim)
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    if self.gradient_checkpointing and self.training:
                        delta = checkpoint(lambda tgt, dense, sem, tgt_geo, cond_geo, rel, avail, sig, module=self.graph_memory[cva_idx]: module(tgt, dense, sem, tgt_geo, cond_geo, rel, avail, sig), target_frames, dense_memory.to(hidden.device, hidden.dtype), semantic_memory.to(hidden.device, hidden.dtype), target_plucker.to(hidden.device, hidden.dtype), condition_plucker.to(hidden.device, hidden.dtype), relative_pose_features.to(hidden.device, hidden.dtype), condition_available_mask.to(hidden.device), sigma.to(hidden.device), use_reentrant=self.use_reentrant)
                        style_delta = checkpoint(lambda tgt, sb, sig, module=self.style_attn[cva_idx], vid=target_view_id: module(tgt, sb, sig, vid), target_hidden, style_bank, sigma.to(hidden.device), use_reentrant=self.use_reentrant)
                    else:
                        delta = self.graph_memory[cva_idx](target_frames, dense_memory.to(hidden.device, hidden.dtype), semantic_memory.to(hidden.device, hidden.dtype), target_plucker.to(hidden.device, hidden.dtype), condition_plucker.to(hidden.device, hidden.dtype), relative_pose_features.to(hidden.device, hidden.dtype), condition_available_mask.to(hidden.device), sigma.to(hidden.device))
                        style_delta = self.style_attn[cva_idx](target_hidden, style_bank, sigma.to(hidden.device), target_view_id)
                hidden_view = hidden.view(batch, P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                front_stream = hidden_view[:, 0]
                target_stream = hidden_view[:, P61_TARGET_STREAM_INDEX] + delta + style_delta
                hidden = torch.stack([front_stream, target_stream], dim=1).reshape(batch * P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                cva_idx += 1
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, P61_BACKBONE_STREAMS, *outputs[0].shape)
