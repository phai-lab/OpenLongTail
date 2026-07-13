from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.models.dit_p5_vace import P5_VACE_CONTEXT_CHANNELS
from openlongtail.models.dit_p61_vace import P61_BACKBONE_STREAMS, P61_CONDITION_SLOTS, P61_TARGET_STREAM_INDEX, DiTP61VACE

class P62SigmaGate(nn.Module):

    def __init__(self, init_bias: float=-2.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.sigmoid(self.net(sigma.float().view(-1, 1).to(dtype=self.net[-1].weight.dtype)))
        return (values[:, 0].view(-1, 1, 1), values[:, 1].view(-1, 1, 1))

class GeometrySDPACrossAttention(nn.Module):

    def __init__(self, dim: int, heads: int, query_geo_dim: int, memory_geo_dim: int, geo_head_dim: int=16, zero_init_out: bool=False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.geo_head_dim = int(geo_head_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q_geo = nn.Linear(query_geo_dim, heads * self.geo_head_dim)
        self.to_k_geo = nn.Linear(memory_geo_dim, heads * self.geo_head_dim)
        self.geo_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.geo_input_clip = 256.0
        self.geo_projection_temperature = 8.0
        self.out = nn.Linear(dim, dim)
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, query_geo: torch.Tensor, memory_geo: torch.Tensor, memory_available: torch.Tensor | None=None) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError(f'expected query/memory rank 3, got {tuple(query.shape)}, {tuple(memory.shape)}')
        if query_geo.shape[:2] != query.shape[:2]:
            raise ValueError(f'expected query_geo prefix {query.shape[:2]}, got {tuple(query_geo.shape)}')
        if memory_geo.shape[:2] != memory.shape[:2]:
            raise ValueError(f'expected memory_geo prefix {memory.shape[:2]}, got {tuple(memory_geo.shape)}')
        (batch, q_len, _) = query.shape
        if memory.shape[0] != batch:
            raise ValueError(f'expected memory batch {batch}, got {memory.shape[0]}')
        q = self.to_q(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        query_geo = query_geo.to(device=query.device, dtype=query.dtype)
        memory_geo = memory_geo.to(device=query.device, dtype=query.dtype)
        query_geo = torch.nan_to_num(query_geo, nan=0.0, posinf=self.geo_input_clip, neginf=-self.geo_input_clip).clamp(-self.geo_input_clip, self.geo_input_clip)
        memory_geo = torch.nan_to_num(memory_geo, nan=0.0, posinf=self.geo_input_clip, neginf=-self.geo_input_clip).clamp(-self.geo_input_clip, self.geo_input_clip)
        q_geo = self.to_q_geo(query_geo)
        k_geo = self.to_k_geo(memory_geo)
        q_geo = torch.tanh(q_geo.float() / self.geo_projection_temperature).to(dtype=query.dtype)
        k_geo = torch.tanh(k_geo.float() / self.geo_projection_temperature).to(dtype=query.dtype)
        q_geo = q_geo.view(batch, q_len, self.heads, self.geo_head_dim).transpose(1, 2)
        k_geo = k_geo.view(batch, memory.shape[1], self.heads, self.geo_head_dim).transpose(1, 2)
        scale = self.geo_logit_scale.to(device=query.device, dtype=query.dtype).clamp(0.0, 4.0)
        q = torch.cat([q, scale * q_geo], dim=-1)
        k = torch.cat([k, scale * k_geo], dim=-1)
        attn_mask = None
        if memory_available is not None:
            if memory_available.shape != (batch, memory.shape[1]):
                raise ValueError(f'expected memory_available shape {(batch, memory.shape[1])}, got {tuple(memory_available.shape)}')
            attn_mask = torch.zeros(batch, 1, 1, memory.shape[1], device=query.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~memory_available[:, None, None, :].to(torch.bool), float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, q_len, self.dim)
        return self.out(out)

class P62GraphMemoryAttention(nn.Module):

    def __init__(self, dim: int, heads: int=16, temporal_window: int=2, relative_pose_hidden: int=256, graph_gate_init_bias: float=-2.0, geo_head_dim: int=16) -> None:
        super().__init__()
        if temporal_window < 0:
            raise ValueError(f'expected temporal_window >= 0, got {temporal_window}')
        self.dim = int(dim)
        self.temporal_window = int(temporal_window)
        self.relative_pose_mlp = nn.Sequential(nn.Linear(6, relative_pose_hidden), nn.SiLU(), nn.Linear(relative_pose_hidden, dim))
        self.temporal_offset_embed = nn.Embedding(2 * temporal_window + 1, dim)
        self.semantic_attn = GeometrySDPACrossAttention(dim, heads, query_geo_dim=6, memory_geo_dim=13, geo_head_dim=geo_head_dim)
        self.dense_attn = GeometrySDPACrossAttention(dim, heads, query_geo_dim=6, memory_geo_dim=13, geo_head_dim=geo_head_dim)
        self.gate = P62SigmaGate(init_bias=graph_gate_init_bias)
        self.last_window_token_count: int | None = None

    @staticmethod
    def _target_query_geo(target_plucker: torch.Tensor, frame_idx: int, dtype: torch.dtype) -> torch.Tensor:
        query_geo = target_plucker[:, frame_idx].permute(0, 2, 3, 1).reshape(target_plucker.shape[0], -1, 6)
        return query_geo.to(dtype=dtype)

    def _window_memory(self, memory: torch.Tensor, condition_plucker: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, frame_idx: int, semantic: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (batch, slots, frames, tokens, dim) = memory.shape
        lo = max(0, frame_idx - self.temporal_window)
        hi = min(frames, frame_idx + self.temporal_window + 1)
        mem = memory[:, :, lo:hi]
        rel = relative_pose_features[:, frame_idx, :, lo:hi]
        rel_emb = self.relative_pose_mlp(rel.to(device=memory.device, dtype=memory.dtype))
        offsets = torch.arange(lo, hi, device=memory.device) - frame_idx + self.temporal_window
        off_emb = self.temporal_offset_embed(offsets).to(dtype=memory.dtype)
        rel_emb = rel_emb + off_emb.view(1, 1, hi - lo, dim)
        mem = mem + rel_emb.unsqueeze(3)
        if semantic:
            cond_geo = condition_plucker[:, :, lo:hi].mean(dim=(-1, -2))
            cond_geo = cond_geo[:, :, :, None, :].expand(batch, slots, hi - lo, tokens, 6)
        else:
            cond_geo = condition_plucker[:, :, lo:hi].permute(0, 1, 2, 4, 5, 3).reshape(batch, slots, hi - lo, tokens, 6)
        rel_geo = rel[:, :, :, None, :].expand(batch, slots, hi - lo, tokens, 6)
        denom = max(1, self.temporal_window)
        offset_geo = (torch.arange(lo, hi, device=memory.device, dtype=memory.dtype) - frame_idx) / float(denom)
        offset_geo = offset_geo.view(1, 1, hi - lo, 1, 1).expand(batch, slots, hi - lo, tokens, 1)
        geo = torch.cat([cond_geo.to(memory.dtype), rel_geo.to(memory.dtype), offset_geo], dim=-1)
        mem = mem.reshape(batch, slots * (hi - lo) * tokens, dim)
        geo = geo.reshape(batch, slots * (hi - lo) * tokens, 13)
        valid = condition_available_mask.to(device=memory.device, dtype=torch.bool)
        valid = valid.view(1, slots, 1, 1).expand(batch, slots, hi - lo, tokens)
        valid = valid.reshape(batch, slots * (hi - lo) * tokens)
        self.last_window_token_count = int(mem.shape[1])
        return (mem, valid, geo)

    def forward(self, target_tokens: torch.Tensor, dense_memory: torch.Tensor, semantic_memory: torch.Tensor, target_plucker: torch.Tensor, condition_plucker: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if target_tokens.ndim != 4:
            raise ValueError(f'expected target_tokens shape (B,T,N,D), got {tuple(target_tokens.shape)}')
        (batch, frames, tokens, dim) = target_tokens.shape
        if dense_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected dense_memory shape {tuple(dense_memory.shape)}')
        if semantic_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected semantic_memory shape {tuple(semantic_memory.shape)}')
        if target_plucker.shape[:3] != (batch, frames, 6):
            raise ValueError(f'unexpected target_plucker shape {tuple(target_plucker.shape)}')
        if condition_plucker.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected condition_plucker shape {tuple(condition_plucker.shape)}')
        if target_plucker.shape[-2] * target_plucker.shape[-1] != tokens:
            raise ValueError(f'target plucker token count does not match target tokens: {target_plucker.shape[-2]}*{target_plucker.shape[-1]} vs {tokens}')
        if relative_pose_features.shape[:4] != (batch, frames, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected relative_pose_features shape {tuple(relative_pose_features.shape)}')
        if condition_available_mask.shape != (P61_CONDITION_SLOTS,):
            raise ValueError(f'expected condition_available_mask shape ({P61_CONDITION_SLOTS},), got {tuple(condition_available_mask.shape)}')
        (g_sem, g_dense) = self.gate(sigma.to(target_tokens.device))
        delta = torch.zeros_like(target_tokens)
        target_plucker = target_plucker.to(device=target_tokens.device, dtype=target_tokens.dtype)
        condition_plucker = condition_plucker.to(device=target_tokens.device, dtype=target_tokens.dtype)
        for frame_idx in range(frames):
            query = target_tokens[:, frame_idx]
            query_geo = self._target_query_geo(target_plucker, frame_idx, query.dtype)
            (sem, sem_valid, sem_geo) = self._window_memory(semantic_memory, condition_plucker, relative_pose_features, condition_available_mask, frame_idx, semantic=True)
            (dense, dense_valid, dense_geo) = self._window_memory(dense_memory, condition_plucker, relative_pose_features, condition_available_mask, frame_idx, semantic=False)
            sem_out = self.semantic_attn(query, sem, query_geo, sem_geo, sem_valid)
            dense_out = self.dense_attn(query, dense, query_geo, dense_geo, dense_valid)
            delta[:, frame_idx] = g_sem.to(dtype=query.dtype) * sem_out + g_dense.to(dtype=query.dtype) * dense_out
        return delta.reshape(batch, frames * tokens, dim)

class DiTP62VACE(DiTP61VACE):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True, graph_gate_init_bias: float=-2.0, geo_head_dim: int=16) -> None:
        super().__init__(expert, dim=dim, dim_attn=dim_attn, heads=heads, cross_view_blocks=cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding)
        self.graph_gate_init_bias = float(graph_gate_init_bias)
        self.geo_head_dim = int(geo_head_dim)
        self.graph_memory = nn.ModuleList([P62GraphMemoryAttention(self.dim, heads=heads, temporal_window=self.sync_temporal_window, graph_gate_init_bias=self.graph_gate_init_bias, geo_head_dim=self.geo_head_dim) for _ in self.cross_view_blocks])
        self.target_pose_mlp = nn.Sequential(nn.Linear(6, self.dim), nn.SiLU(), nn.Linear(self.dim, 6 * self.dim))
        self.target_view_mod_embed = nn.Embedding(6, 6 * self.dim)
        nn.init.zeros_(self.target_pose_mlp[-1].weight)
        nn.init.zeros_(self.target_pose_mlp[-1].bias)
        nn.init.normal_(self.target_view_mod_embed.weight, std=0.02)

    def build_vace_context(self, z_backbone: torch.Tensor) -> torch.Tensor:
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        (batch, _, _, frames, height, width) = z_backbone.shape
        inactive = torch.zeros_like(z_backbone)
        reactive = torch.zeros_like(z_backbone)
        mask = torch.zeros(batch, P61_BACKBONE_STREAMS, 64, frames, height, width, device=z_backbone.device, dtype=z_backbone.dtype)
        front = z_backbone[:, 0]
        inactive[:, 0] = front
        reactive[:, P61_TARGET_STREAM_INDEX] = front
        mask[:, P61_TARGET_STREAM_INDEX] = 1.0
        return torch.cat([inactive, reactive, mask], dim=2)

    def _target_pose_modulation(self, target_pose_features: torch.Tensor, backbone_view_ids: torch.Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if target_pose_features.ndim != 3 or target_pose_features.shape[0] != batch or target_pose_features.shape[-1] != 6:
            raise ValueError(f'expected target_pose_features shape (B,T,6), got {tuple(target_pose_features.shape)}')
        module_dtype = self.target_pose_mlp[-1].weight.dtype
        pooled = target_pose_features.to(device=device, dtype=module_dtype).mean(dim=1)
        pose_mod = self.target_pose_mlp(pooled).view(batch, 6, self.dim)
        target_view_id = backbone_view_ids[P61_TARGET_STREAM_INDEX].to(device=device).view(1).expand(batch)
        view_mod = self.target_view_mod_embed(target_view_id).to(dtype=module_dtype).view(batch, 6, self.dim)
        return (pose_mod + view_mod).to(dtype=dtype)

    def forward(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, target_pose_features: torch.Tensor, vace_context: torch.Tensor | None=None) -> torch.Tensor:
        del text_mask
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        batch = z_backbone.shape[0]
        if backbone_view_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_view_ids shape (2,), got {tuple(backbone_view_ids.shape)}')
        if backbone_role_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_role_ids shape (2,), got {tuple(backbone_role_ids.shape)}')
        if vace_context is None:
            vace_context = self.build_vace_context(z_backbone)
        if vace_context.ndim != 6 or vace_context.shape[1] != P61_BACKBONE_STREAMS or vace_context.shape[2] != P5_VACE_CONTEXT_CHANNELS:
            raise ValueError(f'expected vace_context shape (B,2,96,T,H,W), got {tuple(vace_context.shape)}')
        (dense_memory, semantic_memory) = self._condition_bank_memory(condition_latents, condition_plucker, condition_view_ids, condition_type_ids, condition_available_mask, T_anchor_front)
        (hidden, grid_sizes, seq_lens) = self._patchify_base(z_backbone)
        pre_cond_hidden_norm = self._debug_norm(hidden) if self.collect_geometry_debug else 0.0
        cond_emb = self._backbone_embedding(backbone_plucker.to(hidden.device, hidden.dtype), backbone_view_ids.to(hidden.device), backbone_role_ids.to(hidden.device), T_anchor_front.to(hidden.device), batch)
        hidden = hidden + cond_emb
        post_cond_hidden_norm = self._debug_norm(hidden) if self.collect_geometry_debug else 0.0
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
        debug_record: dict[str, object] | None = None
        if self.collect_geometry_debug:
            debug_record = {'sigma': float(sigma.detach().float().mean().cpu()), 'backbone_view_ids': [int(item) for item in backbone_view_ids.detach().cpu().tolist()], 'condition_view_ids': [int(item) for item in condition_view_ids.detach().cpu().tolist()], 'condition_available_mask': [int(item) for item in condition_available_mask.detach().cpu().to(torch.long).tolist()], 'pre_cond_hidden_norm': pre_cond_hidden_norm, 'post_cond_hidden_norm': post_cond_hidden_norm, 'cond_emb_to_hidden_ratio': self._debug_norm(cond_emb) / max(pre_cond_hidden_norm, 1e-12), 'target_pose_mod_norm': self._debug_norm(pose_mod), **self._last_backbone_embedding_debug, **self._last_condition_memory_debug, 'graph_blocks': []}
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
                    else:
                        delta = self.graph_memory[cva_idx](target_frames, dense_memory.to(hidden.device, hidden.dtype), semantic_memory.to(hidden.device, hidden.dtype), target_plucker.to(hidden.device, hidden.dtype), condition_plucker.to(hidden.device, hidden.dtype), relative_pose_features.to(hidden.device, hidden.dtype), condition_available_mask.to(hidden.device), sigma.to(hidden.device))
                if debug_record is not None:
                    target_norm = self._debug_norm(target_hidden)
                    delta_norm = self._debug_norm(delta)
                    gate_module = self.graph_memory[cva_idx].gate
                    with torch.no_grad(), torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        (g_sem, g_dense) = gate_module(sigma.to(hidden.device))
                    debug_record['graph_blocks'].append({'block_idx': int(block_idx), 'target_hidden_norm': target_norm, 'graph_delta_norm': delta_norm, 'graph_delta_to_target_ratio': delta_norm / max(target_norm, 1e-12), 'g_sem_mean': float(g_sem.detach().float().mean().cpu()), 'g_dense_mean': float(g_dense.detach().float().mean().cpu())})
                hidden_view = hidden.view(batch, P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                front_stream = hidden_view[:, 0]
                target_stream = hidden_view[:, P61_TARGET_STREAM_INDEX] + delta
                hidden = torch.stack([front_stream, target_stream], dim=1).reshape(batch * P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                cva_idx += 1
        if debug_record is not None:
            self.geometry_debug_records.append(debug_record)
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, P61_BACKBONE_STREAMS, *outputs[0].shape)
