from __future__ import annotations
import math
from collections.abc import Callable, Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.models.camera_embed import CamIDEmbed, PluckerMLP
from openlongtail.models.ray_cross_view_attention import RayCrossViewAttention

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f'expected even sinusoidal dim, got {dim}')
    half = dim // 2
    position = position.to(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half))
    sinusoid = torch.outer(position, freqs)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

class DiT(nn.Module):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(7, 15, 23, 31), blank_image_cond_latent: torch.Tensor | None=None) -> None:
        super().__init__()
        if blank_image_cond_latent is None:
            raise ValueError('blank_image_cond_latent is required for target-view I2V conditioning')
        self.expert = expert
        self.dim = int(dim or getattr(expert, 'dim'))
        self.cross_view_blocks = tuple((int(idx) for idx in cross_view_blocks))
        self.gradient_checkpointing = False
        self.use_reentrant = True
        self.plucker_mlp = PluckerMLP(out_dim=self.dim)
        self.cam_id_embed = CamIDEmbed(dim=self.dim)
        self.cross_view = nn.ModuleList([RayCrossViewAttention(dim_in=self.dim, dim_attn=dim_attn, heads=heads) for _ in self.cross_view_blocks])
        self.register_buffer('blank_image_cond_latent', blank_image_cond_latent.detach().clone(), persistent=False)
        self.last_patch_input_shape: tuple[int, ...] | None = None
        self.last_t_per_view: torch.Tensor | None = None
        self.last_svt_boundary_devices: list[str] = []

    def enable_gradient_checkpointing(self, use_reentrant: bool=True) -> None:
        if use_reentrant is not True:
            raise ValueError('OpenLongTail requires gradient checkpointing use_reentrant=True')
        self.gradient_checkpointing = True
        self.use_reentrant = True

    def _build_i2v_patch_input(self, z_input: torch.Tensor) -> torch.Tensor:
        if z_input.ndim != 6 or z_input.shape[2] != 16:
            raise ValueError(f'expected z_input shape (B, V, 16, T, H, W), got {tuple(z_input.shape)}')
        (batch, views, _, frames, height, width) = z_input.shape
        mask = torch.zeros(batch, views, 4, frames, height, width, device=z_input.device, dtype=z_input.dtype)
        mask[:, 0] = 1
        blank = self.blank_image_cond_latent.to(device=z_input.device, dtype=z_input.dtype)
        if tuple(blank.shape) != (16, frames, height, width):
            raise ValueError(f'expected blank_image_cond_latent shape (16, {frames}, {height}, {width}), got {tuple(blank.shape)}')
        image_cond = blank.view(1, 1, 16, frames, height, width).expand(batch, views, -1, -1, -1, -1).clone()
        image_cond[:, 0] = z_input[:, 0]
        return torch.cat([z_input, mask, image_cond], dim=2)

    def _build_i2v_patch_input_single(self, z_input: torch.Tensor, view_id: int) -> torch.Tensor:
        if z_input.ndim != 5 or z_input.shape[1] != 16:
            raise ValueError(f'expected z_input shape (B, 16, T, H, W), got {tuple(z_input.shape)}')
        if view_id < 0 or view_id >= 6:
            raise ValueError(f'expected view_id in [0, 5], got {view_id}')
        (batch, _, frames, height, width) = z_input.shape
        if view_id == 0:
            mask = torch.ones(batch, 4, frames, height, width, device=z_input.device, dtype=z_input.dtype)
            image_cond = z_input
        else:
            mask = torch.zeros(batch, 4, frames, height, width, device=z_input.device, dtype=z_input.dtype)
            blank = self.blank_image_cond_latent.to(device=z_input.device, dtype=z_input.dtype)
            if tuple(blank.shape) != (16, frames, height, width):
                raise ValueError(f'expected blank_image_cond_latent shape (16, {frames}, {height}, {width}), got {tuple(blank.shape)}')
            image_cond = blank.view(1, 16, frames, height, width).expand(batch, -1, -1, -1, -1)
        return torch.cat([z_input, mask, image_cond], dim=1)

    def _prepare_time(self, sigma: torch.Tensor, batch: int, num_views: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.shape != (batch,):
            raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
        t_real = sigma / (5.0 - 4.0 * sigma)
        t_per_view = t_real.unsqueeze(1).expand(batch, num_views).clone()
        t_per_view[:, 0] = 0
        self.last_t_per_view = t_per_view.detach().cpu()
        t_tokens = t_per_view.reshape(batch * num_views, 1).expand(batch * num_views, seq_len)
        freq_dim = int(getattr(self.expert, 'freq_dim', 256))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            flat_t = t_tokens.flatten().to(device)
            e = self.expert.time_embedding(sinusoidal_embedding_1d(freq_dim, flat_t).unflatten(0, (batch * num_views, seq_len)).float())
            e0 = self.expert.time_projection(e).unflatten(2, (6, self.dim))
        return (e, e0)

    def _prepare_time_single(self, sigma: torch.Tensor, batch: int, seq_len: int, device: torch.device, view_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.shape != (batch,):
            raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
        t_real = sigma / (5.0 - 4.0 * sigma)
        if view_id == 0:
            t_real = torch.zeros_like(t_real)
        self.last_t_per_view = t_real.detach().cpu().view(batch, 1)
        t_tokens = t_real.view(batch, 1).expand(batch, seq_len)
        freq_dim = int(getattr(self.expert, 'freq_dim', 256))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            flat_t = t_tokens.flatten().to(device)
            e = self.expert.time_embedding(sinusoidal_embedding_1d(freq_dim, flat_t).unflatten(0, (batch, seq_len)).float())
            e0 = self.expert.time_projection(e).unflatten(2, (6, self.dim))
        return (e, e0)

    def _prepare_context(self, text_emb: torch.Tensor, batch: int, num_views: int) -> torch.Tensor:
        if text_emb.ndim != 3 or text_emb.shape[0] != batch or text_emb.shape[-1] != 4096:
            raise ValueError(f'expected text_emb shape (B, L, 4096), got {tuple(text_emb.shape)}')
        context = text_emb[:, None].expand(batch, num_views, -1, -1).reshape(batch * num_views, text_emb.shape[1], text_emb.shape[2])
        return self.expert.text_embedding(context)

    def _prepare_context_single(self, text_emb: torch.Tensor, batch: int) -> torch.Tensor:
        if text_emb.ndim != 3 or text_emb.shape[0] != batch or text_emb.shape[-1] != 4096:
            raise ValueError(f'expected text_emb shape (B, L, 4096), got {tuple(text_emb.shape)}')
        return self.expert.text_embedding(text_emb)

    def _patchify_with_conditions(self, z_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_input = self._build_i2v_patch_input(z_input)
        (batch, views) = patch_input.shape[:2]
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = patch_input.reshape(batch * views, *patch_input.shape[2:]).to(dtype=patch_dtype)
        self.last_patch_input_shape = tuple(flat.shape)
        x = self.expert.patch_embedding(flat)
        grid_sizes = torch.tensor([x.shape[2:]] * (batch * views), dtype=torch.long, device=x.device)
        hidden = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((batch * views,), hidden.shape[1], dtype=torch.long, device=x.device)
        return (hidden, grid_sizes, seq_lens)

    def _patchify_single_with_conditions(self, z_input: torch.Tensor, view_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_input = self._build_i2v_patch_input_single(z_input, view_id)
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = patch_input.to(dtype=patch_dtype)
        self.last_patch_input_shape = tuple(flat.shape)
        x = self.expert.patch_embedding(flat)
        grid_sizes = torch.tensor([x.shape[2:]] * z_input.shape[0], dtype=torch.long, device=x.device)
        hidden = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((z_input.shape[0],), hidden.shape[1], dtype=torch.long, device=x.device)
        return (hidden, grid_sizes, seq_lens)

    def _run_block_segment(self, hidden: torch.Tensor, start_idx: int, end_idx: int, e0: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor | None, context: torch.Tensor, context_lens: object | None) -> torch.Tensor:
        for block_idx in range(start_idx, end_idx):
            hidden = self.expert.blocks[block_idx](hidden, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=context_lens)
        return hidden

    def _run_block_segment_all_views(self, hidden_0: torch.Tensor, hidden_1: torch.Tensor, hidden_2: torch.Tensor, hidden_3: torch.Tensor, hidden_4: torch.Tensor, hidden_5: torch.Tensor, start_idx: int, end_idx: int, e0_by_view: list[torch.Tensor], seq_lens_by_view: list[torch.Tensor], grid_sizes_by_view: list[torch.Tensor], freqs: torch.Tensor | None, context_by_view: list[torch.Tensor], context_lens: object | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_views = [hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5]
        out: list[torch.Tensor] = []
        for (view_id, hidden) in enumerate(hidden_views):
            out.append(self._run_block_segment(hidden, start_idx, end_idx, e0_by_view[view_id], seq_lens_by_view[view_id], grid_sizes_by_view[view_id], freqs, context_by_view[view_id], context_lens))
        return (out[0], out[1], out[2], out[3], out[4], out[5])

    def _apply_cva_all_views(self, hidden_by_view: torch.Tensor, rays: torch.Tensor, cva_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out: list[torch.Tensor] = []
        module = self.cross_view[cva_idx]
        for view_id in range(6):
            out.append(hidden_by_view[:, view_id] + module.forward_local(hidden_by_view, rays, view_id))
        return (out[0], out[1], out[2], out[3], out[4], out[5])

    def _flatten_plucker_embedding(self, plucker: torch.Tensor, view_ids: torch.Tensor, batch: int) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, V, T, 6, H, W), got {tuple(plucker.shape)}')
        num_views = plucker.shape[1]
        if view_ids.numel() != num_views:
            raise ValueError(f'expected {num_views} view ids, got {tuple(view_ids.shape)}')
        plucker_emb = self.plucker_mlp(plucker).permute(0, 1, 2, 4, 5, 3).reshape(batch * num_views, -1, self.dim)
        cam_emb = self.cam_id_embed(view_ids.to(plucker.device)).view(1, num_views, 1, self.dim)
        cam_emb = cam_emb.expand(batch, -1, plucker_emb.shape[1], -1).reshape(batch * num_views, plucker_emb.shape[1], self.dim)
        return plucker_emb + cam_emb

    def _flatten_plucker_embedding_single(self, plucker: torch.Tensor, local_view_pos: int, global_view_id: int, batch: int) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, V, T, 6, H, W), got {tuple(plucker.shape)}')
        if local_view_pos < 0 or local_view_pos >= plucker.shape[1]:
            raise ValueError(f'expected local_view_pos in [0, {plucker.shape[1] - 1}], got {local_view_pos}')
        plucker_view = plucker[:, local_view_pos:local_view_pos + 1]
        plucker_emb = self.plucker_mlp(plucker_view)[:, 0].permute(0, 1, 3, 4, 2).reshape(batch, -1, self.dim)
        view_ids = torch.tensor([global_view_id], device=plucker.device, dtype=torch.long)
        cam_emb = self.cam_id_embed(view_ids).view(1, 1, self.dim).expand(batch, plucker_emb.shape[1], -1)
        return plucker_emb + cam_emb

    def _ensure_reentrant_checkpoint_input(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and (not hidden.requires_grad):
            return hidden.detach().requires_grad_(True)
        return hidden

    def forward(self, z_input: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, plucker: torch.Tensor, view_ids: torch.Tensor | int, hidden_gather: Callable[[torch.Tensor], torch.Tensor] | None=None, training_mode: str='standard', active_views: list[int] | None=None) -> torch.Tensor:
        if training_mode == 'sequential_view_recompute':
            if not isinstance(view_ids, torch.Tensor):
                raise ValueError(f'expected tensor view_ids for sequential_view_recompute, got {type(view_ids).__name__}')
            return self.forward_sequential_views_recompute(z_input, sigma, text_emb, text_mask, plucker, view_ids, active_views)
        if z_input.ndim == 5:
            if not isinstance(view_ids, int):
                raise ValueError(f'expected int view_id for single-view z_input, got {type(view_ids).__name__}')
            return self.forward_single_view(z_input, sigma, text_emb, text_mask, plucker, view_ids, hidden_gather)
        del text_mask
        batch = z_input.shape[0]
        num_views = z_input.shape[1]
        if active_views is None:
            active_views = list(range(num_views))
        if len(active_views) != num_views:
            raise ValueError(f'expected active_views length {num_views}, got {active_views}')
        (hidden, grid_sizes, seq_lens) = self._patchify_with_conditions(z_input)
        hidden = hidden + self._flatten_plucker_embedding(plucker.to(hidden.device, hidden.dtype), view_ids, batch)
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time(sigma.to(hidden.device, hidden.dtype), batch, num_views, seq_len, hidden.device)
        context = self._prepare_context(text_emb.to(hidden.device, hidden.dtype), batch, num_views)
        context_lens = None
        freqs = self.expert.freqs.to(hidden.device) if hasattr(self.expert, 'freqs') else None
        autocast_enabled = hidden.device.type == 'cuda'
        cva_idx = 0
        for (block_idx, block) in enumerate(self.expert.blocks):
            kwargs = {'e': e0, 'seq_lens': seq_lens, 'grid_sizes': grid_sizes, 'freqs': freqs, 'context': context, 'context_lens': context_lens}
            with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(lambda x, b=block, kw=kwargs: b(x, **kw), hidden, use_reentrant=True)
                else:
                    hidden = block(hidden, **kwargs)
            if block_idx in self.cross_view_blocks:
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    rays = plucker.to(hidden.device, hidden.dtype)
                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda x, p, module=self.cross_view[cva_idx], av=active_views: x + module(x, p, active_views=av), hidden, rays, use_reentrant=True)
                    else:
                        hidden = hidden + self.cross_view[cva_idx](hidden, rays, active_views=active_views)
                cva_idx += 1
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, num_views, *outputs[0].shape)

    def forward_single_view(self, z_input: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, plucker: torch.Tensor, view_id: int, hidden_gather: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
        del text_mask
        batch = z_input.shape[0]
        (hidden, grid_sizes, seq_lens) = self._patchify_single_with_conditions(z_input, view_id)
        hidden = hidden + self._flatten_plucker_embedding_single(plucker.to(hidden.device, hidden.dtype), view_id, view_id, batch)
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time_single(sigma.to(hidden.device, hidden.dtype), batch, seq_len, hidden.device, view_id)
        context = self._prepare_context_single(text_emb.to(hidden.device, hidden.dtype), batch)
        context_lens = None
        freqs = self.expert.freqs.to(hidden.device) if hasattr(self.expert, 'freqs') else None
        autocast_enabled = hidden.device.type == 'cuda'
        cva_idx = 0
        for (block_idx, block) in enumerate(self.expert.blocks):
            kwargs = {'e': e0, 'seq_lens': seq_lens, 'grid_sizes': grid_sizes, 'freqs': freqs, 'context': context, 'context_lens': context_lens}
            with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(lambda x, b=block, kw=kwargs: b(x, **kw), hidden, use_reentrant=True)
                else:
                    hidden = block(hidden, **kwargs)
            if block_idx in self.cross_view_blocks:
                if hidden_gather is None:
                    raise ValueError('hidden_gather is required for single-view cross-view attention')
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    hidden_by_view = hidden_gather(hidden)
                    rays = plucker.to(hidden.device, hidden.dtype)
                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda h, p, module=self.cross_view[cva_idx]: h[:, view_id] + module.forward_local(h, p, view_id), hidden_by_view, rays, use_reentrant=True)
                    else:
                        hidden = hidden + self.cross_view[cva_idx].forward_local(hidden_by_view, rays, view_id)
                cva_idx += 1
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0)

    def forward_sequential_views_recompute(self, z_input: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, plucker: torch.Tensor, view_ids: torch.Tensor, active_views: list[int] | None=None) -> torch.Tensor:
        del text_mask
        if z_input.ndim != 6 or z_input.shape[2] != 16:
            raise ValueError(f'expected z_input shape (B, V, 16, T, H, W), got {tuple(z_input.shape)}')
        batch = z_input.shape[0]
        num_views = z_input.shape[1]
        if active_views is None:
            active_views = [int(item) for item in view_ids.detach().cpu().tolist()]
        if len(active_views) != num_views:
            raise ValueError(f'expected active_views length {num_views}, got {active_views}')
        if view_ids.numel() != num_views:
            raise ValueError(f'expected {num_views} view ids, got {tuple(view_ids.shape)}')
        autocast_enabled = z_input.device.type == 'cuda'
        freqs = self.expert.freqs.to(z_input.device) if hasattr(self.expert, 'freqs') else None
        context_lens = None
        self.last_svt_boundary_devices = []
        hidden_views: list[torch.Tensor] = []
        grid_sizes_by_view: list[torch.Tensor] = []
        seq_lens_by_view: list[torch.Tensor] = []
        e_by_view: list[torch.Tensor] = []
        e0_by_view: list[torch.Tensor] = []
        context_by_view: list[torch.Tensor] = []
        for (local_pos, global_view_id) in enumerate(active_views):
            (hidden, grid_sizes, seq_lens) = self._patchify_single_with_conditions(z_input[:, local_pos], global_view_id)
            hidden = hidden + self._flatten_plucker_embedding_single(plucker.to(hidden.device, hidden.dtype), local_pos, global_view_id, batch)
            hidden = self._ensure_reentrant_checkpoint_input(hidden)
            (e, e0) = self._prepare_time_single(sigma.to(hidden.device, hidden.dtype), batch, hidden.shape[1], hidden.device, global_view_id)
            context = self._prepare_context_single(text_emb.to(hidden.device, hidden.dtype), batch)
            hidden_views.append(hidden)
            grid_sizes_by_view.append(grid_sizes)
            seq_lens_by_view.append(seq_lens)
            e_by_view.append(e)
            e0_by_view.append(e0)
            context_by_view.append(context)
        segment_start = 0
        cva_idx = 0
        final_block_idx = len(self.expert.blocks) - 1
        segment_ends = list(self.cross_view_blocks)
        if not segment_ends or segment_ends[-1] < final_block_idx:
            segment_ends.append(final_block_idx)
        for segment_end in segment_ends:
            block_stop = segment_end + 1
            for view_id in range(num_views):
                with torch.amp.autocast(z_input.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    if self.gradient_checkpointing and self.training:
                        hidden_views[view_id] = checkpoint(lambda h, v=view_id, s=segment_start, e=block_stop: self._run_block_segment(h, s, e, e0_by_view[v], seq_lens_by_view[v], grid_sizes_by_view[v], freqs, context_by_view[v], context_lens), hidden_views[view_id], use_reentrant=True)
                    else:
                        hidden_views[view_id] = self._run_block_segment(hidden_views[view_id], segment_start, block_stop, e0_by_view[view_id], seq_lens_by_view[view_id], grid_sizes_by_view[view_id], freqs, context_by_view[view_id], context_lens)
            self.last_svt_boundary_devices.extend((hidden.device.type for hidden in hidden_views))
            if segment_end in self.cross_view_blocks:
                hidden_by_view = torch.stack(hidden_views, dim=1)
                rays = plucker.to(hidden_by_view.device, hidden_by_view.dtype)
                updated: list[torch.Tensor] = []
                for (local_pos, global_view_id) in enumerate(active_views):
                    with torch.amp.autocast(z_input.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        if self.gradient_checkpointing and self.training:
                            updated_hidden = checkpoint(lambda h, p, lp=local_pos, gv=global_view_id, av=active_views, module=self.cross_view[cva_idx]: h[:, lp] + module.forward_local(h, p, gv, active_views=av), hidden_by_view, rays, use_reentrant=True)
                        else:
                            updated_hidden = hidden_views[local_pos] + self.cross_view[cva_idx].forward_local(hidden_by_view, rays, global_view_id, active_views=active_views)
                    updated.append(updated_hidden)
                hidden_views = updated
                cva_idx += 1
            segment_start = block_stop
        outputs: list[torch.Tensor] = []
        for (view_id, hidden) in enumerate(hidden_views):
            with torch.amp.autocast(z_input.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                hidden = self.expert.head(hidden, e_by_view[view_id])
            outputs.append(torch.stack(self.expert.unpatchify(hidden, grid_sizes_by_view[view_id]), dim=0))
        return torch.stack(outputs, dim=1)
