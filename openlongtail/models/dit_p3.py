from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.models.camera_embed import CamIDEmbed, PluckerMLP
from openlongtail.models.ray_cross_view_attention_p3 import RayCrossViewAttentionP3
from openlongtail.models.dit import sinusoidal_embedding_1d

class DiTP3(nn.Module):

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
        self.role_embed = nn.Embedding(2, self.dim)
        nn.init.normal_(self.role_embed.weight, std=0.02)
        self.cross_view = nn.ModuleList([RayCrossViewAttentionP3(dim_in=self.dim, dim_attn=dim_attn, heads=heads) for _ in self.cross_view_blocks])
        self.register_buffer('blank_image_cond_latent', blank_image_cond_latent.detach().clone(), persistent=False)
        self.last_patch_input_shape: tuple[int, ...] | None = None
        self.last_t_per_stream: torch.Tensor | None = None

    def enable_gradient_checkpointing(self, use_reentrant: bool=True) -> None:
        if use_reentrant is not True:
            raise ValueError('OpenLongTail requires gradient checkpointing use_reentrant=True')
        self.gradient_checkpointing = True
        self.use_reentrant = True

    def _build_i2v_patch_input(self, z_streams: torch.Tensor) -> torch.Tensor:
        if z_streams.ndim != 6 or z_streams.shape[2] != 16:
            raise ValueError(f'expected z_streams shape (B, S, 16, T, H, W), got {tuple(z_streams.shape)}')
        (batch, streams, _, frames, height, width) = z_streams.shape
        mask = torch.zeros(batch, streams, 4, frames, height, width, device=z_streams.device, dtype=z_streams.dtype)
        mask[:, 0] = 1
        blank = self.blank_image_cond_latent.to(device=z_streams.device, dtype=z_streams.dtype)
        if tuple(blank.shape) != (16, frames, height, width):
            raise ValueError(f'expected blank_image_cond_latent shape (16, {frames}, {height}, {width}), got {tuple(blank.shape)}')
        image_cond = blank.view(1, 1, 16, frames, height, width).expand(batch, streams, -1, -1, -1, -1).clone()
        image_cond[:, 0] = z_streams[:, 0]
        return torch.cat([z_streams, mask, image_cond], dim=2)

    def _patchify_with_conditions(self, z_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_input = self._build_i2v_patch_input(z_streams)
        (batch, streams) = patch_input.shape[:2]
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = patch_input.reshape(batch * streams, *patch_input.shape[2:]).to(dtype=patch_dtype)
        self.last_patch_input_shape = tuple(flat.shape)
        x = self.expert.patch_embedding(flat)
        grid_sizes = torch.tensor([x.shape[2:]] * (batch * streams), dtype=torch.long, device=x.device)
        hidden = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((batch * streams,), hidden.shape[1], dtype=torch.long, device=x.device)
        return (hidden, grid_sizes, seq_lens)

    def _flatten_condition_embedding(self, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor, batch: int) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, S, T, 6, H, W), got {tuple(plucker.shape)}')
        streams = plucker.shape[1]
        if stream_view_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream view ids, got {tuple(stream_view_ids.shape)}')
        if stream_role_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream role ids, got {tuple(stream_role_ids.shape)}')
        plucker_emb = self.plucker_mlp(plucker).permute(0, 1, 2, 4, 5, 3).reshape(batch * streams, -1, self.dim)
        cam_emb = self.cam_id_embed(stream_view_ids.to(plucker.device)).view(1, streams, 1, self.dim)
        role_emb = self.role_embed(stream_role_ids.to(plucker.device)).view(1, streams, 1, self.dim)
        tokens = plucker_emb.shape[1]
        cam_emb = cam_emb.expand(batch, -1, tokens, -1).reshape(batch * streams, tokens, self.dim)
        role_emb = role_emb.expand(batch, -1, tokens, -1).reshape(batch * streams, tokens, self.dim)
        return plucker_emb + cam_emb + role_emb

    def _prepare_time(self, sigma: torch.Tensor, batch: int, streams: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.shape != (batch,):
            raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
        t_real = sigma / (5.0 - 4.0 * sigma)
        t_per_stream = t_real.unsqueeze(1).expand(batch, streams).clone()
        t_per_stream[:, 0] = 0
        self.last_t_per_stream = t_per_stream.detach().cpu()
        t_tokens = t_per_stream.reshape(batch * streams, 1).expand(batch * streams, seq_len)
        freq_dim = int(getattr(self.expert, 'freq_dim', 256))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            flat_t = t_tokens.flatten().to(device)
            e = self.expert.time_embedding(sinusoidal_embedding_1d(freq_dim, flat_t).unflatten(0, (batch * streams, seq_len)).float())
            e0 = self.expert.time_projection(e).unflatten(2, (6, self.dim))
        return (e, e0)

    def _prepare_context(self, text_emb: torch.Tensor, batch: int, streams: int) -> torch.Tensor:
        if text_emb.ndim != 3 or text_emb.shape[0] != batch or text_emb.shape[-1] != 4096:
            raise ValueError(f'expected text_emb shape (B, L, 4096), got {tuple(text_emb.shape)}')
        context = text_emb[:, None].expand(batch, streams, -1, -1).reshape(batch * streams, text_emb.shape[1], text_emb.shape[2])
        return self.expert.text_embedding(context)

    def _ensure_reentrant_checkpoint_input(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and (not hidden.requires_grad):
            return hidden.detach().requires_grad_(True)
        return hidden

    def forward(self, z_streams: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor) -> torch.Tensor:
        del text_mask
        if z_streams.ndim != 6 or z_streams.shape[2] != 16:
            raise ValueError(f'expected z_streams shape (B, S, 16, T, H, W), got {tuple(z_streams.shape)}')
        (batch, streams) = z_streams.shape[:2]
        if streams < 2:
            raise ValueError(f'expected at least 2 P3 streams, got {streams}')
        (hidden, grid_sizes, seq_lens) = self._patchify_with_conditions(z_streams)
        hidden = hidden + self._flatten_condition_embedding(plucker.to(hidden.device, hidden.dtype), stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device), batch)
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time(sigma.to(hidden.device, hidden.dtype), batch, streams, seq_len, hidden.device)
        context = self._prepare_context(text_emb.to(hidden.device, hidden.dtype), batch, streams)
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
                rays = plucker.to(hidden.device, hidden.dtype)
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda x, p, module=self.cross_view[cva_idx]: x + module(x, p), hidden, rays, use_reentrant=True)
                    else:
                        hidden = hidden + self.cross_view[cva_idx](hidden, rays)
                cva_idx += 1
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, streams, *outputs[0].shape)
