from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

def _reshape_stream_tokens(hidden: torch.Tensor, plucker: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if plucker.ndim != 6 or plucker.shape[3] != 6:
        raise ValueError(f'expected plucker shape (B, S, T, 6, H, W), got {tuple(plucker.shape)}')
    (batch, streams, frames, _, height, width) = plucker.shape
    tokens_per_frame = height * width
    expected = (batch * streams, frames * tokens_per_frame, hidden.shape[-1])
    if tuple(hidden.shape) != expected:
        raise ValueError(f'expected hidden shape {expected}, got {tuple(hidden.shape)}')
    return (hidden.view(batch, streams, frames, tokens_per_frame, hidden.shape[-1]), (batch, streams, frames, tokens_per_frame))

class RaySynchronizedFrontAttention(nn.Module):

    def __init__(self, dim: int, heads: int=16, temporal_window: int=2) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        if temporal_window < 0:
            raise ValueError(f'expected temporal_window >= 0, got {temporal_window}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.temporal_window = int(temporal_window)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor) -> torch.Tensor:
        (stream_tokens, (batch, streams, frames, tokens_per_frame)) = _reshape_stream_tokens(hidden, plucker)
        if stream_view_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_view_ids, got {tuple(stream_view_ids.shape)}')
        if stream_role_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_role_ids, got {tuple(stream_role_ids.shape)}')
        front_candidates = (stream_role_ids.to(hidden.device) == 0).nonzero(as_tuple=False).flatten()
        if front_candidates.numel() == 0:
            raise ValueError('expected at least one condition/front stream')
        front_idx = int(front_candidates[0].item())
        target_indices = (stream_role_ids.to(hidden.device) == 1).nonzero(as_tuple=False).flatten().tolist()
        if not target_indices:
            return torch.zeros_like(hidden)
        delta = torch.zeros_like(stream_tokens)
        for frame_idx in range(frames):
            lo = max(0, frame_idx - self.temporal_window)
            hi = min(frames, frame_idx + self.temporal_window + 1)
            kv = stream_tokens[:, front_idx, lo:hi].reshape(batch, (hi - lo) * tokens_per_frame, self.dim)
            for stream_idx in target_indices:
                q = stream_tokens[:, int(stream_idx), frame_idx]
                (attended, _) = self.attn(q, kv, kv, need_weights=False)
                delta[:, int(stream_idx), frame_idx] = self.out(attended)
        return delta.reshape(batch * streams, frames * tokens_per_frame, self.dim)
