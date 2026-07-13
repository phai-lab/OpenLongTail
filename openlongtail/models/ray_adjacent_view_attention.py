from __future__ import annotations
import torch
from torch import nn
from openlongtail.models.ray_sync_front_attention import _reshape_stream_tokens
DEFAULT_CAMERA_GRAPH: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5))

class RayAdjacentViewAttention(nn.Module):

    def __init__(self, dim: int, heads: int=16, camera_graph: tuple[tuple[int, int], ...]=DEFAULT_CAMERA_GRAPH) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.camera_graph = tuple(((int(a), int(b)) for (a, b) in camera_graph))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _neighbors(self, view_id: int) -> set[int]:
        neighbors: set[int] = set()
        for (src, dst) in self.camera_graph:
            if src == view_id:
                neighbors.add(dst)
            if dst == view_id:
                neighbors.add(src)
        return neighbors

    def forward(self, hidden: torch.Tensor, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor) -> torch.Tensor:
        (stream_tokens, (batch, streams, frames, tokens_per_frame)) = _reshape_stream_tokens(hidden, plucker)
        if stream_view_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_view_ids, got {tuple(stream_view_ids.shape)}')
        if stream_role_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_role_ids, got {tuple(stream_role_ids.shape)}')
        view_ids = [int(item) for item in stream_view_ids.detach().cpu().tolist()]
        role_ids = [int(item) for item in stream_role_ids.detach().cpu().tolist()]
        delta = torch.zeros_like(stream_tokens)
        for (target_stream_idx, (view_id, role_id)) in enumerate(zip(view_ids, role_ids)):
            if role_id != 1:
                continue
            neighbor_views = self._neighbors(view_id)
            neighbor_streams = [idx for (idx, item) in enumerate(view_ids) if item in neighbor_views]
            if not neighbor_streams:
                continue
            for frame_idx in range(frames):
                q = stream_tokens[:, target_stream_idx, frame_idx]
                kv = stream_tokens[:, neighbor_streams, frame_idx].reshape(batch, len(neighbor_streams) * tokens_per_frame, self.dim)
                (attended, _) = self.attn(q, kv, kv, need_weights=False)
                delta[:, target_stream_idx, frame_idx] = self.out(attended)
        return delta.reshape(batch * streams, frames * tokens_per_frame, self.dim)
