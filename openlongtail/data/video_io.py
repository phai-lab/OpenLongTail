from __future__ import annotations
from pathlib import Path
import decord
import torch
import torch.nn.functional as F

def load_rgb_clip(video_path: Path, frame_indices: list[int], output_size: tuple[int, int]) -> torch.Tensor:
    if not frame_indices:
        raise ValueError('expected at least one frame index, got empty frame_indices')
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    num_frames = len(reader)
    bad_indices = [idx for idx in frame_indices if idx < 0 or idx >= num_frames]
    if bad_indices:
        raise IndexError(f'frame indices out of range for {video_path}: {bad_indices[:5]} with num_frames={num_frames}')
    batch = reader.get_batch(frame_indices).asnumpy()
    frames = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().float()
    if tuple(frames.shape[-2:]) != output_size:
        frames = F.interpolate(frames, size=output_size, mode='bilinear', align_corners=False)
    return frames
