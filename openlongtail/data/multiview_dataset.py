from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from openlongtail.data.rig_parquet import SENSOR_ORDER, SixCamMetadata, build_6cam_metadata, discover_uuid_dirs, filter_uuid_dirs, load_uuid_allowlist, read_timestamps, validate_6cam_metadata
from openlongtail.data.text_emb_cache import load_text_embedding, load_text_embedding_for_uuid, pad_text_embedding_batch
from openlongtail.data.transforms import compute_T_anchor_front_for_clip, nearest_indices, quat_xyz_to_se3, scale_intrinsics
from openlongtail.data.video_io import load_rgb_clip

@dataclass(frozen=True)
class RayMultiViewDataConfig:
    data_root: Path
    text_emb_cache_root: Path
    uuid_allowlist_json: Path | None = None
    clip_length: int = 81
    output_size: tuple[int, int] = (480, 832)
    target_fps: int = 16
    clip_anchor_seconds: float = 10.0
    clip_jitter_seconds: float = 2.0
    use_undistorted_simplecalib: bool = True
    use_offline_extrinsics: bool = True
    text_drop_prob: float = 0.1
    max_items: int | None = None
    num_workers: int = 4
    include_p4_front_pose: bool = False

def _sample_clip_start(num_frames: int, clip_length: int, target_fps: int, anchor_seconds: float, jitter_seconds: float) -> int:
    if num_frames <= 0:
        raise ValueError(f'expected num_frames > 0, got {num_frames}')
    duration_source_frames = int(round((clip_length - 1) * 30.0 / float(target_fps))) + 1
    max_start = max(0, num_frames - duration_source_frames)
    anchor_frame = int(round(anchor_seconds * 30.0))
    jitter_frames = int(round(jitter_seconds * 30.0))
    low = max(0, anchor_frame - jitter_frames)
    high = min(max_start, anchor_frame + jitter_frames)
    if high < low:
        return min(anchor_frame, max_start)
    return random.randint(low, high)

def _target_timestamps(front_ts: list[int], clip_start: int, clip_length: int, target_fps: int) -> list[int]:
    if clip_start < 0 or clip_start >= len(front_ts):
        raise ValueError(f'expected clip_start in [0, {len(front_ts)}), got {clip_start}')
    source_dt = front_ts[min(clip_start + 1, len(front_ts) - 1)] - front_ts[max(clip_start - 1, 0)]
    source_dt = abs(source_dt) if source_dt != 0 else abs(front_ts[-1] - front_ts[0]) / max(len(front_ts) - 1, 1)
    target_dt = (1000000000 if source_dt > 1000000 else 1000000) // target_fps
    first_ts = int(front_ts[clip_start])
    return [first_ts + idx * target_dt for idx in range(clip_length)]

class RaySixCamDataset(Dataset[dict[str, Any]]):

    def __init__(self, config: RayMultiViewDataConfig) -> None:
        self.config = config
        allowlist = load_uuid_allowlist(config.uuid_allowlist_json)
        uuid_dirs = discover_uuid_dirs(config.data_root)
        self.uuid_dirs = filter_uuid_dirs(uuid_dirs, allowlist, config.max_items)
        if not self.uuid_dirs:
            raise FileNotFoundError(f'no UUID directories available under {config.data_root}')

    def __len__(self) -> int:
        return len(self.uuid_dirs)

    def _metadata(self, uuid_dir: Path) -> SixCamMetadata:
        metadata = build_6cam_metadata(uuid_dir, use_undistorted_simplecalib=self.config.use_undistorted_simplecalib, use_offline_extrinsics=self.config.use_offline_extrinsics)
        validate_6cam_metadata(metadata)
        return metadata

    def __getitem__(self, idx: int) -> dict[str, Any]:
        metadata = self._metadata(self.uuid_dirs[idx])
        front_ts = read_timestamps(metadata.camera_timestamp_paths[SENSOR_ORDER[0]])
        clip_start = _sample_clip_start(len(front_ts), self.config.clip_length, self.config.target_fps, self.config.clip_anchor_seconds, self.config.clip_jitter_seconds)
        target_ts = _target_timestamps(front_ts, clip_start, self.config.clip_length, self.config.target_fps)
        rgb_views: list[torch.Tensor] = []
        frame_indices: dict[str, torch.Tensor] = {}
        for sensor_name in SENSOR_ORDER:
            cam_ts = read_timestamps(metadata.camera_timestamp_paths[sensor_name])
            indices = nearest_indices(cam_ts, target_ts)
            frame_indices[sensor_name] = torch.tensor(indices, dtype=torch.long)
            rgb_views.append(load_rgb_clip(metadata.camera_video_paths[sensor_name], indices, output_size=self.config.output_size))
        K = scale_intrinsics(pd.read_parquet(metadata.intrinsics_path), output_size=self.config.output_size)
        E = quat_xyz_to_se3(pd.read_parquet(metadata.extrinsics_path))
        if random.random() < self.config.text_drop_prob:
            (text_emb, text_mask) = load_text_embedding(self.config.text_emb_cache_root, 'null')
        else:
            (text_emb, text_mask) = load_text_embedding_for_uuid(self.config.text_emb_cache_root, metadata.uuid)
        item: dict[str, Any] = {'uuid': metadata.uuid, 'rgb': torch.stack(rgb_views, dim=0), 'K': K, 'E': E, 'text_emb': text_emb, 'text_mask': text_mask, 'view_ids': torch.arange(len(SENSOR_ORDER), dtype=torch.long), 'clip_start_frame': torch.tensor(clip_start, dtype=torch.long), 'target_timestamps': torch.tensor(target_ts, dtype=torch.long), 'frame_indices': frame_indices}
        if self.config.include_p4_front_pose:
            item['T_anchor_front'] = compute_T_anchor_front_for_clip(metadata.uuid_dir, target_ts, E, t_lat=(self.config.clip_length - 1) // 4 + 1, vae_temporal_stride=4)
        return item

def ray_six_cam_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError('expected at least one item to collate')
    (text_emb, text_mask) = pad_text_embedding_batch([(item['text_emb'], item['text_mask']) for item in items])
    batch: dict[str, Any] = {'uuid': [item['uuid'] for item in items], 'rgb': torch.stack([item['rgb'] for item in items], dim=0), 'K': torch.stack([item['K'] for item in items], dim=0), 'E': torch.stack([item['E'] for item in items], dim=0), 'text_emb': text_emb, 'text_mask': text_mask, 'view_ids': torch.stack([item['view_ids'] for item in items], dim=0), 'clip_start_frame': torch.stack([item['clip_start_frame'] for item in items], dim=0), 'target_timestamps': torch.stack([item['target_timestamps'] for item in items], dim=0), 'frame_indices': {sensor_name: torch.stack([item['frame_indices'][sensor_name] for item in items], dim=0) for sensor_name in SENSOR_ORDER}}
    if 'T_anchor_front' in items[0]:
        batch['T_anchor_front'] = torch.stack([item['T_anchor_front'] for item in items], dim=0)
    return batch
