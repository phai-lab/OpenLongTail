from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pandas as pd
SENSOR_ORDER: tuple[str, ...] = ('camera_front_wide_120fov_undistorted', 'camera_cross_left_120fov_undistorted', 'camera_cross_right_120fov_undistorted', 'camera_rear_left_70fov_undistorted', 'camera_rear_right_70fov_undistorted', 'camera_rear_tele_30fov_undistorted')

@dataclass(frozen=True)
class SixCamMetadata:
    uuid: str
    uuid_dir: Path
    rig_root: Path
    camera_root: Path
    intrinsics_path: Path
    extrinsics_path: Path
    caption_path: Path
    camera_video_paths: dict[str, Path]
    camera_timestamp_paths: dict[str, Path]

def load_uuid_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if isinstance(payload, dict) and 'uuids' in payload:
        return {str(item) for item in payload['uuids']}
    raise ValueError(f"expected UUID allowlist list or dict with 'uuids', got {type(payload).__name__}")

def discover_uuid_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f'expected data_root to exist, got {data_root}')
    return sorted((path for path in data_root.iterdir() if path.is_dir()))

def filter_uuid_dirs(uuid_dirs: list[Path], allowlist: set[str] | None, max_items: int | None=None) -> list[Path]:
    filtered = [path for path in uuid_dirs if allowlist is None or path.name in allowlist]
    if max_items is not None:
        filtered = filtered[:max_items]
    return filtered

def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def build_6cam_metadata(uuid_dir: Path, use_undistorted_simplecalib: bool=True, use_offline_extrinsics: bool=True) -> SixCamMetadata:
    uuid = uuid_dir.name
    if use_undistorted_simplecalib:
        rig_root = uuid_dir / 'all_views_undistorted_simplecalib'
    else:
        rig_root = uuid_dir / 'all_views_undistorted'
    camera_root = rig_root / 'camera'
    intrinsics_path = _first_existing(rig_root / 'calibration/camera_intrinsics/camera_intrinsics.parquet', rig_root / 'calibration/camera_intrinsics.offline/camera_intrinsics.offline.parquet')
    extrinsics_path = rig_root / 'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.parquet' if use_offline_extrinsics else rig_root / 'calibration/sensor_extrinsics/sensor_extrinsics.parquet'
    camera_video_paths: dict[str, Path] = {}
    camera_timestamp_paths: dict[str, Path] = {}
    for sensor_name in SENSOR_ORDER:
        camera_dir = camera_root / sensor_name
        camera_video_paths[sensor_name] = camera_dir / f'{uuid}.{sensor_name}.mp4'
        camera_timestamp_paths[sensor_name] = camera_dir / f'{uuid}.{sensor_name}.timestamps.parquet'
    return SixCamMetadata(uuid=uuid, uuid_dir=uuid_dir, rig_root=rig_root, camera_root=camera_root, intrinsics_path=intrinsics_path, extrinsics_path=extrinsics_path, caption_path=uuid_dir / 'vlm_caption.txt', camera_video_paths=camera_video_paths, camera_timestamp_paths=camera_timestamp_paths)

def validate_6cam_metadata(metadata: SixCamMetadata) -> None:
    missing: list[Path] = []
    for sensor_name in SENSOR_ORDER:
        missing.extend((path for path in (metadata.camera_video_paths[sensor_name], metadata.camera_timestamp_paths[sensor_name]) if not path.exists()))
    for path in (metadata.intrinsics_path, metadata.extrinsics_path):
        if not path.exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError('missing OpenLongTail dataset files:\n' + '\n'.join((str(path) for path in missing)))

@lru_cache(maxsize=4096)
def read_timestamps(path: Path) -> list[int]:
    frame_table = pd.read_parquet(path)
    if 'timestamp' not in frame_table.columns:
        raise ValueError(f'expected timestamp column in {path}, got {frame_table.columns.tolist()}')
    return frame_table['timestamp'].astype('int64').tolist()
