from __future__ import annotations
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from openlongtail.data.rig_parquet import SENSOR_ORDER

def nearest_indices(source_timestamps: Sequence[int], target_timestamps: Sequence[int]) -> list[int]:
    source = np.asarray(source_timestamps, dtype=np.int64)
    target = np.asarray(target_timestamps, dtype=np.int64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError(f'expected 1-D timestamps, got source {source.shape}, target {target.shape}')
    if len(source) == 0:
        raise ValueError('expected non-empty source_timestamps')
    positions = np.searchsorted(source, target, side='left')
    positions = np.clip(positions, 0, len(source) - 1)
    prev_positions = np.clip(positions - 1, 0, len(source) - 1)
    use_prev = np.abs(source[prev_positions] - target) <= np.abs(source[positions] - target)
    return np.where(use_prev, prev_positions, positions).astype(np.int64).tolist()

def scale_intrinsics(K_raw: pd.DataFrame, output_size: tuple[int, int]=(480, 832)) -> torch.Tensor:
    if not isinstance(K_raw, pd.DataFrame):
        raise TypeError(f'expected pandas DataFrame for K_raw, got {type(K_raw).__name__}')
    required = {'camera_name', 'width', 'height', 'fx', 'fy', 'cx', 'cy'}
    missing = sorted(required - set(K_raw.columns))
    if missing:
        raise ValueError(f'expected intrinsics columns {sorted(required)}, missing {missing}')
    (out_h, out_w) = output_size
    K = torch.zeros(len(SENSOR_ORDER), 3, 3, dtype=torch.float32)
    for (idx, sensor_name) in enumerate(SENSOR_ORDER):
        rows = K_raw[K_raw['camera_name'] == sensor_name]
        if rows.empty:
            raise ValueError(f'missing intrinsics row for {sensor_name}')
        row = rows.iloc[0]
        sx = float(out_w) / float(row['width'])
        sy = float(out_h) / float(row['height'])
        K[idx] = torch.tensor([[float(row['fx']) * sx, 0.0, float(row['cx']) * sx], [0.0, float(row['fy']) * sy, float(row['cy']) * sy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K

def _quat_xyzw_to_rotation(qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    quat = torch.tensor([qx, qy, qz, qw], dtype=torch.float64)
    quat = quat / torch.linalg.vector_norm(quat)
    (x, y, z, w) = quat.tolist()
    return torch.tensor([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=torch.float32)

def quat_xyz_to_se3(quat_xyz_dict_or_df: pd.DataFrame) -> torch.Tensor:
    if not isinstance(quat_xyz_dict_or_df, pd.DataFrame):
        raise TypeError(f'expected pandas DataFrame for extrinsics, got {type(quat_xyz_dict_or_df).__name__}')
    required = {'sensor_name', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z'}
    missing = sorted(required - set(quat_xyz_dict_or_df.columns))
    if missing:
        raise ValueError(f'expected extrinsics columns {sorted(required)}, missing {missing}')
    E = torch.eye(4, dtype=torch.float32).repeat(len(SENSOR_ORDER), 1, 1)
    for (idx, sensor_name) in enumerate(SENSOR_ORDER):
        rows = quat_xyz_dict_or_df[quat_xyz_dict_or_df['sensor_name'] == sensor_name]
        if rows.empty:
            raise ValueError(f'missing extrinsics row for {sensor_name}')
        row = rows.iloc[0]
        E[idx, :3, :3] = _quat_xyzw_to_rotation(float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw']))
        E[idx, :3, 3] = torch.tensor([float(row['x']), float(row['y']), float(row['z'])], dtype=torch.float32)
    return E

def slerp_quat_xyzw(q0: np.ndarray | torch.Tensor | Sequence[float], q1: np.ndarray | torch.Tensor | Sequence[float], frac: float) -> np.ndarray:
    q0_arr = np.asarray(q0, dtype=np.float64)
    q1_arr = np.asarray(q1, dtype=np.float64)
    if q0_arr.shape != (4,) or q1_arr.shape != (4,):
        raise ValueError(f'expected quaternion shapes (4,), got {q0_arr.shape} and {q1_arr.shape}')
    q0_norm = np.linalg.norm(q0_arr)
    q1_norm = np.linalg.norm(q1_arr)
    if q0_norm == 0.0 or q1_norm == 0.0:
        raise ValueError('expected non-zero quaternions for SLERP')
    q0_arr = q0_arr / q0_norm
    q1_arr = q1_arr / q1_norm
    dot = float(np.dot(q0_arr, q1_arr))
    if dot < 0.0:
        q1_arr = -q1_arr
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0_arr + float(frac) * (q1_arr - q0_arr)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * float(frac)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    out = s0 * q0_arr + s1 * q1_arr
    return out / np.linalg.norm(out)

def se3_from_quat_xyz_pose(q_xyzw: np.ndarray | torch.Tensor | Sequence[float], t_xyz: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    quat = np.asarray(q_xyzw, dtype=np.float64)
    trans = np.asarray(t_xyz, dtype=np.float64)
    if quat.shape != (4,) or trans.shape != (3,):
        raise ValueError(f'expected q shape (4,) and t shape (3,), got {quat.shape} and {trans.shape}')
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError('expected non-zero quaternion')
    (x, y, z, w) = (quat / norm).tolist()
    R = np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float64)
    residual = np.linalg.norm(R.T @ R - np.eye(3, dtype=np.float64), ord='fro')
    if residual > 0.001:
        (u, _, vh) = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0.0:
            u[:, -1] *= -1.0
            R = u @ vh
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T

def se3_inverse(T: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(T, torch.Tensor):
        if T.shape[-2:] != (4, 4):
            raise ValueError(f'expected SE(3) shape (..., 4, 4), got {tuple(T.shape)}')
        R = T[..., :3, :3]
        t = T[..., :3, 3:4]
        Rt = R.transpose(-1, -2)
        out = torch.zeros_like(T)
        out[..., :3, :3] = Rt
        out[..., :3, 3:4] = -Rt @ t
        out[..., 3, 3] = 1.0
        return out
    arr = np.asarray(T)
    if arr.shape[-2:] != (4, 4):
        raise ValueError(f'expected SE(3) shape (..., 4, 4), got {arr.shape}')
    R = arr[..., :3, :3]
    t = arr[..., :3, 3:4]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(arr)
    out[..., :3, :3] = Rt
    out[..., :3, 3:4] = -(Rt @ t)
    out[..., 3, 3] = 1.0
    return out

def interpolate_T_world_rig_at(ego_ts: np.ndarray, ego_quat: np.ndarray, ego_xyz: np.ndarray, ts_us: int) -> np.ndarray:
    ego_ts_arr = np.asarray(ego_ts, dtype=np.int64)
    ego_quat_arr = np.asarray(ego_quat, dtype=np.float64)
    ego_xyz_arr = np.asarray(ego_xyz, dtype=np.float64)
    if ego_ts_arr.ndim != 1 or len(ego_ts_arr) < 2:
        raise ValueError(f'expected at least two timestamps, got shape {ego_ts_arr.shape}')
    if ego_quat_arr.shape != (len(ego_ts_arr), 4) or ego_xyz_arr.shape != (len(ego_ts_arr), 3):
        raise ValueError(f'expected ego_quat shape (N,4) and ego_xyz shape (N,3), got {ego_quat_arr.shape} and {ego_xyz_arr.shape}')
    pos = int(np.searchsorted(ego_ts_arr, int(ts_us), side='left'))
    pos = max(1, min(pos, len(ego_ts_arr) - 1))
    (t0, t1) = (int(ego_ts_arr[pos - 1]), int(ego_ts_arr[pos]))
    frac = 0.0 if t1 == t0 else (int(ts_us) - t0) / (t1 - t0)
    q = slerp_quat_xyzw(ego_quat_arr[pos - 1], ego_quat_arr[pos], frac)
    p = (1.0 - frac) * ego_xyz_arr[pos - 1] + frac * ego_xyz_arr[pos]
    return se3_from_quat_xyz_pose(q, p)

@lru_cache(maxsize=32)
def _load_egomotion_arrays(ego_path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_path = Path(ego_path_str)
    table = pd.read_parquet(ego_path).sort_values('timestamp').reset_index(drop=True)
    required = {'timestamp', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z'}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f'expected columns {sorted(required)} in {ego_path}, missing {missing}')
    return (table['timestamp'].to_numpy(dtype=np.int64), table[['qx', 'qy', 'qz', 'qw']].to_numpy(dtype=np.float64), table[['x', 'y', 'z']].to_numpy(dtype=np.float64))

def latent_center_rgb_index(latent_idx: int, vae_temporal_stride: int=4) -> float:
    if latent_idx < 0:
        raise ValueError(f'expected latent_idx >= 0, got {latent_idx}')
    if vae_temporal_stride <= 0:
        raise ValueError(f'expected vae_temporal_stride > 0, got {vae_temporal_stride}')
    if latent_idx == 0:
        return 0.0
    return (latent_idx - 1) * vae_temporal_stride + (vae_temporal_stride - 1) / 2.0 + 1.0

def latent_wall_clock_us(target_timestamps: Sequence[int] | torch.Tensor, latent_idx: int, stride: int=4) -> int:
    if isinstance(target_timestamps, torch.Tensor):
        timestamps = [int(item) for item in target_timestamps.detach().cpu().tolist()]
    else:
        timestamps = [int(item) for item in target_timestamps]
    if not timestamps:
        raise ValueError('expected non-empty target_timestamps')
    r = latent_center_rgb_index(latent_idx, stride)
    lo = math.floor(r)
    if lo >= len(timestamps):
        raise ValueError(f'latent index {latent_idx} maps past {len(timestamps)} target timestamps')
    hi = min(lo + 1, len(timestamps) - 1)
    frac = r - lo
    return int(round((1.0 - frac) * timestamps[lo] + frac * timestamps[hi]))

def compute_T_anchor_front_for_clip(uuid_dir: Path, target_timestamps: Sequence[int] | torch.Tensor, rig_calibration: torch.Tensor | np.ndarray, t_lat: int, vae_temporal_stride: int=4) -> torch.Tensor:
    if t_lat <= 0:
        raise ValueError(f'expected t_lat > 0, got {t_lat}')
    uuid_dir = Path(uuid_dir)
    uuid = uuid_dir.name
    ego_path = uuid_dir / 'labels' / 'egomotion' / f'{uuid}.egomotion.parquet'
    (ego_ts, ego_quat, ego_xyz) = _load_egomotion_arrays(str(ego_path))
    latent_times = [latent_wall_clock_us(target_timestamps, idx, stride=vae_temporal_stride) for idx in range(t_lat)]
    T_world_rig_np = np.stack([interpolate_T_world_rig_at(ego_ts, ego_quat, ego_xyz, ts_us) for ts_us in latent_times], axis=0)
    rig = torch.as_tensor(rig_calibration, dtype=torch.float64)
    if rig.shape != (len(SENSOR_ORDER), 4, 4):
        raise ValueError(f'expected rig_calibration shape ({len(SENSOR_ORDER)}, 4, 4), got {tuple(rig.shape)}')
    T_world_rig = torch.from_numpy(T_world_rig_np).to(dtype=torch.float64)
    T_world_front = T_world_rig @ rig[0]
    T_anchor_front = se3_inverse(T_world_front[0]) @ T_world_front
    T_anchor_front[0] = torch.eye(4, dtype=torch.float64)
    return T_anchor_front.to(dtype=torch.float32)

def se3_log_rotation_angle(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f'expected SE(3) shape (..., 4, 4), got {tuple(T.shape)}')
    R = T[..., :3, :3]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)
