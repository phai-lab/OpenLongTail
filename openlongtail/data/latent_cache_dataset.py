from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from openlongtail.data.rig_parquet import SENSOR_ORDER
from openlongtail.data.text_emb_cache import load_text_embedding, load_text_embedding_for_uuid, pad_text_embedding_batch

@dataclass(frozen=True)
class RayLatentCacheDataConfig:
    latent_cache_root: Path
    text_emb_cache_root: Path
    cache_version: str = 'latent_t41_stride4_v1'
    cache_versions: tuple[str, ...] = ()
    text_drop_prob: float = 0.1
    max_items: int | None = None
    index_filename: str = 'index.jsonl'

def _read_index_entries(latent_cache_root: Path, index_filename: str='index.jsonl') -> list[dict[str, Any]]:
    index_path = latent_cache_root / index_filename
    if index_path.exists():
        entries: list[dict[str, Any]] = []
        with index_path.open() as handle:
            for (line_no, line) in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if 'path' not in payload:
                    raise KeyError(f'expected path in {index_path}:{line_no}')
                entries.append(payload)
        return entries
    SIDECAR_SUFFIXES = ('_p4', '_warp', '_lookback')
    files = sorted((path for path in (latent_cache_root / 'per_uuid').glob('*/clip_*.pt') if not any((path.stem.endswith(s) for s in SIDECAR_SUFFIXES))))
    return [{'path': str(path.relative_to(latent_cache_root))} for path in files]

class RayLatentCacheDataset(Dataset[dict[str, Any]]):

    def __init__(self, config: RayLatentCacheDataConfig, require_p4_sidecar: bool=False) -> None:
        self.config = config
        self.require_p4_sidecar = bool(require_p4_sidecar)
        entries = _read_index_entries(config.latent_cache_root, config.index_filename)
        if config.max_items is not None:
            entries = entries[:config.max_items]
        if not entries:
            raise FileNotFoundError(f'no latent cache entries under {config.latent_cache_root}')
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_clip_payload(self, idx: int) -> dict[str, Any]:
        rel_path = Path(self.entries[idx]['path'])
        cache_path = self.config.latent_cache_root / rel_path
        if not cache_path.exists():
            raise FileNotFoundError(f'expected latent cache payload at {cache_path}')
        payload = torch.load(cache_path, map_location='cpu', weights_only=True)
        expected = self.config.cache_version
        actual = payload.get('cache_version')
        if actual != expected and actual not in self.config.cache_versions:
            allowed = (expected, *self.config.cache_versions) if self.config.cache_versions else expected
            raise ValueError(f'expected cache_version in {allowed!r} for {cache_path}, got {actual!r}')
        return payload

    def _sidecar_path(self, idx: int) -> Path:
        rel_path = Path(self.entries[idx]['path'])
        cache_path = self.config.latent_cache_root / rel_path
        return cache_path.with_name(f'{cache_path.stem}_p4{cache_path.suffix}')

    def __getitem__(self, idx: int) -> dict[str, Any]:
        payload = self._load_clip_payload(idx)
        uuid = str(payload['uuid'])
        z_all = payload['z_all'].to(dtype=torch.bfloat16, device='cpu')
        if z_all.ndim != 5 or z_all.shape[0] != len(SENSOR_ORDER) or z_all.shape[1] != 16:
            raise ValueError(f"expected z_all shape (6, 16, T', H', W'), got {tuple(z_all.shape)}")
        if random.random() < self.config.text_drop_prob:
            (text_emb, text_mask) = load_text_embedding(self.config.text_emb_cache_root, 'null')
        else:
            (text_emb, text_mask) = load_text_embedding_for_uuid(self.config.text_emb_cache_root, uuid)
        item: dict[str, Any] = {'uuid': uuid, 'z_all': z_all, 'K': payload['K'].to(dtype=torch.float32, device='cpu'), 'E': payload['E'].to(dtype=torch.float32, device='cpu'), 'text_emb': text_emb, 'text_mask': text_mask, 'view_ids': torch.arange(len(SENSOR_ORDER), dtype=torch.long), 'clip_start_frame': torch.tensor(int(payload['clip_start_frame']), dtype=torch.long), 'target_timestamps': payload['target_timestamps'].to(dtype=torch.long, device='cpu'), 'frame_indices': {sensor_name: payload['frame_indices'][sensor_name].to(dtype=torch.long, device='cpu') for sensor_name in SENSOR_ORDER}, 'timestamp_error': {sensor_name: payload['timestamp_error'][sensor_name].to(dtype=torch.long, device='cpu') for sensor_name in SENSOR_ORDER}, 'cache_path': str(Path(self.entries[idx]['path'])), 'clip_id': torch.tensor(int(payload['clip_id']), dtype=torch.long)}
        if self.require_p4_sidecar:
            sidecar_path = self._sidecar_path(idx)
            if not sidecar_path.exists():
                raise FileNotFoundError(f'P4 sidecar missing: {sidecar_path}')
            sidecar = torch.load(sidecar_path, map_location='cpu', weights_only=True)
            if sorted(sidecar) != ['T_anchor_front']:
                raise ValueError(f'expected P4 sidecar to contain only T_anchor_front, got {sorted(sidecar)} at {sidecar_path}')
            item['T_anchor_front'] = sidecar['T_anchor_front'].to(dtype=torch.float32, device='cpu')
        return item

def ray_latent_cache_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError('expected at least one item to collate')
    (text_emb, text_mask) = pad_text_embedding_batch([(item['text_emb'], item['text_mask']) for item in items])
    batch: dict[str, Any] = {'uuid': [item['uuid'] for item in items], 'z_all': torch.stack([item['z_all'] for item in items], dim=0), 'K': torch.stack([item['K'] for item in items], dim=0), 'E': torch.stack([item['E'] for item in items], dim=0), 'text_emb': text_emb, 'text_mask': text_mask, 'view_ids': torch.stack([item['view_ids'] for item in items], dim=0), 'clip_start_frame': torch.stack([item['clip_start_frame'] for item in items], dim=0), 'target_timestamps': torch.stack([item['target_timestamps'] for item in items], dim=0), 'frame_indices': {sensor_name: torch.stack([item['frame_indices'][sensor_name] for item in items], dim=0) for sensor_name in SENSOR_ORDER}, 'timestamp_error': {sensor_name: torch.stack([item['timestamp_error'][sensor_name] for item in items], dim=0) for sensor_name in SENSOR_ORDER}, 'cache_path': [item['cache_path'] for item in items], 'clip_id': torch.stack([item['clip_id'] for item in items], dim=0)}
    if 'T_anchor_front' in items[0]:
        batch['T_anchor_front'] = torch.stack([item['T_anchor_front'] for item in items], dim=0)
    return batch
