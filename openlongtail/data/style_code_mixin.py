from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import torch
from torch.utils.data import Dataset
from openlongtail.configs.openlongtail_style_vace import STYLE_AXIS_CARDINALITIES, STYLE_UNKNOWN_INDICES

def _coerce_style_ids(payload: Any, num_axes: int) -> torch.Tensor | None:
    if isinstance(payload, dict):
        payload = payload.get('style_ids', None)
    if payload is None:
        return None
    try:
        ids = torch.as_tensor(payload, dtype=torch.long).flatten()
    except (TypeError, ValueError, RuntimeError):
        return None
    if ids.numel() != num_axes:
        return None
    return ids.clone()

class StyleCodeDatasetWrapper(Dataset[dict[str, Any]]):

    def __init__(self, inner: Dataset, style_cache_root: Path | str, axis_cardinalities: tuple[int, ...]=STYLE_AXIS_CARDINALITIES, unknown_indices: tuple[int, ...]=STYLE_UNKNOWN_INDICES, strict: bool=False) -> None:
        self.inner = inner
        self.style_cache_root = Path(style_cache_root)
        self.axis_cardinalities = tuple((int(c) for c in axis_cardinalities))
        self.unknown_indices = tuple((int(i) for i in unknown_indices))
        if len(self.unknown_indices) != len(self.axis_cardinalities):
            raise ValueError('unknown_indices must match axis_cardinalities length')
        self.num_axes = len(self.axis_cardinalities)
        self.strict = bool(strict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__['inner'], name)

    def __len__(self) -> int:
        return len(self.inner)

    def _unknown_ids(self) -> torch.Tensor:
        return torch.tensor(self.unknown_indices, dtype=torch.long)

    def _style_path_for_uuid(self, uuid: str) -> Path:
        return self.style_cache_root / 'per_uuid' / f'{uuid}.pt'

    def load_style_ids(self, uuid: str) -> torch.Tensor:
        path = self._style_path_for_uuid(uuid)
        if not path.exists():
            if self.strict:
                raise FileNotFoundError(f'style-cache entry missing: {path}')
            return self._unknown_ids()
        payload = torch.load(path, map_location='cpu', weights_only=True)
        ids = _coerce_style_ids(payload, self.num_axes)
        if ids is None:
            if self.strict:
                raise ValueError(f'malformed style-cache entry at {path}')
            return self._unknown_ids()
        for axis in range(self.num_axes):
            if not 0 <= int(ids[axis]) < self.axis_cardinalities[axis]:
                ids[axis] = self.unknown_indices[axis]
        return ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.inner[idx]
        uuid = str(item.get('uuid', ''))
        item['style_ids'] = self.load_style_ids(uuid) if uuid else self._unknown_ids()
        return item

def style_collate(inner_collate: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:

    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        batch = inner_collate(items)
        if 'style_ids' in items[0]:
            batch['style_ids'] = torch.stack([it['style_ids'] for it in items], dim=0).to(torch.long)
        return batch
    return _collate
