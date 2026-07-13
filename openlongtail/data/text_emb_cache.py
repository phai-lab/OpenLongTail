from __future__ import annotations
from pathlib import Path
import torch

def _extract_embedding_payload(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_path, map_location='cpu', weights_only=True)
    emb_key = 'text_emb' if 'text_emb' in payload else 'emb'
    mask_key = 'text_mask' if 'text_mask' in payload else 'mask' if 'mask' in payload else 'attention_mask'
    if emb_key not in payload:
        raise KeyError(f'expected text_emb or emb in {cache_path}')
    if mask_key not in payload:
        raise KeyError(f'expected text_mask, mask, or attention_mask in {cache_path}')
    emb = payload[emb_key].to(dtype=torch.bfloat16, device='cpu')
    mask = payload[mask_key].to(dtype=torch.bool, device='cpu')
    return (emb, mask)

def load_text_embedding(cache_root: Path, name: str='null') -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = cache_root / f'{name}.pt'
    if not cache_path.exists():
        raise FileNotFoundError(f'expected text embedding cache at {cache_path}')
    return _extract_embedding_payload(cache_path)

def load_text_embedding_for_uuid(cache_root: Path, uuid: str, fallback_name: str='null') -> tuple[torch.Tensor, torch.Tensor]:
    per_uuid_path = cache_root / 'per_uuid' / f'{uuid}.pt'
    if per_uuid_path.exists():
        return _extract_embedding_payload(per_uuid_path)
    return load_text_embedding(cache_root, fallback_name)

def pad_text_embedding_batch(items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError('expected at least one text embedding to pad')
    hidden_dim = items[0][0].shape[-1]
    dtype = items[0][0].dtype
    max_len = max((emb.shape[0] for (emb, _) in items))
    batch_emb = torch.zeros(len(items), max_len, hidden_dim, dtype=dtype)
    batch_mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    for (item_idx, (emb, mask)) in enumerate(items):
        if emb.ndim != 2:
            raise ValueError(f'expected text_emb shape (L, D), got {tuple(emb.shape)}')
        if mask.ndim != 1:
            raise ValueError(f'expected text_mask shape (L,), got {tuple(mask.shape)}')
        if emb.shape[0] != mask.shape[0]:
            raise ValueError(f'expected text_emb/text_mask same length, got {emb.shape[0]} and {mask.shape[0]}')
        if emb.shape[1] != hidden_dim:
            raise ValueError(f'expected text_emb hidden dim {hidden_dim}, got {emb.shape[1]}')
        length = emb.shape[0]
        batch_emb[item_idx, :length] = emb.to(dtype=dtype, device='cpu')
        batch_mask[item_idx, :length] = mask.to(dtype=torch.bool, device='cpu')
    return (batch_emb, batch_mask)
