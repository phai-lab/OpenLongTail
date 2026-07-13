from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from openlongtail.data.latent_cache_dataset import RayLatentCacheDataset, ray_latent_cache_collate

class RayLatentCacheDatasetWarp(RayLatentCacheDataset):

    def __init__(self, config, require_p4_sidecar: bool=True, restrict_to_existing_sidecars: bool=False, require_lookback_sidecar: bool=False) -> None:
        super().__init__(config, require_p4_sidecar=require_p4_sidecar)
        self.require_lookback_sidecar = bool(require_lookback_sidecar)
        if restrict_to_existing_sidecars:
            kept = []
            for entry in self.entries:
                rel = Path(entry['path'])
                cache_path = config.latent_cache_root / rel
                v0 = cache_path.with_name(f'{cache_path.stem}_warp{cache_path.suffix}')
                if not v0.exists():
                    continue
                if self.require_lookback_sidecar:
                    v1 = cache_path.with_name(f'{cache_path.stem}_lookback{cache_path.suffix}')
                    if not v1.exists():
                        continue
                kept.append(entry)
            if not kept:
                raise FileNotFoundError(f'restrict_to_existing_sidecars=True but no usable sidecars found under {config.latent_cache_root}')
            self.entries = kept

    def _sidecar_path(self, idx: int) -> Path:
        rel_path = Path(self.entries[idx]['path'])
        cache_path = self.config.latent_cache_root / rel_path
        return cache_path.with_name(f'{cache_path.stem}_warp{cache_path.suffix}')

    def _adj_sidecar_path(self, idx: int) -> Path:
        rel_path = Path(self.entries[idx]['path'])
        cache_path = self.config.latent_cache_root / rel_path
        return cache_path.with_name(f'{cache_path.stem}_lookback{cache_path.suffix}')

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        sidecar_path = self._sidecar_path(idx)
        if not sidecar_path.exists():
            raise FileNotFoundError(f'warp sidecar missing: {sidecar_path}')
        sidecar = torch.load(sidecar_path, map_location='cpu', weights_only=True)
        warped = sidecar['warped_target_latents']
        vis = sidecar['warped_target_visibility']
        vis_pct = sidecar['warped_target_visibility_pixel_pct']
        if warped.shape != (5, 16, 11, 60, 104):
            raise ValueError(f'unexpected warped_target_latents shape {tuple(warped.shape)} at {sidecar_path}')
        if vis.shape != (5, 1, 11, 60, 104):
            raise ValueError(f'unexpected warped_target_visibility shape {tuple(vis.shape)} at {sidecar_path}')
        if vis_pct.shape != (5,):
            raise ValueError(f'unexpected warped_target_visibility_pixel_pct shape {tuple(vis_pct.shape)} at {sidecar_path}')
        if self.require_lookback_sidecar:
            adj_path = self._adj_sidecar_path(idx)
            if not adj_path.exists():
                raise FileNotFoundError(f'adjacent sidecar missing: {adj_path}')
            adj = torch.load(adj_path, map_location='cpu', weights_only=True)
            warped_adj = adj['adjacent_warped_target_latents']
            vis_adj = adj['adjacent_warped_target_visibility']
            vis_adj_pct = adj['adjacent_warped_target_visibility_pixel_pct']
            if warped_adj.shape != warped.shape:
                raise ValueError(f'adjacent warped shape {tuple(warped_adj.shape)} != v0 shape {tuple(warped.shape)} at {adj_path}')
            if vis_adj.shape != vis.shape:
                raise ValueError(f'adjacent visibility shape {tuple(vis_adj.shape)} != v0 shape {tuple(vis.shape)} at {adj_path}')
            vis_v0_f = vis.float()
            vis_adj_f = vis_adj.float()
            use_adj = vis_adj_f > vis_v0_f
            use_adj_warp = use_adj.expand_as(warped)
            warped_merged = torch.where(use_adj_warp, warped_adj.to(warped.dtype), warped.to(warped.dtype))
            vis_merged = torch.maximum(vis_v0_f, vis_adj_f).to(vis.dtype)
            warped = warped_merged
            vis = vis_merged
            vis_pct = torch.maximum(vis_pct.float(), vis_adj_pct.float()).to(vis_pct.dtype)
            item['warped_target_v0_visibility_pixel_pct'] = sidecar['warped_target_visibility_pixel_pct'].to(torch.float32)
            item['warped_target_adj_visibility_pixel_pct'] = vis_adj_pct.to(torch.float32)
        item['warped_target_latents'] = warped.to(dtype=torch.bfloat16)
        item['warped_target_visibility'] = vis.to(dtype=torch.float16)
        item['warped_target_visibility_pixel_pct'] = vis_pct.to(dtype=torch.float32)
        return item

def ray_latent_cache_collate_warp(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = ray_latent_cache_collate(items)
    if 'warped_target_latents' not in items[0]:
        return batch
    batch['warped_target_latents'] = torch.stack([it['warped_target_latents'] for it in items], dim=0)
    batch['warped_target_visibility'] = torch.stack([it['warped_target_visibility'] for it in items], dim=0)
    batch['warped_target_visibility_pixel_pct'] = torch.stack([it['warped_target_visibility_pixel_pct'] for it in items], dim=0)
    return batch
