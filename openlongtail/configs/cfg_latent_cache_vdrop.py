from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import REPO_ROOT, LatentCacheConfig
from openlongtail.configs.cfg_12h_warmup30_vdrop import CFG_12H_WARMUP30_VDROP_CONFIG
LATENT_CACHE_VDROP_CONFIG = replace(CFG_12H_WARMUP30_VDROP_CONFIG, latent_cache=replace(LatentCacheConfig(), use_latent_cache=True, latent_cache_root=REPO_ROOT / 'cache' / 'latent_t41_stride4_v1'))
