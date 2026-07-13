from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import LONGTAIL_ROOT, DataConfig, RunConfig, TrainConfig
SMOKE_61_CONFIG = replace(TrainConfig(), data=replace(DataConfig(), data_root=LONGTAIL_ROOT / 'data_ft' / 'qwen_frontwide_filter_simplecalib_smoke_2x1' / 'smoke_by_uuid', clip_length=61, num_workers=0, text_emb_cache_root=LONGTAIL_ROOT / 'BEV_WAN' / 'text_cache', text_drop_prob=0.1, max_items=1), run=replace(RunConfig(), stage_a1_steps=10, stage_b_steps=10, save_every=10))
