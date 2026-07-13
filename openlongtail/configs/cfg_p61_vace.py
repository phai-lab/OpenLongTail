from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.cfg_p6_vace import P6_VACE_CONFIG
P61_VACE_CONFIG = replace(P6_VACE_CONFIG, model=replace(P6_VACE_CONFIG.model, sigma_shift=3.0, sample_shift=3.0), run=replace(P6_VACE_CONFIG.run, train_log_name='train_log_p61.jsonl'))
P61_VACE_1P3B_CONFIG = replace(P61_VACE_CONFIG, model=replace(P61_VACE_CONFIG.model, wan_dim=1536, wan_num_heads=12, wan_num_layers=30, wan_ffn_dim=8960, cross_view_blocks=(4, 9, 14, 19, 24, 29), cross_view_heads=12), run=replace(P61_VACE_CONFIG.run, train_log_name='train_log_p61_1p3b.jsonl'))
P61_FRONT_CONDITION_NOISE_PROB = 0.05
P61_FRONT_CONDITION_NOISE_MIN = 0.005
P61_FRONT_CONDITION_NOISE_MAX = 0.03
P61_NEIGHBOR_CONDITION_NOISE_PROB = 0.3
P61_NEIGHBOR_CONDITION_NOISE_MIN = 0.02
P61_NEIGHBOR_CONDITION_NOISE_MAX = 0.15
P61_SYNC_TEMPORAL_WINDOW = 2
P61_CONDITION_ENCODER_LAYERS = 2
P61_SEMANTIC_QUERIES = 64
