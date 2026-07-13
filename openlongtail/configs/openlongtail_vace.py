from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.cfg_p61_vace import P61_VACE_1P3B_CONFIG, P61_VACE_CONFIG
OPENLONGTAIL_BASE_CONFIG = replace(P61_VACE_CONFIG, run=replace(P61_VACE_CONFIG.run, train_log_name='train_log.jsonl'))
OPENLONGTAIL_1P3B_CONFIG = replace(P61_VACE_1P3B_CONFIG, run=replace(P61_VACE_1P3B_CONFIG.run, train_log_name='train_log_1p3b.jsonl'))
OPENLONGTAIL_14B_CONFIG = replace(P61_VACE_CONFIG, model=replace(P61_VACE_CONFIG.model, wan_dim=5120, wan_num_heads=40, wan_num_layers=40, wan_ffn_dim=13824, cross_view_blocks=(4, 9, 14, 19, 24, 29, 34, 39), cross_view_heads=40), run=replace(P61_VACE_CONFIG.run, train_log_name='train_log_14b.jsonl'))
GRAPH_GATE_INIT_BIAS = -1.4
GEO_HEAD_DIM = 16
GEO_PROJECTION_TEMPERATURE = 4.0
WARP_TARGET_VIEW_IDS = (1, 2)
