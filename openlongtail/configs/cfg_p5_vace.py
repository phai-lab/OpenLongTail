from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.cfg_p4 import P4_CONFIG
P5_VACE_CONFIG = replace(P4_CONFIG, model=replace(P4_CONFIG.model, cross_view_blocks=(5, 11, 17, 23, 29, 35)), run=replace(P4_CONFIG.run, train_log_name='train_log_p5.jsonl'))
P5_SHARED_NOISE_ALPHA = 0.5
P5_SYNC_TEMPORAL_WINDOW = 2
