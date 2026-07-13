from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.cfg_p3 import P3_CONFIG
P4_CONFIG = replace(P3_CONFIG, run=replace(P3_CONFIG.run, train_log_name='train_log_p4.jsonl'))
