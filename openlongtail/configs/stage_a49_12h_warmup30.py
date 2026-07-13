from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import MultiViewOptimConfig
from openlongtail.configs.stage_a49_12h import STAGE_A49_12H_CONFIG
STAGE_A49_12H_WARMUP30_CONFIG = replace(STAGE_A49_12H_CONFIG, optim=replace(MultiViewOptimConfig(), warmup_steps=30))
