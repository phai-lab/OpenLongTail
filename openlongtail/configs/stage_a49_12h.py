from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import DataConfig, RunConfig, TrainConfig
STAGE_A49_12H_CONFIG = replace(TrainConfig(), data=replace(DataConfig(), clip_length=49, max_items=None), run=replace(RunConfig(), stage_a1_steps=300, save_every=50))
