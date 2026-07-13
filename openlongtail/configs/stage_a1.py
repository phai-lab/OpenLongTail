from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import DataConfig, RunConfig, TrainConfig
STAGE_A1_CONFIG = replace(TrainConfig(), data=replace(DataConfig(), clip_length=81), run=replace(RunConfig(), stage_a1_steps=20000))
