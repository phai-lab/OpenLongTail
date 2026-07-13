from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import DataConfig, RunConfig, TrainConfig
STAGE_A0_CONFIG = replace(TrainConfig(), data=replace(DataConfig(), clip_length=33), run=replace(RunConfig(), stage_a0_steps=5000))
