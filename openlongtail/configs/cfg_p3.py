from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import DataConfig, MultiViewModelConfig, MultiViewOptimConfig, RunConfig, TrainConfig
P3_CONFIG = replace(TrainConfig(), model=replace(MultiViewModelConfig(), view_dropout_active_targets=3), optim=replace(MultiViewOptimConfig(), warmup_steps=10), data=replace(DataConfig(), clip_length=41, max_items=None, num_workers=0), run=replace(RunConfig(), stage_a1_steps=10, save_every=10))
