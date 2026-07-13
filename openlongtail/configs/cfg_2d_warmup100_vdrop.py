from __future__ import annotations
from dataclasses import replace
from openlongtail.configs.default import DataConfig, MultiViewModelConfig, MultiViewOptimConfig, RunConfig, TrainConfig
CFG_2D_WARMUP100_VDROP_CONFIG = replace(TrainConfig(), model=replace(MultiViewModelConfig(), view_dropout_active_targets=3), optim=replace(MultiViewOptimConfig(), warmup_steps=100), data=replace(DataConfig(), clip_length=41, max_items=None), run=replace(RunConfig(), stage_a1_steps=3000, save_every=500))
