#!/usr/bin/env bash
# ============================================================================
# OpenLongTail — shared environment. Source this before any script.
#   source scripts/env.sh
# Sets the checkpoint / backbone paths the config reads via env vars, plus the
# Wan2.1 code root on PYTHONPATH. Override any of these before sourcing if your
# layout differs.
# ============================================================================

# Repo root (works whether sourced or executed)
OLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export OLT_ROOT

# --- where download.sh puts things (override to relocate the big weights) ---
export OLT_CKPT_ROOT="${OLT_CKPT_ROOT:-$OLT_ROOT/checkpoints}"
export OLT_WM_DIR="${OLT_WM_DIR:-$OLT_CKPT_ROOT/openlongtail-ckpt}"
export OLT_WAN_VACE_DIR="${OLT_WAN_VACE_DIR:-$OLT_CKPT_ROOT/Wan2.1-VACE-1.3B}"
export OLT_WAN21_CODE_ROOT="${OLT_WAN21_CODE_ROOT:-$OLT_CKPT_ROOT/Wan2.1}"
# Wan2.2 code: the VAE loader (openlongtail/models/wan_vae.py) imports
# wan.modules.vae2_1 from a Wan2.2 checkout placed as a sibling of the
# openlongtail package (it resolves $OLT_ROOT/Wan2.2). Wan2.1 ships only the
# older vae.py, so Wan2.2 is required even though the DiT is 2.1-VACE.
export OLT_WAN22_CODE_ROOT="${OLT_WAN22_CODE_ROOT:-$OLT_CKPT_ROOT/Wan2.2}"

# --- config-read env vars (openlongtail/configs/default.py) ---
export OPENLONGTAIL_VAE_PATH="$OLT_WAN_VACE_DIR/Wan2.1_VAE.pth"
export OPENLONGTAIL_UMT5_PATH="$OLT_WAN_VACE_DIR/models_t5_umt5-xxl-enc-bf16.pth"
export OPENLONGTAIL_UMT5_TOKENIZER="$OLT_WAN_VACE_DIR/google/umt5-xxl"
export OPENLONGTAIL_WAN21_CODE_ROOT="$OLT_WAN21_CODE_ROOT"
export OPENLONGTAIL_WAN21_VACE_1P3B_DIR="$OLT_WAN_VACE_DIR"
# The shared UMT5 'null' embedding (CFG's unconditional branch) is bundled with
# the demo clips; without this the config falls back to a hardcoded BEV_WAN path.
export OPENLONGTAIL_TEXT_EMB_ROOT="${OPENLONGTAIL_TEXT_EMB_ROOT:-$OLT_ROOT/demo/cached/text_cache}"

# The VAE loader resolves Wan2.2 as $OLT_ROOT/Wan2.2 (sibling of the package).
# If the actual checkout lives elsewhere (e.g. under checkpoints/), link it in.
if [ -d "$OLT_WAN22_CODE_ROOT/wan" ] && [ ! -e "$OLT_ROOT/Wan2.2/wan" ]; then
  ln -sfn "$OLT_WAN22_CODE_ROOT" "$OLT_ROOT/Wan2.2" 2>/dev/null || true
fi

# --- import path: the openlongtail package + the Wan2.x code it depends on ---
export PYTHONPATH="$OLT_ROOT:$OLT_WAN21_CODE_ROOT:$OLT_WAN22_CODE_ROOT:${PYTHONPATH:-}"

# --- runtime niceties ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- THE §3 RECIPE (do not change — this is what makes the checkpoint good) ---
# Shared core (accepted by BOTH inference_cached and inference).
export OLT_RECIPE="--cross-guide 3.5 --rear-guide 7.0 --num-steps 50 --start-sigma 1.0 --shared-noise-alpha 0.5 --seed 20260710"
# Raw-tier only: the cached path bakes splatting into the _warp.pt sidecar and
# does NOT accept --splat-radius; the raw path (inference.py) needs it.
export OLT_SPLAT="--splat-radius 1"
