#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared environment for the inference/training scripts.
#
#   source scripts/common.sh
#
# This file is meant to be *sourced*, not executed. It resolves the repo
# layout, picks the Python interpreter, and exports the model-asset and
# latent-cache environment variables the scripts read at import time. Every
# variable uses ${VAR:-default} so callers can override any of them.
# ---------------------------------------------------------------------------

# --- Resolve repo layout ------------------------------------------------------
_COMMON_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
export REPO_ROOT="${REPO_ROOT:-$( cd -- "${_COMMON_DIR}/../.." &>/dev/null && pwd )}"
export LONGTAIL_ROOT="${LONGTAIL_ROOT:-$( cd -- "${REPO_ROOT}/.." &>/dev/null && pwd )}"

# --- Python interpreter --------------------------------------------------------
# Uses the virtualenv shipped alongside the repo. Override PYTHON to point at
# any env with the same deps.
export PYTHON="${PYTHON:-${LONGTAIL_ROOT}/trajcrafter/bin/python}"

# --- Base model assets (Wan2.1-VACE) -------------------------------------------
# WAN21_VACE_DIR is relative to REPO_ROOT (scripts cd there before launching).
# For the 14B backbone, point WAN21_VACE_DIR at the 14B checkpoint and select
# the matching config.
export WAN21_VACE_DIR="${WAN21_VACE_DIR:-checkpoints/Wan2.1-VACE-1.3B}"
export OPENLONGTAIL_VAE_PATH="${OPENLONGTAIL_VAE_PATH:-${REPO_ROOT}/checkpoints/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth}"
export OPENLONGTAIL_UMT5_PATH="${OPENLONGTAIL_UMT5_PATH:-${REPO_ROOT}/checkpoints/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth}"
export OPENLONGTAIL_UMT5_TOKENIZER="${OPENLONGTAIL_UMT5_TOKENIZER:-${REPO_ROOT}/checkpoints/Wan2.1-VACE-1.3B/google/umt5-xxl}"

# --- Latent cache version stamp ------------------------------------------------
# The dataset loader only accepts clips stamped with this version. Set
# OPENLONGTAIL_CACHE_VERSIONS (comma-separated) to whitelist additional stamps when
# combining multiple caches.
export OPENLONGTAIL_CACHE_VERSION="${OPENLONGTAIL_CACHE_VERSION:-latent_t41_stride1_v1}"

# --- Runtime knobs -------------------------------------------------------------
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Module invocations assume CWD == repo root.
cd "${REPO_ROOT}"
