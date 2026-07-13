#!/usr/bin/env bash
# ===========================================================================
# OpenLongTail — surround-view inference
# ---------------------------------------------------------------------------
# Generate the 5 synchronized non-front views (Cross-Left, Cross-Right,
# Rear-Left, Rear-Right, Rear-Tele) from a single front-camera clip. Each clip
# produces the 5 generated views, the front input, and a preview grid.
#
#   bash infer.sh --input <FRONT_CLIPS> --output <OUT_DIR>            # raw dash-cam clips
#   bash infer.sh --input <CACHE_DIR>   --output <OUT_DIR> --cached   # preprocessed clips
#
# Override the model with:  MODEL=/path/to/checkpoint bash infer.sh ...
# ===========================================================================
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
source "${HERE}/common.sh"

INPUT=""; OUTPUT="outputs/openlongtail_infer"; MODE="dashcam"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="$2";  shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --cached) MODE="cached"; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done
[[ -z "${INPUT}" ]] && { echo "usage: bash infer.sh --input <front_clips> --output <out_dir> [--cached]"; exit 1; }

# --- release defaults (kept out of the way) --------------------------------
MODEL="${MODEL:-${REPO_ROOT}/outputs/openlongtail_1p3b}"
CONFIG="${CONFIG:-openlongtail_1p3b}"
NUM_STEPS="${NUM_STEPS:-50}"; CROSS_GUIDE="${CROSS_GUIDE:-3.5}"; REAR_GUIDE="${REAR_GUIDE:-7.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6,max_split_size_mb:512}"
mkdir -p "${OUTPUT}"

if [[ "${MODE}" == "cached" ]]; then
  "${PYTHON}" -m openlongtail.scripts.inference_cached \
    --v3-root "${INPUT}" --output-dir "${OUTPUT}" \
    --checkpoint-dir "${MODEL}" --config "${CONFIG}" --wan21-vace-dir "${WAN21_VACE_DIR}" \
    --num-steps "${NUM_STEPS}" --start-sigma 1.0 \
    --cross-guide "${CROSS_GUIDE}" --rear-guide "${REAR_GUIDE}" \
    --shared-noise-alpha 0.5 --device cuda
else
  CAPTIONS="${OUTPUT}/captions"; mkdir -p "${CAPTIONS}"
  "${PYTHON}" -m openlongtail.scripts.build_captions \
    --test-data-root "${INPUT}" --out-root "${CAPTIONS}" --frames 4 --device cuda
  "${PYTHON}" -m openlongtail.scripts.inference \
    --test-data-root "${INPUT}" --treat-as-nv --output-dir "${OUTPUT}" \
    --checkpoint-dir "${MODEL}" --config "${CONFIG}" --wan21-vace-dir "${WAN21_VACE_DIR}" \
    --latent-cache-root "${LATENT_CACHE_ROOT}" --caption-cache "${CAPTIONS}" \
    --num-steps "${NUM_STEPS}" --start-sigma 1.0 \
    --cross-guide "${CROSS_GUIDE}" --rear-guide "${REAR_GUIDE}" \
    --shared-noise-alpha 0.5 --splat-radius 1 --device cuda
fi
echo "[OpenLongTail] inference done -> ${OUTPUT}"
