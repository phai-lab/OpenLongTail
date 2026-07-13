#!/usr/bin/env bash
# ===========================================================================
# OpenLongTail — download the base backbone + released checkpoint (Hugging Face)
# ---------------------------------------------------------------------------
#   bash download.sh
# Override targets with CKPT_DIR=... OUT_DIR=... if you want a custom layout.
# ===========================================================================
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT="$( cd -- "${HERE}/.." &>/dev/null && pwd )"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs}"

command -v huggingface-cli >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"

echo "[download] Wan2.1-VACE-1.3B base model -> ${CKPT_DIR}/Wan2.1-VACE-1.3B"
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir "${CKPT_DIR}/Wan2.1-VACE-1.3B"

echo "[download] OpenLongTail-1.3B checkpoint -> ${OUT_DIR}/openlongtail_1p3b"
huggingface-cli download openlongtail/OpenLongTail-1.3B \
  --local-dir "${OUT_DIR}/openlongtail_1p3b"

echo "[download] done."
