#!/usr/bin/env bash
# ============================================================================
# OpenLongTail — fetch everything needed for inference.
#   bash scripts/download.sh              # weights + Wan2.1 code + demo clips
#   bash scripts/download.sh --no-demo    # skip the cached demo dataset
#   bash scripts/download.sh --weights    # only the model weights
#
# Downloads (into $OLT_CKPT_ROOT, default ./checkpoints):
#   1. OpenLongTail WM 1.3B ckpt          (~0.4 GB)  HF model
#   2. Wan2.1-VACE-1.3B backbone          (~18 GB)   HF model (VAE + UMT5 + DiT)
#   3. Wan2.1 inference code              (git)      cross-view attention modules
# And the one-click demo dataset (into ./demo/cached, ~0.1 GB) unless --no-demo.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "$HERE/scripts/env.sh"

WANT_WEIGHTS=1 WANT_CODE=1 WANT_DEMO=1
for a in "$@"; do case "$a" in
  --no-demo) WANT_DEMO=0 ;;
  --weights) WANT_CODE=0; WANT_DEMO=0 ;;
  *) echo "unknown arg: $a"; exit 2 ;;
esac; done

# HF repo ids (override via env if you mirror them)
WM_REPO="${OLT_WM_REPO:-luuuulinnnn/openlongtail-ckpt}"
VACE_REPO="${OLT_VACE_REPO:-Wan-AI/Wan2.1-VACE-1.3B}"
DEMO_REPO="${OLT_DEMO_REPO:-luuuulinnnn/openlongtail-demo-clips}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found. $2"; exit 1; }; }
need git "Install git."
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found; using 'python -m huggingface_hub' fallback."
  HF() { python -m huggingface_hub download "$@"; }
else
  HF() { huggingface-cli download "$@"; }
fi

mkdir -p "$OLT_CKPT_ROOT"

if [ "$WANT_WEIGHTS" = 1 ]; then
  echo "== [1/3] OpenLongTail WM 1.3B ckpt -> $OLT_WM_DIR =="
  HF "$WM_REPO" --local-dir "$OLT_WM_DIR"
  echo "== [2/3] Wan2.1-VACE-1.3B backbone (~18 GB) -> $OLT_WAN_VACE_DIR =="
  HF "$VACE_REPO" --local-dir "$OLT_WAN_VACE_DIR"
fi

if [ "$WANT_CODE" = 1 ]; then
  echo "== [3/3] Wan2.1 + Wan2.2 inference code =="
  # Wan2.1: the VACE DiT backbone modules.
  if [ -d "$OLT_WAN21_CODE_ROOT/wan" ]; then
    echo "   Wan2.1 already present, skipping clone"
  else
    git clone --depth 1 https://github.com/Wan-Video/Wan2.1 "$OLT_WAN21_CODE_ROOT"
  fi
  # Wan2.2: the VAE loader imports wan.modules.vae2_1 from here (Wan2.1 ships
  # only the older vae.py). env.sh links $OLT_ROOT/Wan2.2 -> this checkout.
  if [ -d "$OLT_WAN22_CODE_ROOT/wan" ]; then
    echo "   Wan2.2 already present, skipping clone"
  else
    git clone --depth 1 https://github.com/Wan-Video/Wan2.2 "$OLT_WAN22_CODE_ROOT"
  fi
  ln -sfn "$OLT_WAN22_CODE_ROOT" "$OLT_ROOT/Wan2.2" 2>/dev/null || true
fi

if [ "$WANT_DEMO" = 1 ]; then
  if [ -f "$HERE/demo/cached/text_cache/null.pt" ]; then
    echo "== demo clips already bundled in-repo ($HERE/demo/cached) — skipping =="
  else
    echo "== demo clips -> $HERE/demo =="
    HF "$DEMO_REPO" --repo-type dataset --local-dir "$HERE/demo"
  fi
fi

echo
echo "DONE. Sanity check:"
echo "  WM checkpoint : $([ -f "$OLT_WM_DIR/checkpoint/shared_modules.pt" ] && echo OK || echo MISSING)"
echo "  VAE           : $([ -f "$OPENLONGTAIL_VAE_PATH" ] && echo OK || echo MISSING)"
echo "  UMT5          : $([ -f "$OPENLONGTAIL_UMT5_PATH" ] && echo OK || echo MISSING)"
echo "  Wan2.1 code   : $([ -d "$OLT_WAN21_CODE_ROOT/wan" ] && echo OK || echo MISSING)"
echo "  Wan2.2 code   : $([ -f "$OLT_WAN22_CODE_ROOT/wan/modules/vae2_1.py" ] && echo OK || echo MISSING)"
echo "  demo clips    : $([ -f "$HERE/demo/cached/text_cache/null.pt" ] && echo OK || echo MISSING)"
echo
echo "Next:  bash scripts/demo.sh"
