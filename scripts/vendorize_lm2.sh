#!/usr/bin/env bash
# Copy only LM2 archival YOLOv5 detector from VoucherVision into vendor/lm2.
# After this, you can remove the VoucherVision/ tree from the repo (keep weights on disk).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_VV="${ROOT}/VoucherVision/vouchervision"
DST="${ROOT}/vendor/lm2"

if [[ ! -f "${SRC_VV}/component_detector/detect.py" ]]; then
  echo "Missing ${SRC_VV}/component_detector — is VoucherVision present?" >&2
  exit 1
fi

mkdir -p "${DST}"
rsync -a --delete \
  "${SRC_VV}/component_detector/" \
  "${DST}/component_detector/"

echo "LM2 detector → ${DST}/component_detector"
echo ""
echo "Weights: vendor/lm2/weights/best.pt (Git LFS, baked into Docker image)."
echo "Override: export LM2_WEIGHTS_PATH=/path/to/best.pt"
