#!/usr/bin/env bash
# Verify vendored LM2 archival detector under vendor/lm2/.
# To refresh component_detector from upstream, clone Gene-Weaver/VoucherVision
# elsewhere and rsync vouchervision/component_detector/ into vendor/lm2/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${ROOT}/vendor/lm2"

if [[ ! -f "${DST}/component_detector/detect.py" ]]; then
  echo "Missing ${DST}/component_detector/detect.py" >&2
  echo "Ensure vendor/lm2 is present (git clone + git lfs pull)." >&2
  exit 1
fi

echo "OK: LM2 detector at ${DST}/component_detector"
echo "Weights: ${DST}/weights/best.pt (Git LFS, baked into Docker image)."
echo "Override: export LM2_WEIGHTS_PATH=/path/to/best.pt"
