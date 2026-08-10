from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.types import Detection

# OpenCV HSV: H 0–179, S/V 0–255. Milder S/V for faded / aged red labels.
_RED_H_LOW_MAX = 10
_RED_H_HIGH_MIN = 170
_MIN_S = 50
_MIN_V = 50
_MIN_RED_RATIO = 0.25
_MIN_AREA = 500


def any_red_label(
    image_path: Path,
    detections: tuple[Detection, ...],
    *,
    label_category: str = "label",
) -> bool:
    """Return True if any label bbox is predominantly red (typus sticker heuristic)."""
    want = label_category.casefold()
    labels = [d for d in detections if d.category.casefold() == want]
    if not labels:
        return False

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    height, width = rgb.shape[:2]

    for det in labels:
        x1, y1, x2, y2 = _clip_bbox(det.bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if _is_red_crop(crop):
            return True
    return False


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _is_red_crop(crop_rgb: np.ndarray) -> bool:
    h, w = crop_rgb.shape[:2]
    if h * w < _MIN_AREA:
        return False
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (
        ((h_ch <= _RED_H_LOW_MAX) | (h_ch >= _RED_H_HIGH_MIN))
        & (s_ch >= _MIN_S)
        & (v_ch >= _MIN_V)
    )
    return float(np.count_nonzero(red)) / float(h * w) >= _MIN_RED_RATIO
