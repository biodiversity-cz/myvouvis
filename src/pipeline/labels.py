from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from pipeline.exceptions import NoLabelDetected
from pipeline.types import Detection


def select_primary_label(
    detections: tuple[Detection, ...],
    *,
    label_category: str = "label",
) -> Detection:
    want = label_category.casefold()
    labels = [d for d in detections if d.category.casefold() == want]
    if not labels:
        raise NoLabelDetected(f"No detection with category {label_category!r}")
    return max(labels, key=lambda d: d.area * d.confidence)


def crop_label(image_path: Path, detection: Detection) -> tuple[bytes, str]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        x1, y1, x2, y2 = detection.bbox
        crop = im.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "image/jpeg"
