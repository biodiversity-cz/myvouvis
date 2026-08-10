from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.types import Detection

# 1-based IDs follow COCO convention; order matches _ARCHIVAL_NAMES in vendor/lm2.py
ARCHIVAL_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "ruler", "supercategory": "archival"},
    {"id": 2, "name": "barcode", "supercategory": "archival"},
    {"id": 3, "name": "colorcard", "supercategory": "archival"},
    {"id": 4, "name": "label", "supercategory": "archival"},
    {"id": 5, "name": "map", "supercategory": "archival"},
    {"id": 6, "name": "envelope", "supercategory": "archival"},
    {"id": 7, "name": "photo", "supercategory": "archival"},
    {"id": 8, "name": "attached item", "supercategory": "archival"},
    {"id": 9, "name": "weights", "supercategory": "archival"},
]

_CATEGORY_ID: dict[str, int] = {c["name"]: c["id"] for c in ARCHIVAL_CATEGORIES}


def _detection_to_annotation(
    detection: Detection,
    ann_id: int,
    image_id: int,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox
    w = x2 - x1
    h = y2 - y1
    ann: dict[str, Any] = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": _CATEGORY_ID.get(detection.category, 0),
        "bbox": [x1, y1, w, h],
        "area": w * h,
        "iscrowd": 0,
        "score": round(detection.confidence, 5),
    }
    if image_size is not None:
        iw, ih = image_size
        if iw > 0 and ih > 0:
            ann["bbox_normalized"] = [
                round(x1 / iw, 6),
                round(y1 / ih, 6),
                round(w / iw, 6),
                round(h / ih, 6),
            ]
    return ann


def to_coco(
    detections: tuple[Detection, ...],
    file_name: str,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    """Build a single-image COCO-compatible dict from pipeline detections."""
    image_id = 1
    image_entry: dict[str, Any] = {"id": image_id, "file_name": file_name}
    if image_size is not None:
        image_entry["width"] = image_size[0]
        image_entry["height"] = image_size[1]

    annotations = [
        _detection_to_annotation(d, ann_id=i + 1, image_id=image_id, image_size=image_size)
        for i, d in enumerate(detections)
    ]

    return {
        "images": [image_entry],
        "annotations": annotations,
        "categories": ARCHIVAL_CATEGORIES,
    }
