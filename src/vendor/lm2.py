from __future__ import annotations

import logging
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from pipeline.types import Detection
from settings import Settings

# LM2 Archival_Detector class index → name (ColorProfile__PREP.csv order)
_ARCHIVAL_NAMES = (
    "ruler",
    "barcode",
    "colorcard",
    "label",
    "map",
    "envelope",
    "photo",
    "attached item",
    "weights",
)


def _ensure_vendor_path(lm2_root: Path) -> None:
    """LM2 YOLOv5: component_detector must be importable from lm2_root."""
    root = lm2_root.resolve()
    if not (root / "component_detector" / "detect.py").is_file():
        raise FileNotFoundError(
            f"LM2 component_detector not found under {root} (expected detect.py)."
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


@lru_cache
def _run_detector(
    weights: str,
    conf: float,
    device: str,
    lm2_root: str,
) -> None:
    """Warm-up placeholder: weights path validated on first detect."""
    wp = Path(weights)
    if not wp.is_file():
        raise FileNotFoundError(
            f"LM2 weights not found: {wp}. Expected vendor/lm2/weights/best.pt in the image, or set LM2_WEIGHTS_PATH."
        )
    _ensure_vendor_path(Path(lm2_root))


def detect_archival_labels(image_path: Path, settings: Settings) -> tuple[Detection, ...]:
    weights = settings.default_lm2_weights()
    _run_detector(
        str(weights.resolve()),
        settings.lm2_conf_threshold,
        settings.lm2_device,
        str(settings.resolved_lm2_root().resolve()),
    )

    import tempfile

    from component_detector.detect import run  # noqa: PLC0415

    logger = logging.getLogger("lm2")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        staged = tmp_path / "sheet.jpg"
        shutil.copy2(image_path, staged)
        out_labels = tmp_path / "out" / "labels"
        out_labels.mkdir(parents=True, exist_ok=True)

        run(
            weights=str(weights),
            source=str(staged),
            project=str(tmp_path / "out"),
            name="run",
            imgsz=(1280, 1280),
            conf_thres=settings.lm2_conf_threshold,
            nosave=True,
            save_txt=True,
            save_conf=True,
            anno_type="Archival_Detector",
            mode="LM2",
            exist_ok=True,
            device=settings.lm2_device,
            LOGGER=logger,
        )

        label_file = out_labels / f"{staged.stem}.txt"
        if not label_file.is_file():
            return ()

        from PIL import Image  # noqa: PLC0415

        with Image.open(image_path) as im:
            w, h = im.size

        return _parse_yolo_labels(label_file, w, h)


def _parse_yolo_labels(path: Path, width: int, height: int) -> tuple[Detection, ...]:
    out: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)
        name = _ARCHIVAL_NAMES[cls_id] if 0 <= cls_id < len(_ARCHIVAL_NAMES) else str(cls_id)
        out.append(Detection(bbox=(x1, y1, x2, y2), category=name, confidence=conf))
    return tuple(out)
