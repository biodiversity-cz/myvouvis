from __future__ import annotations

from pathlib import Path

from pipeline.deps import PipelineDeps, default_deps
from pipeline.dwc import map_to_dwc
from pipeline.labels import crop_label, select_primary_label
from pipeline.types import SheetResult


def process_sheet(path: Path, *, deps: PipelineDeps | None = None) -> SheetResult:
    d = deps or default_deps()
    image_path = path.resolve()
    detections = d.detector(image_path)
    primary = select_primary_label(detections, label_category=d.label_category)
    crop_bytes, mime = crop_label(image_path, primary)
    raw = d.llm(crop_bytes, mime)
    return SheetResult(
        darwin_core=map_to_dwc(raw, d.settings.dwc_map_path),
        detections=detections,
        primary_label=primary,
    )
