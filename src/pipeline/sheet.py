from __future__ import annotations

from pathlib import Path

from pipeline.deps import PipelineDeps, default_deps
from pipeline.dwc import map_to_dwc
from pipeline.images import materialize_sheet_path
from pipeline.labels import crop_label, select_primary_label
from pipeline.types import DarwinCoreResult, OutputMode, SheetResult


def process_sheet(
    path: Path,
    *,
    deps: PipelineDeps | None = None,
    output_mode: OutputMode = OutputMode.full,
) -> SheetResult:
    d = deps or default_deps()
    with materialize_sheet_path(path) as work_path:
        detections = d.detector(work_path)
        primary = select_primary_label(detections, label_category=d.label_category)
        if output_mode == OutputMode.bbox:
            return SheetResult(
                darwin_core=DarwinCoreResult(dwc={}, validation={"ok": True, "missing": []}),
                detections=detections,
                primary_label=primary,
            )
        crop_bytes, mime = crop_label(work_path, primary)
        raw = d.llm(crop_bytes, mime)
        return SheetResult(
            darwin_core=map_to_dwc(raw, d.settings.dwc_map_path),
            detections=detections,
            primary_label=primary,
        )
