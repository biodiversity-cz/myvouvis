from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from pipeline.deps import PipelineDeps, default_deps
from pipeline.dwc import map_to_dwc
from pipeline.images import is_multipage_tiff, is_tiff_path, materialize_sheet_path
from pipeline.labels import crop_label, select_primary_label
from pipeline.types import DarwinCoreResult, OutputMode, SheetResult
from pipeline.typus import any_red_label


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size  # (width, height)


def process_sheet(
    path: Path,
    *,
    deps: PipelineDeps | None = None,
    output_mode: OutputMode = OutputMode.full,
    image_name: str | None = None,
) -> SheetResult:
    d = deps or default_deps()
    display_name = image_name or path.name
    multipage = is_tiff_path(path) and is_multipage_tiff(path)
    t_start = time.perf_counter()
    with materialize_sheet_path(path) as work_path:
        t0 = time.perf_counter()
        detections = d.detector(work_path)
        detection_s = time.perf_counter() - t0

        primary = select_primary_label(detections, label_category=d.label_category)
        image_size = _image_size(work_path)
        typus = any_red_label(work_path, detections, label_category=d.label_category)

        if output_mode == OutputMode.bbox:
            return SheetResult(
                darwin_core=DarwinCoreResult(dwc={}, validation={"ok": True, "missing": []}),
                detections=detections,
                primary_label=primary,
                image_name=display_name,
                image_size=image_size,
                multipage=multipage,
                typus=typus,
                timing={
                    "detection_s": round(detection_s, 3),
                    "llm_s": None,
                    "total_s": round(time.perf_counter() - t_start, 3),
                },
            )

        crop_bytes, mime = crop_label(work_path, primary)

        t0 = time.perf_counter()
        raw = d.llm(crop_bytes, mime)
        llm_s = time.perf_counter() - t0

        return SheetResult(
            darwin_core=map_to_dwc(raw, d.settings.dwc_map_path),
            detections=detections,
            primary_label=primary,
            image_name=display_name,
            image_size=image_size,
            multipage=multipage,
            typus=typus,
            llm_version=d.settings.resolved_llm_model(),
            timing={
                "detection_s": round(detection_s, 3),
                "llm_s": round(llm_s, 3),
                "total_s": round(time.perf_counter() - t_start, 3),
            },
        )
