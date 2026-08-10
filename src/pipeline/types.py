from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pipeline.coco import to_coco


class OutputMode(StrEnum):
    full = "full"
    dwc = "dwc"
    bbox = "bbox"


@dataclass(frozen=True, slots=True)
class Detection:
    """Axis-aligned box in pixel coordinates (xyxy)."""

    bbox: tuple[int, int, int, int]
    category: str
    confidence: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "category": self.category,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class DarwinCoreResult:
    dwc: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SheetResult:
    """Pipeline result persisted to databot_results as DwC + LM2 geometry."""

    darwin_core: DarwinCoreResult
    detections: tuple[Detection, ...]
    primary_label: Detection
    timing: dict[str, float | None] = field(default_factory=dict)
    llm_version: str | None = None
    image_name: str = "sheet.jpg"
    image_size: tuple[int, int] | None = None
    multipage: bool = False
    typus: bool = False

    def as_score(self, output_mode: OutputMode = OutputMode.full) -> dict[str, Any]:
        """Serialize result; shape depends on output_mode."""
        out: dict[str, Any] = {}
        # Always present (detection-time flag; independent of output_mode).
        out["typus"] = self.typus
        if output_mode in (OutputMode.full, OutputMode.dwc):
            out["dwc"] = self.darwin_core.dwc
            out["validation"] = self.darwin_core.validation
            # Always present; derived from dwc.labelType (handwritten|mixed → True).
            out["handwritten"] = _is_handwritten(self.darwin_core.dwc)
        if output_mode in (OutputMode.full, OutputMode.bbox):
            out["detections"] = [d.as_dict() for d in self.detections]
            out["primary_label"] = self.primary_label.as_dict()
            out["coco"] = to_coco(self.detections, self.image_name, self.image_size)
        if self.llm_version is not None:
            out["llm_version"] = self.llm_version
        if self.timing:
            out["timing"] = self.timing
        if self.multipage:
            out["multipage"] = True
        return out


def _is_handwritten(dwc: dict[str, Any]) -> bool:
    return str(dwc.get("labelType", "")).casefold() in {"handwritten", "mixed"}


class Detector(Protocol):
    def __call__(self, image_path: Path) -> tuple[Detection, ...]: ...


class LabelLlm(Protocol):
    def __call__(self, image_bytes: bytes, mime: str) -> dict[str, Any]: ...
