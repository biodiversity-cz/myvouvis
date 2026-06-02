from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


@dataclass(frozen=True, slots=True)
class DarwinCoreResult:
    dwc: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SheetResult:
    """Internal pipeline state; only DwC is persisted to databot_results."""

    darwin_core: DarwinCoreResult

    def as_score(self) -> dict[str, Any]:
        """PostgreSQL result_data: DwC terms + lightweight validation only."""
        return {
            "dwc": self.darwin_core.dwc,
            "validation": self.darwin_core.validation,
        }


class Detector(Protocol):
    def __call__(self, image_path: Path) -> tuple[Detection, ...]: ...


class LabelLlm(Protocol):
    def __call__(self, image_bytes: bytes, mime: str) -> dict[str, Any]: ...
