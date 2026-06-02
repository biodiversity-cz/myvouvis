from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline.llm import extract_label_json
from pipeline.types import Detection, LabelLlm
from settings import Settings, get_settings
from vendor import lm2


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    settings: Settings
    detector: Callable[[Path], tuple[Detection, ...]]
    llm: LabelLlm
    label_category: str

    @staticmethod
    def from_settings(settings: Settings | None = None) -> PipelineDeps:
        s = settings or get_settings()

        def detector(path: Path) -> tuple[Detection, ...]:
            return lm2.detect_archival_labels(path, s)

        def llm(image_bytes: bytes, mime: str) -> dict:
            return extract_label_json(image_bytes, mime, s)

        return PipelineDeps(
            settings=s,
            detector=detector,  # type: ignore[arg-type]
            llm=llm,
            label_category=s.label_category,
        )


@lru_cache
def default_deps() -> PipelineDeps:
    return PipelineDeps.from_settings()
