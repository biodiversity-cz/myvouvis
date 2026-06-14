from __future__ import annotations

from pathlib import Path

from bots.base.abstract import AbstractDatabot
from core.domain.DatabotRole import DatabotRole
from pipeline.deps import PipelineDeps
from pipeline.sheet import process_sheet
from utils.types import Score


class HerbariumDwcDatabot(AbstractDatabot):
    NAME = "herbarium-dwc"
    DESCRIPTION = (
        "Herbarium sheet → vision LLM → Darwin Core terms in databot_results "
        "(dwc + validation + LM2 bounding boxes)."
    )
    VERSION = 1
    ROLE = DatabotRole.SCANNER

    def __init__(self, deps: PipelineDeps | None = None):
        self._deps = deps or PipelineDeps.from_settings()
        super().__init__()

    def compute(self, image_local_path: str) -> Score:
        return process_sheet(Path(image_local_path), deps=self._deps).as_score()
