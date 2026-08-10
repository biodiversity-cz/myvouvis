from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pipeline.deps import PipelineDeps
from pipeline.sheet import process_sheet
from pipeline.types import Detection, OutputMode
from settings import Settings


def test_process_sheet_bbox_skips_llm(tmp_path: Path) -> None:
    img = tmp_path / "sheet.png"
    Image.new("RGB", (200, 200), color="white").save(img)
    label = Detection(bbox=(10, 10, 50, 50), category="label", confidence=0.9)

    llm_calls = 0

    def detector(_path: Path) -> tuple[Detection, ...]:
        return (label,)

    def llm(_bytes: bytes, _mime: str) -> dict:
        nonlocal llm_calls
        llm_calls += 1
        return {"scientificName": "X"}

    settings = Settings(openai_api_key="test")
    deps = PipelineDeps(
        settings=settings,
        detector=detector,
        llm=llm,
        label_category="label",
    )

    result = process_sheet(img, deps=deps, output_mode=OutputMode.bbox)

    assert llm_calls == 0
    score = result.as_score(OutputMode.bbox)
    assert score["primary_label"]["bbox"] == [10, 10, 50, 50]
    assert "detections" in score
    assert "coco" in score
    assert score["coco"]["images"][0]["file_name"] == "sheet.png"
    assert "bbox_normalized" in score["coco"]["annotations"][0]
