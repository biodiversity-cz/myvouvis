from __future__ import annotations

from pipeline.types import DarwinCoreResult, Detection, OutputMode, SheetResult


def _sample_result() -> SheetResult:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    other = Detection(bbox=(10, 20, 30, 40), category="barcode", confidence=0.94)
    return SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis"},
            validation={"ok": True, "missing": []},
        ),
        detections=(other, primary),
        primary_label=primary,
    )


def test_as_score_full() -> None:
    score = _sample_result().as_score(OutputMode.full)
    assert set(score.keys()) == {"dwc", "validation", "detections", "primary_label"}
    assert score["dwc"]["scientificName"] == "Bellis perennis"


def test_as_score_dwc_only() -> None:
    score = _sample_result().as_score(OutputMode.dwc)
    assert set(score.keys()) == {"dwc", "validation"}
    assert "detections" not in score


def test_as_score_bbox_only() -> None:
    score = _sample_result().as_score(OutputMode.bbox)
    assert set(score.keys()) == {"detections", "primary_label"}
    assert "dwc" not in score
    assert score["primary_label"]["category"] == "label"
