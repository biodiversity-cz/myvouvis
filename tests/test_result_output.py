from __future__ import annotations

from pipeline.types import DarwinCoreResult, Detection, SheetResult


def test_as_score_includes_dwc_validation_and_geometry() -> None:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    other = Detection(bbox=(10, 20, 30, 40), category="barcode", confidence=0.94)
    result = SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis"},
            validation={"ok": True, "missing": []},
        ),
        detections=(other, primary),
        primary_label=primary,
    )
    score = result.as_score()
    assert set(score.keys()) == {"dwc", "validation", "detections", "primary_label"}
    assert score["dwc"]["scientificName"] == "Bellis perennis"
    assert score["detections"] == [
        {"bbox": [10, 20, 30, 40], "category": "barcode", "confidence": 0.94},
        {"bbox": [100, 200, 300, 400], "category": "label", "confidence": 0.88},
    ]
    assert score["primary_label"] == {
        "bbox": [100, 200, 300, 400],
        "category": "label",
        "confidence": 0.88,
    }
