from __future__ import annotations

from pipeline.types import DarwinCoreResult, SheetResult


def test_as_score_is_dwc_only() -> None:
    result = SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis"},
            validation={"ok": True, "missing": []},
        )
    )
    score = result.as_score()
    assert set(score.keys()) == {"dwc", "validation"}
    assert "detections" not in score
    assert "primary_label" not in score
    assert score["dwc"]["scientificName"] == "Bellis perennis"
