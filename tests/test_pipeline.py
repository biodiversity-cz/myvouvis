from __future__ import annotations

from pathlib import Path

from pipeline.dwc import map_to_dwc, normalize_iso_date
from pipeline.labels import select_primary_label
from pipeline.types import Detection

_REPO = Path(__file__).resolve().parent.parent
_DWC_MAP = _REPO / "config" / "dwc_map.yaml"


def test_select_primary_label_by_area_times_confidence() -> None:
    large = Detection(bbox=(0, 0, 100, 100), category="label", confidence=0.5)
    small = Detection(bbox=(0, 0, 10, 10), category="label", confidence=0.99)
    assert select_primary_label((small, large)) is large


def test_normalize_iso_date_strips_zero_month_day() -> None:
    assert normalize_iso_date("2003-00-00") == "2003"
    assert normalize_iso_date("1983-08-00") == "1983-08"
    assert normalize_iso_date("1983-08-28") == "1983-08-28"
    assert normalize_iso_date("2003") == "2003"


def test_map_to_dwc_normalizes_date_identified() -> None:
    raw = {"identifiedDate": "2003-00-00", "identifiedBy": "A. B."}
    result = map_to_dwc(raw, _DWC_MAP)
    assert result.dwc["dateIdentified"] == "2003"
    assert result.dwc["identifiedBy"] == "A. B."
    assert result.dwc["labelType"] == "unknown"


def test_map_to_dwc_maps_collected_by() -> None:
    raw = {"collectedBy": "Darwin", "scientificName": "Homo sapiens"}
    result = map_to_dwc(raw, _DWC_MAP)
    assert result.dwc["recordedBy"] == "Darwin"
    assert result.dwc["scientificName"] == "Homo sapiens"
    assert result.dwc["labelType"] == "unknown"
    assert result.validation["ok"] is True


def test_map_to_dwc_preserves_label_type() -> None:
    raw = {"labelType": "mixed", "scientificName": "Melica picta"}
    result = map_to_dwc(raw, _DWC_MAP)
    assert result.dwc["labelType"] == "mixed"
