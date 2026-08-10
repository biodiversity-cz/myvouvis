from __future__ import annotations

from pipeline.types import DarwinCoreResult, Detection, OutputMode, SheetResult


def _sample_result() -> SheetResult:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    other = Detection(bbox=(10, 20, 30, 40), category="barcode", confidence=0.94)
    return SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis", "labelType": "printed"},
            validation={"ok": True, "missing": []},
        ),
        detections=(other, primary),
        primary_label=primary,
        image_name="herbarium_001.jpg",
        image_size=(4000, 6000),
    )


def test_as_score_full() -> None:
    score = _sample_result().as_score(OutputMode.full)
    assert set(score.keys()) == {
        "dwc",
        "validation",
        "handwritten",
        "detections",
        "primary_label",
        "coco",
    }
    assert score["dwc"]["scientificName"] == "Bellis perennis"
    assert score["handwritten"] is False


def test_as_score_dwc_only() -> None:
    score = _sample_result().as_score(OutputMode.dwc)
    assert set(score.keys()) == {"dwc", "validation", "handwritten"}
    assert "detections" not in score
    assert "coco" not in score
    assert score["handwritten"] is False


def test_as_score_bbox_only() -> None:
    score = _sample_result().as_score(OutputMode.bbox)
    assert set(score.keys()) == {"detections", "primary_label", "coco"}
    assert "dwc" not in score
    assert "handwritten" not in score
    assert score["primary_label"]["category"] == "label"


def test_handwritten_true_for_mixed_and_handwritten() -> None:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    for label_type in ("handwritten", "mixed", "Handwritten"):
        result = SheetResult(
            darwin_core=DarwinCoreResult(
                dwc={"labelType": label_type},
                validation={"ok": True, "missing": []},
            ),
            detections=(primary,),
            primary_label=primary,
        )
        assert result.as_score(OutputMode.dwc)["handwritten"] is True


def test_coco_structure() -> None:
    result = _sample_result()
    score = result.as_score(OutputMode.full)
    coco = score["coco"]
    assert set(coco.keys()) == {"images", "annotations", "categories"}
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert len(coco["categories"]) == 9


def test_coco_bbox_conversion() -> None:
    """Detection xyxy → COCO xywh (+ bbox_normalized)."""
    score = _sample_result().as_score(OutputMode.full)
    anns = {a["category_id"]: a for a in score["coco"]["annotations"]}
    # barcode: bbox=(10,20,30,40) → xywh=[10,20,20,20]
    barcode_ann = anns[2]
    assert barcode_ann["bbox"] == [10, 20, 20, 20]
    assert barcode_ann["area"] == 400
    assert barcode_ann["bbox_normalized"] == [
        round(10 / 4000, 6),
        round(20 / 6000, 6),
        round(20 / 4000, 6),
        round(20 / 6000, 6),
    ]
    # label: bbox=(100,200,300,400) → xywh=[100,200,200,200]
    label_ann = anns[4]
    assert label_ann["bbox"] == [100, 200, 200, 200]
    assert label_ann["area"] == 40000
    assert label_ann["bbox_normalized"] == [
        round(100 / 4000, 6),
        round(200 / 6000, 6),
        round(200 / 4000, 6),
        round(200 / 6000, 6),
    ]


def test_multipage_flag_included_when_true() -> None:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    result = SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis"},
            validation={"ok": True, "missing": []},
        ),
        detections=(primary,),
        primary_label=primary,
        multipage=True,
    )
    for mode in OutputMode:
        assert result.as_score(mode).get("multipage") is True


def test_multipage_flag_absent_by_default() -> None:
    score = _sample_result().as_score(OutputMode.full)
    assert "multipage" not in score


def test_coco_image_size_included() -> None:
    primary = Detection(bbox=(100, 200, 300, 400), category="label", confidence=0.88)
    result = SheetResult(
        darwin_core=DarwinCoreResult(
            dwc={"scientificName": "Bellis perennis"},
            validation={"ok": True, "missing": []},
        ),
        detections=(primary,),
        primary_label=primary,
        image_name="herbarium_001.jpg",
        image_size=(4000, 6000),
    )
    coco = result.as_score(OutputMode.full)["coco"]
    img = coco["images"][0]
    assert img["file_name"] == "herbarium_001.jpg"
    assert img["width"] == 4000
    assert img["height"] == 6000
