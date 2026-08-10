from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pipeline.types import Detection
from pipeline.typus import any_red_label


def test_any_red_label_true_for_red_bbox(tmp_path: Path) -> None:
    img = tmp_path / "sheet.png"
    # White background, solid red sticker region
    im = Image.new("RGB", (200, 200), color=(255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            im.putpixel((x, y), (220, 20, 20))
    im.save(img)

    det = Detection(bbox=(20, 20, 80, 80), category="label", confidence=0.9)
    assert any_red_label(img, (det,)) is True


def test_any_red_label_false_for_beige_label(tmp_path: Path) -> None:
    img = tmp_path / "sheet.png"
    im = Image.new("RGB", (200, 200), color=(255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            im.putpixel((x, y), (240, 230, 200))
    im.save(img)

    det = Detection(bbox=(20, 20, 80, 80), category="label", confidence=0.9)
    assert any_red_label(img, (det,)) is False


def test_any_red_label_false_when_no_labels(tmp_path: Path) -> None:
    img = tmp_path / "sheet.png"
    Image.new("RGB", (50, 50), color=(255, 0, 0)).save(img)
    barcode = Detection(bbox=(0, 0, 50, 50), category="barcode", confidence=0.9)

    with patch("pipeline.typus.Image.open") as open_mock:
        assert any_red_label(img, (barcode,)) is False
        open_mock.assert_not_called()

    assert any_red_label(img, ()) is False


def test_any_red_label_clips_out_of_bounds_bbox(tmp_path: Path) -> None:
    img = tmp_path / "sheet.png"
    im = Image.new("RGB", (100, 100), color=(255, 255, 255))
    for x in range(70, 100):
        for y in range(70, 100):
            im.putpixel((x, y), (220, 20, 20))
    im.save(img)

    # Extends past image edge; clipped crop should still be mostly red
    det = Detection(bbox=(70, 70, 150, 150), category="label", confidence=0.9)
    assert any_red_label(img, (det,)) is True
