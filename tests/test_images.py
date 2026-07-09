from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from pipeline.images import is_multipage_tiff, materialize_sheet_path


def _write_pyramid_tiff(path: Path, large: tuple[int, int], small: tuple[int, int]) -> None:
    big = np.zeros((large[1], large[0], 3), dtype=np.uint8)
    thumb = np.zeros((small[1], small[0], 3), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tif:
        tif.write(big, photometric="rgb", subifds=1)
        tif.write(thumb, photometric="rgb")


def _write_multipage_tiff(path: Path, n_pages: int = 2) -> None:
    with tifffile.TiffWriter(path) as tif:
        for _ in range(n_pages):
            tif.write(np.zeros((100, 100, 3), dtype=np.uint8), photometric="rgb")


def test_is_multipage_tiff_true(tmp_path: Path) -> None:
    tiff = tmp_path / "multi.tif"
    _write_multipage_tiff(tiff, n_pages=2)
    assert is_multipage_tiff(tiff) is True


def test_is_multipage_tiff_false_for_pyramid(tmp_path: Path) -> None:
    tiff = tmp_path / "pyramid.tif"
    _write_pyramid_tiff(tiff, large=(400, 300), small=(100, 100))
    assert is_multipage_tiff(tiff) is False


def test_is_multipage_tiff_false_for_single_page(tmp_path: Path) -> None:
    tiff = tmp_path / "single.tif"
    _write_multipage_tiff(tiff, n_pages=1)
    assert is_multipage_tiff(tiff) is False


def test_is_multipage_tiff_false_for_non_tiff(tmp_path: Path) -> None:
    png = tmp_path / "sheet.png"
    Image.new("RGB", (50, 40), color="red").save(png)
    assert is_multipage_tiff(png) is False


def test_non_tiff_unchanged(tmp_path: Path) -> None:
    png = tmp_path / "sheet.png"
    Image.new("RGB", (50, 40), color="red").save(png)

    with materialize_sheet_path(png) as work_path:
        assert work_path.resolve() == png.resolve()

    assert png.is_file()


def test_tiff_picks_largest_level(tmp_path: Path) -> None:
    tiff = tmp_path / "sheet.tif"
    _write_pyramid_tiff(tiff, large=(400, 300), small=(100, 100))

    with materialize_sheet_path(tiff) as work_path:
        assert work_path.suffix == ".jpg"
        with Image.open(work_path) as im:
            assert im.size == (400, 300)


def test_tiff_temp_cleaned_up(tmp_path: Path) -> None:
    tiff = tmp_path / "sheet.tiff"
    _write_pyramid_tiff(tiff, large=(200, 150), small=(50, 50))

    temp_path: Path | None = None
    with materialize_sheet_path(tiff) as work_path:
        temp_path = work_path
        assert work_path.is_file()

    assert temp_path is not None
    assert not temp_path.exists()
