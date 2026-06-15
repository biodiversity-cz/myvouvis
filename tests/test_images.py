from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from pipeline.images import materialize_sheet_path


def _write_pyramid_tiff(path: Path, large: tuple[int, int], small: tuple[int, int]) -> None:
    big = np.zeros((large[1], large[0], 3), dtype=np.uint8)
    thumb = np.zeros((small[1], small[0], 3), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tif:
        tif.write(big, photometric="rgb", subifds=1)
        tif.write(thumb, photometric="rgb")


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
