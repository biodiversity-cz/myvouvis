from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from pipeline.exceptions import PipelineError

_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
_JPEG_QUALITY = 92


def is_tiff_path(path: Path) -> bool:
    return path.suffix.casefold() in _TIFF_SUFFIXES


def is_multipage_tiff(path: Path) -> bool:
    """Return True when the TIFF contains more than one document page.

    Pyramid TIFFs store lower resolutions as sub-IFDs, which tifffile does not
    count in tif.pages — so len(tif.pages) > 1 reliably identifies multi-page
    documents while leaving pyramid TIFFs unaffected.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            return len(tif.pages) > 1
    except (OSError, ValueError, tifffile.TiffFileError):
        return False


def _level_area(shape: tuple[int, ...]) -> int:
    if len(shape) < 2:
        return 0
    return int(shape[-2]) * int(shape[-1])


def _largest_pyramid_array(path: Path) -> np.ndarray:
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.series and tif.series[0].levels:
                level = max(tif.series[0].levels, key=lambda lev: _level_area(lev.shape))
                return level.asarray()
            if tif.pages:
                page = max(tif.pages, key=lambda p: _level_area(p.shape))
                return page.asarray()
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise PipelineError(f"Cannot read TIFF {path.name}: {exc}") from exc
    raise PipelineError(f"No image levels in TIFF: {path}")


def _array_to_rgb_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return Image.fromarray(arr[..., :3]).convert("RGB")
    raise PipelineError("Unsupported TIFF array shape")


def _materialize_tiff_to_jpeg(path: Path) -> Path:
    arr = _largest_pyramid_array(path)
    im = _array_to_rgb_pil(arr)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    im.save(tmp_path, format="JPEG", quality=_JPEG_QUALITY)
    return tmp_path


@contextmanager
def materialize_sheet_path(path: Path) -> Iterator[Path]:
    """Yield a path suitable for LM2/crop; pyramid TIFF → temp JPEG of largest level."""
    resolved = path.resolve()
    if not is_tiff_path(resolved):
        yield resolved
        return

    tmp_path = _materialize_tiff_to_jpeg(resolved)
    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
