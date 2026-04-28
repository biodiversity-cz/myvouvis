"""Stejný model jako biodiversity-cz/myhepsi: jen JPEG, PNG, TIFF, JP2."""

from __future__ import annotations

from typing import Optional

ALLOWED_UPLOAD_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/jp2",
        "image/jpeg2000",
    }
)

ALLOWED_UPLOAD_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".jp2",
    }
)


def upload_mimetype_ok(content_type: Optional[str]) -> bool:
    """
    True = akceptovat. Prázdný nebo jen application/octet-stream nechť projdou
    (někdy posílané pro JP2); finální validace je přes PIL při dekódování.
    """
    if not content_type or not str(content_type).strip():
        return True
    main = content_type.split(";", maxsplit=1)[0].strip().lower()
    if main == "application/octet-stream":
        return True
    return main in ALLOWED_UPLOAD_MIME_TYPES
