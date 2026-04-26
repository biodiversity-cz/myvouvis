from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

from vv_api.config import Settings


class ImageValidationError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        self.status = status
        super().__init__(message)


def validate_and_downscale(
    data: bytes,
    settings: Settings,
    max_edge: int | None = None,
) -> Tuple[bytes, str]:
    """
    Enforce file size, decode image, optionally downscale longest side.
    Returns (bytes, content_type) where type is image/jpeg or image/png.
    """
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File too large: {len(data)} bytes (max {settings.max_upload_bytes})",
            status=413,
        )

    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
    except Exception as e:  # noqa: BLE001
        raise ImageValidationError(f"Invalid image: {e}", status=400) from e

    w, h = im.size
    m_edge = max(w, h)
    if m_edge > settings.max_image_edge_px:
        raise ImageValidationError(
            f"Image too large: {w}x{h} (max edge {settings.max_image_edge_px})",
            status=400,
        )

    target = min(max_edge or settings.resize_max_edge_px, settings.resize_max_edge_px)
    if m_edge > target:
        scale = target / float(m_edge)
        new_w, new_h = int(w * scale), int(h * scale)
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    fmt = (im.format or "JPEG").upper()
    out = io.BytesIO()
    if fmt == "PNG" and im.mode in ("RGBA", "P"):
        im.save(out, format="PNG", optimize=True)
        return out.getvalue(), "image/png"
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    elif im.mode == "L":
        im = im.convert("RGB")
    im.save(out, format="JPEG", quality=settings.jpeg_quality, optimize=True)
    return out.getvalue(), "image/jpeg"


def write_temp_image(
    parent: Path,
    data: bytes,
    suffix: str = ".jpg",
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    p = parent / name
    p.write_bytes(data)
    return p
