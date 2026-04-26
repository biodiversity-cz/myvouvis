from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

from vv_api.config import Settings
from vv_api.dwc_map import map_to_dwc, with_raw
from vv_api.images import ImageValidationError, validate_and_downscale
from vv_api import vv_runner

# 1 = LeafMachine2 label collage (typicky celý arch), 0 = původní snímek (nahrán jen štítek)
RGB_WHOLE_SHEET = 1
RGB_LABEL_ONLY = 0

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def collect_artifact_relpaths(artifact_root: pathlib.Path) -> List[str]:
    if not artifact_root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(artifact_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            out.append(str(p.relative_to(artifact_root)).replace("\\", "/"))
            if len(out) >= 48:
                break
    return out


def transcribe_to_dwc(
    raw_bytes: bytes,
    orig_suffix: str,
    settings: Settings,
    *,
    use_rgb_mode: int = RGB_WHOLE_SHEET,
    run_name: Optional[str] = None,
    include_vv_raw: bool = False,
    artifact_dir: Optional[pathlib.Path] = None,
) -> Tuple[Dict[str, Any], Optional[ImageValidationError], Optional[Exception], Any]:
    """
    Returns (response_dict, image_validation_error, processing_error, last_json).
    If successful, response_dict is ready for JSON (darwin_core, voucher_vision, ...).
    """
    if not orig_suffix:
        orig_suffix = ".jpg"
    try:
        img_bytes, _ctype = validate_and_downscale(raw_bytes, settings)
    except ImageValidationError as e:
        return ({}, e, None, None)

    img_suffix = ".png" if _ctype == "image/png" else ".jpg"
    out: Dict[str, Any] = {}
    err: Optional[Exception] = None
    try:
        out = vv_runner.run_in_temp(
            img_bytes,
            img_suffix,
            settings=settings,
            run_name=run_name,
            use_rgb_mode=use_rgb_mode,
            artifact_dest=artifact_dir,
        )
    except TimeoutError as e:
        err = e
    except Exception as e:  # noqa: BLE001
        err = e

    if err is not None:
        return ({}, None, err, None)

    last_json = out.get("last_JSON_response")
    dwc = map_to_dwc(last_json, settings.dwc_map_path)
    result = {
        "darwin_core": dwc,
        "voucher_vision": {
            "n_failed_OCR": out.get("n_failed_OCR"),
            "n_failed_LLM_calls": out.get("n_failed_LLM_calls"),
            "total_cost": out.get("total_cost"),
        },
    }
    result = with_raw(result, include_vv_raw, last_json)
    return (result, None, None, last_json)
