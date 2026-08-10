from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from pipeline.types import DarwinCoreResult

# DwC date terms we normalize after mapping (invalid LLM placeholders like 2003-00-00).
_DWC_DATE_TERMS = frozenset({"dateIdentified", "eventDate"})

_ISO_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def normalize_iso_date(value: str) -> str:
    """Convert LLM date placeholders to valid ISO 8601 partial dates."""
    v = value.strip()
    if not v:
        return v
    m = _ISO_DATE_RE.fullmatch(v)
    if not m:
        return v
    year, month, day = m.group(1), m.group(2), m.group(3)
    if month == "00" or day == "00":
        if month in (None, "00") or month == "00":
            return year
        return f"{year}-{month}"
    if day is None:
        return year if month is None else f"{year}-{month}"
    return f"{year}-{month}-{day}"


def _load_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_record(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        first = raw[0]
        return first if isinstance(first, dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def map_to_dwc(raw: Any, dwc_map_path: Path) -> DarwinCoreResult:
    spec = _load_spec(dwc_map_path)
    field_map: dict[str, str] = spec.get("field_map") or {}
    required: list[str] = spec.get("required_terms") or []

    rec = _extract_record(raw)
    if not rec:
        return DarwinCoreResult(
            dwc={"labelType": "unknown"},
            validation={
                "ok": False,
                "missing": required[:],
                "message": "No JSON record from LLM",
            },
        )

    out: dict[str, Any] = {}
    for src, dst in field_map.items():
        if not dst or src not in rec:
            continue
        v = rec.get(src)
        if v in (None, ""):
            continue
        if dst in out and out[dst] not in (None, ""):
            if isinstance(out[dst], str) and isinstance(v, str):
                out[dst] = f"{out[dst]}; {v}"
            else:
                out[dst] = v
        else:
            out[dst] = v

    for term in _DWC_DATE_TERMS:
        if term in out and isinstance(out[term], str):
            out[term] = normalize_iso_date(out[term])

    missing = [k for k in required if not out.get(k)]
    # Always emit labelType so clients can rely on the key (and handwritten flag).
    if not out.get("labelType"):
        out["labelType"] = "unknown"
    return DarwinCoreResult(
        dwc=out,
        validation={"ok": len(missing) == 0, "missing": missing},
    )
