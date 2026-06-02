from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pipeline.types import DarwinCoreResult


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
            dwc={},
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

    missing = [k for k in required if not out.get(k)]
    return DarwinCoreResult(
        dwc=out,
        validation={"ok": len(missing) == 0, "missing": missing},
    )
