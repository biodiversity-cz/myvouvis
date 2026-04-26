from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml


def _load_mapping(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_record(
    last_json: Any,
) -> Optional[dict]:
    if last_json is None:
        return None
    if isinstance(last_json, list) and last_json:
        first = last_json[0]
        return first if isinstance(first, dict) else None
    if isinstance(last_json, dict):
        return last_json
    return None


def map_to_dwc(
    last_json: Any,
    dwc_map_path,
) -> Dict[str, Any]:
    spec = _load_mapping(dwc_map_path)
    field_map: Dict[str, str] = spec.get("field_map") or {}
    required: List[str] = spec.get("required_terms") or []

    rec = _extract_record(last_json)
    if not rec:
        return {
            "dwc": {},
            "validation": {
                "ok": False,
                "missing": required[:],
                "message": "No JSON record in VoucherVision result",
            },
        }

    out: Dict[str, Any] = {}
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
    return {
        "dwc": out,
        "validation": {
            "ok": len(missing) == 0,
            "missing": missing,
        },
    }


def with_raw(
    result: Dict[str, Any], include_raw: bool, raw: Any
) -> Dict[str, Any]:
    if not include_raw:
        return result
    r = deepcopy(result)
    vv = dict(r.get("voucher_vision") or {})
    vv["last_JSON_response"] = raw
    r["voucher_vision"] = vv
    return r
