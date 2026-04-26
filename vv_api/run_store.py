from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunPayload:
    darwin_core: Dict[str, Any]
    voucher_vision: Dict[str, Any]
    last_json: Optional[Any] = None
    artifact_relpaths: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    created: float = field(default_factory=time.time)


def _store_ttl() -> int:
    from vv_api.config import get_settings

    return int(get_settings().run_result_ttl_s)


class RunStore:
    """In-memory result store (single-replica; use Redis for multi-replica)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, RunPayload] = {}

    def set(self, run_id: str, payload: RunPayload) -> None:
        with self._lock:
            self._purge_nolock()
            self._data[run_id] = payload

    def get(self, run_id: str) -> Optional[RunPayload]:
        with self._lock:
            self._purge_nolock()
            p = self._data.get(run_id)
            if p is None:
                return None
            if time.time() - p.created > _store_ttl():
                del self._data[run_id]
                return None
            return p

    def _purge_nolock(self) -> None:
        now = time.time()
        t = _store_ttl()
        dead = [k for k, v in self._data.items() if now - v.created > t]
        for k in dead:
            del self._data[k]


STORE = RunStore()
