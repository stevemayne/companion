from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any
from uuid import UUID, uuid4

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}", re.IGNORECASE),
]


def sanitize_debug_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


@dataclass
class DebugTraceStore:
    enabled: bool
    limit_per_session: int = 100

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._store: dict[UUID, deque[dict[str, Any]]] = {}

    def add_trace(self, *, chat_session_id: UUID, trace: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            bucket = self._store.setdefault(chat_session_id, deque(maxlen=self.limit_per_session))
            bucket.append(trace)

    def list_traces(self, *, chat_session_id: UUID) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._lock:
            bucket = self._store.get(chat_session_id, deque())
            return list(bucket)


def build_trace_base(*, chat_session_id: UUID) -> dict[str, Any]:
    return {
        "trace_id": str(uuid4()),
        "timestamp": time(),
        "chat_session_id": str(chat_session_id),
    }
