from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "aether_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "aether_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)


@dataclass
class InMemoryRateLimiter:
    limit_per_minute: int
    _requests: dict[str, deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def allow(self, key: str) -> bool:
        now = time()
        window_start = now - 60
        with self._lock:
            bucket = self._requests.setdefault(key, deque())
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit_per_minute:
                return False
            bucket.append(now)
            return True


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
