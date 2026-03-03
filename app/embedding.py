from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingProvider:
    """Deterministic embedding for tests.

    Uses a bag-of-words hash strategy so that texts sharing tokens
    produce vectors with higher cosine similarity than unrelated texts.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self._dimensions):
                start = (i * 4) % len(digest)
                chunk = digest[start : start + 4]
                if len(chunk) < 4:
                    chunk = (chunk + digest)[:4]
                val = struct.unpack("!I", chunk)[0]
                vec[i] += ((val % 1000) / 1000.0) - 0.5
        # L2-normalize
        magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / magnitude for v in vec]


class OpenAICompatibleEmbeddingProvider:
    """Calls an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._timeout = timeout_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body: dict[str, object] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            body["dimensions"] = self._dimensions

        response = self._client.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings, got {len(data)}"
            )
        # Sort by index to guarantee order matches input
        data.sort(key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in data]
