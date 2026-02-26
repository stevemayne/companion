from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from app.config import Settings
from app.inference import (
    EndpointConfig,
    FailoverInferenceProvider,
    OpenAICompatibleProvider,
    build_inference_provider,
)


def _success_payload(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}]}


def test_openai_compatible_provider_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=_success_payload("hello"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(model="m", base_url="http://example/v1", api_key=None),
        timeout_seconds=1.0,
        max_retries=1,
        client=client,
    )

    result = provider.generate(chat_session_id=uuid4(), messages=[{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_openai_compatible_provider_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_success_payload("recovered"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(model="m", base_url="http://example/v1", api_key=None),
        timeout_seconds=1.0,
        max_retries=2,
        client=client,
    )

    result = provider.generate(chat_session_id=uuid4(), messages=[{"role": "user", "content": "hi"}])
    assert result == "recovered"
    assert attempts["count"] == 2


def test_failover_provider_uses_secondary() -> None:
    def primary_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    def secondary_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_payload("from-secondary"))

    primary = OpenAICompatibleProvider(
        endpoint=EndpointConfig(model="m1", base_url="http://primary/v1", api_key=None),
        timeout_seconds=1.0,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(primary_handler)),
    )
    secondary = OpenAICompatibleProvider(
        endpoint=EndpointConfig(model="m2", base_url="http://secondary/v1", api_key=None),
        timeout_seconds=1.0,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(secondary_handler)),
    )

    provider = FailoverInferenceProvider(primary=primary, secondary=secondary)
    result = provider.generate(chat_session_id=uuid4(), messages=[{"role": "user", "content": "hi"}])
    assert result == "from-secondary"


def test_build_inference_provider_requires_fallback_configuration() -> None:
    settings = Settings(
        inference_provider="openai_compatible",
        inference_model="m",
        inference_base_url="http://primary/v1",
        inference_failover_enabled=True,
    )

    with pytest.raises(ValueError):
        build_inference_provider(settings)
