from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    base_url: str
    api_key: str | None


class MockInferenceProvider:
    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str:
        del chat_session_id
        parts = [f"[{m['role']}] {m['content']}" for m in messages]
        return "[mock-response]\n" + "\n".join(parts)


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        endpoint: EndpointConfig,
        timeout_seconds: float,
        max_retries: int,
        temperature: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str:
        del chat_session_id
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_api(messages)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Primary inference call failed (attempt=%s/%s): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise InferenceError("Primary inference endpoint failed.") from last_error

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._endpoint.api_key:
            headers["Authorization"] = f"Bearer {self._endpoint.api_key}"

        body: dict[str, object] = {
            "model": self._endpoint.model,
            "messages": messages,
            "temperature": self._temperature,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
        }
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens

        response = self._client.post(
            f"{self._endpoint.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_content(payload)


class FailoverInferenceProvider:
    def __init__(
        self,
        *,
        primary: OpenAICompatibleProvider,
        secondary: OpenAICompatibleProvider,
    ) -> None:
        self._primary = primary
        self._secondary = secondary

    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str:
        try:
            return self._primary.generate(chat_session_id=chat_session_id, messages=messages)
        except InferenceError:
            logger.warning("Primary inference failed, switching to failover endpoint.")
            return self._secondary.generate(chat_session_id=chat_session_id, messages=messages)


def build_inference_provider(settings: Settings) -> Any:
    provider = settings.inference_provider.strip().lower()
    if provider == "mock":
        return MockInferenceProvider()

    primary = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.inference_model,
            base_url=settings.inference_base_url,
            api_key=settings.inference_api_key,
        ),
        timeout_seconds=settings.inference_timeout_seconds,
        max_retries=settings.inference_max_retries,
        temperature=settings.inference_temperature,
        frequency_penalty=settings.inference_frequency_penalty,
        presence_penalty=settings.inference_presence_penalty,
        max_tokens=settings.inference_max_tokens,
    )

    if not settings.inference_failover_enabled:
        return primary

    if not settings.fallback_inference_model or not settings.fallback_inference_base_url:
        raise ValueError(
            "inference_failover_enabled=true requires FALLBACK_INFERENCE_MODEL and "
            "FALLBACK_INFERENCE_BASE_URL"
        )

    secondary = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.fallback_inference_model,
            base_url=settings.fallback_inference_base_url,
            api_key=settings.fallback_inference_api_key,
        ),
        timeout_seconds=settings.inference_timeout_seconds,
        max_retries=settings.inference_max_retries,
        temperature=settings.inference_temperature,
        frequency_penalty=settings.inference_frequency_penalty,
        presence_penalty=settings.inference_presence_penalty,
        max_tokens=settings.inference_max_tokens,
    )
    return FailoverInferenceProvider(primary=primary, secondary=secondary)


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InferenceError("Inference response missing choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise InferenceError("Inference response has invalid choice format.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise InferenceError("Inference response missing message object.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InferenceError("Inference response missing message content.")
    return content
