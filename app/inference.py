from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

LOGS_DIR: Path | None = Path("logs/inference")


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    base_url: str
    api_key: str | None


def _log_to_session_file(chat_session_id: UUID, record: dict[str, Any]) -> None:
    """Append a JSON-lines record to a per-session log file."""
    if LOGS_DIR is None:
        return
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = LOGS_DIR / f"{chat_session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write inference log for session %s", chat_session_id)


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
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_api(messages, chat_session_id=chat_session_id)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Primary inference call failed (attempt=%s/%s): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise InferenceError("Primary inference endpoint failed.") from last_error

    def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        chat_session_id: UUID | None = None,
    ) -> str:
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

        start = perf_counter()
        response = self._client.post(
            f"{self._endpoint.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=self._timeout_seconds,
        )
        duration_ms = (perf_counter() - start) * 1000
        response.raise_for_status()
        payload = response.json()

        content = _extract_content(payload)
        finish_reason = _extract_finish_reason(payload)
        usage = payload.get("usage")

        logger.info(
            "inference complete model=%s finish_reason=%s "
            "prompt_tokens=%s completion_tokens=%s duration_ms=%.0f",
            self._endpoint.model,
            finish_reason,
            usage.get("prompt_tokens") if usage else None,
            usage.get("completion_tokens") if usage else None,
            duration_ms,
        )

        if finish_reason == "length":
            logger.warning(
                "Response truncated (finish_reason=length). "
                "prompt_tokens=%s completion_tokens=%s max_tokens=%s",
                usage.get("prompt_tokens") if usage else "?",
                usage.get("completion_tokens") if usage else "?",
                self._max_tokens,
            )

        if chat_session_id is not None:
            _log_to_session_file(chat_session_id, {
                "model": self._endpoint.model,
                "max_tokens": self._max_tokens,
                "message_count": len(messages),
                "finish_reason": finish_reason,
                "usage": usage,
                "duration_ms": round(duration_ms, 1),
                "request_messages": messages,
                "response_content": content,
            })

        return content


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
    return content.strip()


def _extract_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0].get("finish_reason")
    return None
