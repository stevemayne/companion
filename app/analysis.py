from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.inference import EndpointConfig, OpenAICompatibleProvider
from app.schemas import PreprocessResult

ENTITY_STOPWORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whom",
    "whose",
}

ALLOWED_INTENTS = {"question", "status_update", "statement"}
ALLOWED_EMOTIONS = {"anxious", "positive", "neutral"}


class ModelProvider(Protocol):
    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class AnalysisOutcome:
    preprocess: PreprocessResult
    requested_provider: str
    used_provider: str
    fallback_reason: str | None
    latency_ms: float

    def as_trace(self) -> dict[str, str | float | None]:
        return {
            "requested_provider": self.requested_provider,
            "used_provider": self.used_provider,
            "fallback_reason": self.fallback_reason,
            "latency_ms": round(self.latency_ms, 2),
        }


class IntentAnalyzer(Protocol):
    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome: ...


class _LLMAnalysisPayload(BaseModel):
    intent: Literal["question", "status_update", "statement", "other"] = "other"
    emotion: str = "neutral"
    entities: list[str] = Field(default_factory=list)

    @field_validator("emotion")
    @classmethod
    def normalize_emotion(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in ("happy", "excited", "great", "good"):
            return "positive"
        if normalized in ("anxious", "worried", "nervous", "scared"):
            return "anxious"
        if normalized in ("sad", "angry", "negative"):
            return "neutral"
        return normalized if normalized in ALLOWED_EMOTIONS else "neutral"

    @field_validator("entities")
    @classmethod
    def normalize_entities(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for token in value:
            entity = token.strip(",.!?;:()[]{}\"'")
            if not entity:
                continue
            if entity.lower() in ENTITY_STOPWORDS:
                continue
            if entity not in cleaned:
                cleaned.append(entity)
        return cleaned

    def to_preprocess_result(self) -> PreprocessResult:
        intent = self.intent if self.intent in ALLOWED_INTENTS else "statement"
        emotion = self.emotion if self.emotion in ALLOWED_EMOTIONS else "neutral"
        return PreprocessResult(intent=intent, emotion=emotion, entities=self.entities)


class HeuristicIntentAnalyzer:
    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome:
        del chat_session_id
        start = perf_counter()
        lowered = content.lower()
        if any(term in lowered for term in ("nervous", "anxious", "worried", "scared")):
            emotion = "anxious"
        elif any(term in lowered for term in ("happy", "excited", "great", "good")):
            emotion = "positive"
        else:
            emotion = "neutral"

        if "?" in content:
            intent = "question"
        elif any(term in lowered for term in ("i am", "i'm", "i feel", "today")):
            intent = "status_update"
        else:
            intent = "statement"

        entities: list[str] = []
        for token in content.split():
            cleaned = token.strip(",.!?;:()[]{}\"'")
            if not cleaned or len(cleaned) <= 1 or not cleaned[:1].isupper():
                continue
            if cleaned.lower() in ENTITY_STOPWORDS:
                continue
            if cleaned not in entities:
                entities.append(cleaned)

        return AnalysisOutcome(
            preprocess=PreprocessResult(intent=intent, emotion=emotion, entities=entities),
            requested_provider="heuristic",
            used_provider="heuristic",
            fallback_reason=None,
            latency_ms=(perf_counter() - start) * 1000,
        )


class LLMIntentAnalyzer:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        fallback: IntentAnalyzer,
    ) -> None:
        self._provider = provider
        self._fallback = fallback

    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome:
        start = perf_counter()
        try:
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[{"role": "user", "content": self._analysis_prompt(content)}],
            )
            payload = _parse_llm_payload(raw)
            return AnalysisOutcome(
                preprocess=payload.to_preprocess_result(),
                requested_provider="llm",
                used_provider="llm",
                fallback_reason=None,
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback.analyze(chat_session_id=chat_session_id, content=content)
            return AnalysisOutcome(
                preprocess=fallback.preprocess,
                requested_provider="llm",
                used_provider=fallback.used_provider,
                fallback_reason=type(exc).__name__,
                latency_ms=(perf_counter() - start) * 1000,
            )

    def _analysis_prompt(self, content: str) -> str:
        return (
            "You are a preprocessing classifier. "
            "Return strict JSON with keys: intent, emotion, entities.\n"
            "intent must be one of: question, status_update, statement, other.\n"
            "emotion should be a short label (prefer: anxious, positive, neutral).\n"
            "entities should contain named people/places/things and must not include "
            "question words.\n"
            f"message: {content}"
        )


def build_intent_analyzer(settings: Settings) -> IntentAnalyzer:
    provider = settings.analysis_provider.strip().lower()
    heuristic = HeuristicIntentAnalyzer()
    if provider != "llm":
        return heuristic

    llm_provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.analysis_model or settings.inference_model,
            base_url=settings.analysis_base_url or settings.inference_base_url,
            api_key=settings.analysis_api_key or settings.inference_api_key,
        ),
        timeout_seconds=settings.analysis_timeout_seconds,
        max_retries=settings.analysis_max_retries,
    )
    return LLMIntentAnalyzer(provider=llm_provider, fallback=heuristic)


def _parse_llm_payload(raw: str) -> _LLMAnalysisPayload:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON returned by analysis model.") from exc
    return _LLMAnalysisPayload.model_validate(data)
