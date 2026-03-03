"""Adaptive retrieval — decide whether to consult long-term memory for a turn.

HeuristicRetrievalDecider: fast keyword check, no LLM call.
LLMRetrievalDecider: lightweight LLM call with heuristic fallback.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

# Short messages that never need memory retrieval.
_SKIP_PHRASES: set[str] = {
    "hi", "hey", "hello", "yo", "sup",
    "ok", "okay", "sure", "yep", "yup", "yeah", "nah",
    "thanks", "thank you", "thx", "ty",
    "good morning", "good night", "good evening", "gm", "gn",
    "bye", "goodbye", "see you", "later", "cya",
    "lol", "haha", "heh", "lmao",
    "wow", "nice", "cool", "great", "awesome",
    "yes", "no", "nope",
    "hmm", "hm", "ah", "oh",
}

_HAS_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}")


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool
    rewritten_query: str | None = None
    reason: str = ""
    latency_ms: float = 0.0


class RetrievalDecider(Protocol):
    def decide(
        self, *, chat_session_id: UUID, message: str,
        intent: str, emotion: str,
    ) -> RetrievalDecision: ...


class _LLMProvider(Protocol):
    def generate(
        self, *, chat_session_id: UUID, messages: list[dict[str, str]],
    ) -> str: ...


# ── Heuristic ─────────────────────────────────────────────────────────


class HeuristicRetrievalDecider:
    def decide(
        self, *, chat_session_id: UUID, message: str,
        intent: str, emotion: str,
    ) -> RetrievalDecision:
        del chat_session_id  # unused in heuristic path
        start = perf_counter()
        normalized = message.strip().lower().rstrip("!?.,:;")

        # Exact match against known skip phrases
        if normalized in _SKIP_PHRASES:
            return RetrievalDecision(
                should_retrieve=False,
                reason="greeting/filler",
                latency_ms=(perf_counter() - start) * 1000,
            )

        words = message.split()
        # Very short messages with no proper nouns — likely filler
        if len(words) < 4 and not _HAS_PROPER_NOUN.search(message):
            return RetrievalDecision(
                should_retrieve=False,
                reason="short message without proper nouns",
                latency_ms=(perf_counter() - start) * 1000,
            )

        return RetrievalDecision(
            should_retrieve=True,
            reason="substantive message",
            latency_ms=(perf_counter() - start) * 1000,
        )


# ── LLM ───────────────────────────────────────────────────────────────

_RETRIEVAL_PROMPT = """\
You are a retrieval decision system. Given a user message and detected \
intent, decide whether to search the user's long-term memory.

User message: {message}
Detected intent: {intent}
Detected emotion: {emotion}

Should we consult long-term memory? If yes, provide an optimized search \
query that captures the key concepts.

Return strict JSON:
{{"retrieve": true/false, "query": "optimized search query or empty string", \
"reason": "brief explanation"}}

Guidelines:
- Greetings, filler, and acknowledgements do NOT need memory.
- Questions about past events, people, preferences DO need memory.
- Statements sharing personal info DO need memory (to check for existing facts).
- When in doubt, retrieve.

Return only JSON, no explanation.\
"""


class LLMRetrievalDecider:
    def __init__(
        self,
        *,
        provider: _LLMProvider,
        fallback: RetrievalDecider,
    ) -> None:
        self._provider = provider
        self._fallback = fallback

    def decide(
        self, *, chat_session_id: UUID, message: str,
        intent: str, emotion: str,
    ) -> RetrievalDecision:
        start = perf_counter()
        try:
            prompt = _RETRIEVAL_PROMPT.format(
                message=message, intent=intent, emotion=emotion,
            )
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[{"role": "user", "content": prompt}],
            )
            decision = _parse_retrieval_response(raw)
            return RetrievalDecision(
                should_retrieve=decision.should_retrieve,
                rewritten_query=decision.rewritten_query,
                reason=decision.reason,
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception:
            logger.debug(
                "LLM retrieval decision failed, falling back to heuristic",
                exc_info=True,
            )
            return self._fallback.decide(
                chat_session_id=chat_session_id, message=message,
                intent=intent, emotion=emotion,
            )


def _parse_retrieval_response(raw: str) -> RetrievalDecision:
    """Parse LLM JSON into a RetrievalDecision."""
    candidate = raw.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    data = json.loads(candidate)
    should_retrieve = bool(data.get("retrieve", True))
    query = str(data.get("query", "")).strip() or None
    reason = str(data.get("reason", ""))
    return RetrievalDecision(
        should_retrieve=should_retrieve,
        rewritten_query=query if should_retrieve else None,
        reason=reason,
    )
