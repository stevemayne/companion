"""Tests for Phase 4: adaptive retrieval."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.retrieval import (
    HeuristicRetrievalDecider,
    LLMRetrievalDecider,
    RetrievalDecision,
    _parse_retrieval_response,
)

# ---------------------------------------------------------------------------
# Heuristic decider
# ---------------------------------------------------------------------------

_SESSION = uuid4()


def _heuristic(
    message: str, *, intent: str = "statement", emotion: str = "neutral",
) -> RetrievalDecision:
    return HeuristicRetrievalDecider().decide(
        chat_session_id=_SESSION, message=message,
        intent=intent, emotion=emotion,
    )


@pytest.mark.parametrize("greeting", [
    "hi", "hello", "hey", "yo", "sup",
    "Hi!", "Hello.", "Hey!",
    "thanks", "thank you", "thx",
    "good morning", "good night",
    "bye", "goodbye", "see you",
    "ok", "sure", "yeah", "nah",
    "lol", "haha", "wow", "nice", "cool",
    "yes", "no", "nope",
    "hmm", "ah", "oh",
])
def test_heuristic_skips_greetings(greeting: str) -> None:
    decision = _heuristic(greeting)
    assert decision.should_retrieve is False


@pytest.mark.parametrize("short", [
    "not bad",
    "go on",
    "me too",
])
def test_heuristic_skips_short_filler(short: str) -> None:
    decision = _heuristic(short)
    assert decision.should_retrieve is False


@pytest.mark.parametrize("substantive", [
    "My sister Sarah is visiting next week",
    "I just got a new job at the startup downtown",
    "Do you remember what I told you about my cat?",
    "I've been feeling anxious about the move",
])
def test_heuristic_retrieves_for_substantive_messages(substantive: str) -> None:
    decision = _heuristic(substantive)
    assert decision.should_retrieve is True


def test_heuristic_retrieves_for_short_with_proper_noun() -> None:
    decision = _heuristic("Ask Sarah")
    assert decision.should_retrieve is True


def test_heuristic_retrieves_for_questions() -> None:
    decision = _heuristic("Do you remember my cat?")
    assert decision.should_retrieve is True


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def test_parse_retrieval_response_retrieve_true() -> None:
    raw = json.dumps({"retrieve": True, "query": "user's cat Luna", "reason": "asking about pet"})
    decision = _parse_retrieval_response(raw)
    assert decision.should_retrieve is True
    assert decision.rewritten_query == "user's cat Luna"
    assert decision.reason == "asking about pet"


def test_parse_retrieval_response_retrieve_false() -> None:
    raw = json.dumps({"retrieve": False, "query": "", "reason": "just a greeting"})
    decision = _parse_retrieval_response(raw)
    assert decision.should_retrieve is False
    assert decision.rewritten_query is None


def test_parse_retrieval_response_fenced_json() -> None:
    raw = '```json\n{"retrieve": true, "query": "sister Sarah", "reason": "people"}\n```'
    decision = _parse_retrieval_response(raw)
    assert decision.should_retrieve is True
    assert decision.rewritten_query == "sister Sarah"


def test_parse_retrieval_response_defaults_to_retrieve() -> None:
    raw = json.dumps({"reason": "unclear"})
    decision = _parse_retrieval_response(raw)
    assert decision.should_retrieve is True


def test_parse_retrieval_response_raises_on_garbage() -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_retrieval_response("not json at all")


# ---------------------------------------------------------------------------
# LLM decider with mock provider
# ---------------------------------------------------------------------------


class _MockProvider:
    def __init__(self, response: str | None = None, *, raise_error: bool = False) -> None:
        self._response = response
        self._raise_error = raise_error

    def generate(self, *, chat_session_id: uuid4, messages: list[dict[str, str]]) -> str:
        if self._raise_error:
            raise RuntimeError("LLM timeout")
        assert self._response is not None
        return self._response


def test_llm_decider_parses_response() -> None:
    provider = _MockProvider(json.dumps({
        "retrieve": True,
        "query": "user's sister Sarah",
        "reason": "asking about family",
    }))
    decider = LLMRetrievalDecider(
        provider=provider, fallback=HeuristicRetrievalDecider(),
    )
    decision = decider.decide(
        chat_session_id=_SESSION, message="Tell me about Sarah",
        intent="question", emotion="neutral",
    )
    assert decision.should_retrieve is True
    assert decision.rewritten_query == "user's sister Sarah"


def test_llm_decider_falls_back_on_error() -> None:
    provider = _MockProvider(raise_error=True)
    decider = LLMRetrievalDecider(
        provider=provider, fallback=HeuristicRetrievalDecider(),
    )
    # "hi" should be skipped by heuristic fallback
    decision = decider.decide(
        chat_session_id=_SESSION, message="hi",
        intent="statement", emotion="neutral",
    )
    assert decision.should_retrieve is False


def test_llm_decider_falls_back_on_bad_json() -> None:
    provider = _MockProvider("this is not json")
    decider = LLMRetrievalDecider(
        provider=provider, fallback=HeuristicRetrievalDecider(),
    )
    # Substantive message — heuristic fallback should retrieve
    decision = decider.decide(
        chat_session_id=_SESSION,
        message="My sister Sarah is coming to visit next week",
        intent="statement", emotion="neutral",
    )
    assert decision.should_retrieve is True
