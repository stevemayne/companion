from __future__ import annotations

from uuid import uuid4

from app.analysis import (
    ExtractionOutcome,
    HeuristicFactExtractor,
    LLMFactExtractor,
    _parse_facts_payload,
    build_fact_extractor,
)
from app.config import Settings


class _JsonProvider:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, *, chat_session_id: object, messages: list[dict[str, str]]) -> str:
        del chat_session_id, messages
        return self.payload


# ---------------------------------------------------------------------------
# HeuristicFactExtractor
# ---------------------------------------------------------------------------


def test_heuristic_extracts_first_person_statements() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I met Sarah at dinner. She was really nice.",
        assistant_message="That sounds lovely!",
    )
    assert len(outcome.facts) >= 1
    assert any("Sarah" in f for f in outcome.facts)
    assert outcome.used_provider == "heuristic"


def test_heuristic_skips_questions() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="How are you doing today?",
        assistant_message="I'm good!",
    )
    assert outcome.facts == []


def test_heuristic_rewrites_to_third_person() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I'm interested in magic.",
        assistant_message="That's cool!",
    )
    assert len(outcome.facts) == 1
    assert outcome.facts[0].startswith("User")
    assert "I'm" not in outcome.facts[0]


def test_heuristic_skips_short_fragments() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="Hi. Ok. Sure.",
        assistant_message="Hello!",
    )
    assert outcome.facts == []


def test_heuristic_deduplicates_facts() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love cats. I really love cats.",
        assistant_message="Cats are great!",
    )
    assert len(outcome.facts) == len(set(outcome.facts))


# ---------------------------------------------------------------------------
# LLMFactExtractor
# ---------------------------------------------------------------------------


def test_llm_extractor_parses_valid_json_array() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider('["User loves pizza", "User has a cat named Luna"]'),
        fallback=HeuristicFactExtractor(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love pizza and my cat Luna is the best.",
        assistant_message="Pizza and cats, great combo!",
    )
    assert outcome.used_provider == "llm"
    assert outcome.fallback_reason is None
    assert outcome.facts == ["User loves pizza", "User has a cat named Luna"]


def test_llm_extractor_handles_fenced_json() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider('```json\n["User is a teacher"]\n```'),
        fallback=HeuristicFactExtractor(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I am a teacher.",
        assistant_message="That's a noble profession.",
    )
    assert outcome.facts == ["User is a teacher"]
    assert outcome.used_provider == "llm"


def test_llm_extractor_returns_empty_on_empty_array() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider("[]"),
        fallback=HeuristicFactExtractor(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="Hello!",
        assistant_message="Hi there!",
    )
    assert outcome.facts == []
    assert outcome.used_provider == "llm"


def test_llm_extractor_falls_back_on_invalid_json() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider("not valid json at all"),
        fallback=HeuristicFactExtractor(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I'm worried about my exam tomorrow.",
        assistant_message="You'll do great!",
    )
    assert outcome.requested_provider == "llm"
    assert outcome.used_provider == "heuristic"
    assert outcome.fallback_reason is not None


def test_llm_extractor_falls_back_on_provider_exception() -> None:
    class _FailingProvider:
        def generate(self, *, chat_session_id: object, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("Connection refused")

    extractor = LLMFactExtractor(
        provider=_FailingProvider(),
        fallback=HeuristicFactExtractor(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love hiking in the mountains.",
        assistant_message="That sounds wonderful!",
    )
    assert outcome.used_provider == "heuristic"
    assert outcome.fallback_reason == "RuntimeError"


# ---------------------------------------------------------------------------
# _parse_facts_payload
# ---------------------------------------------------------------------------


def test_parse_facts_filters_non_strings() -> None:
    result = _parse_facts_payload('[" User is tall ", 42, null, "User likes tea"]')
    assert result == ["User is tall", "User likes tea"]


def test_parse_facts_deduplicates() -> None:
    result = _parse_facts_payload('["User likes tea", "User likes tea", "User is tall"]')
    assert result == ["User likes tea", "User is tall"]


def test_parse_facts_rejects_non_array() -> None:
    import pytest

    with pytest.raises(ValueError, match="JSON array"):
        _parse_facts_payload('{"fact": "oops"}')


# ---------------------------------------------------------------------------
# build_fact_extractor factory
# ---------------------------------------------------------------------------


def test_build_fact_extractor_returns_heuristic_by_default() -> None:
    settings = Settings(inference_provider="mock", analysis_provider="heuristic")
    extractor = build_fact_extractor(settings)
    assert isinstance(extractor, HeuristicFactExtractor)


def test_build_fact_extractor_returns_llm_when_configured() -> None:
    settings = Settings(
        inference_provider="mock",
        analysis_provider="llm",
        analysis_model="test-model",
        analysis_base_url="http://localhost:1234/v1",
    )
    extractor = build_fact_extractor(settings)
    assert isinstance(extractor, LLMFactExtractor)
