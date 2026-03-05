from __future__ import annotations

from uuid import uuid4

import pytest

from app.analysis import (
    ExtractedFact,
    LLMFactExtractor,
    _NoOpFactExtractor,
    _parse_extraction_payload,
    build_fact_extractor,
    validate_facts,
)
from app.config import Settings


class _JsonProvider:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, *, chat_session_id: object, messages: list[dict[str, str]]) -> str:
        del chat_session_id, messages
        return self.payload


# ---------------------------------------------------------------------------
# LLMFactExtractor
# ---------------------------------------------------------------------------


def test_llm_extractor_parses_structured_json() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '[{"subject": "User", "predicate": "loves", "object": "pizza", '
            '"text": "User loves pizza"}, '
            '{"subject": "User", "predicate": "has", "object": "a cat named Luna", '
            '"text": "User has a cat named Luna"}]'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love pizza and my cat Luna is the best.",
        assistant_message="Pizza and cats, great combo!",
    )
    assert outcome.used_provider == "llm"
    assert outcome.fallback_reason is None
    assert len(outcome.facts) == 2
    assert outcome.facts[0].subject == "User"
    assert outcome.facts[0].predicate == "loves"
    assert outcome.facts[0].object == "pizza"
    assert outcome.facts[0].text == "User loves pizza"


def test_llm_extractor_handles_fenced_json() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '```json\n[{"subject": "User", "predicate": "is", '
            '"object": "a teacher", "text": "User is a teacher"}]\n```'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I am a teacher.",
        assistant_message="That's a noble profession.",
    )
    assert len(outcome.facts) == 1
    assert outcome.facts[0].text == "User is a teacher"
    assert outcome.used_provider == "llm"


def test_llm_extractor_returns_empty_on_empty_array() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider("[]"),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="Hello!",
        assistant_message="Hi there!",
    )
    assert outcome.facts == []
    assert outcome.used_provider == "llm"


def test_llm_extractor_returns_empty_on_invalid_json() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider("not valid json at all"),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I'm worried about my exam tomorrow.",
        assistant_message="You'll do great!",
    )
    assert outcome.requested_provider == "llm"
    assert outcome.used_provider == "llm"
    assert outcome.fallback_reason is not None
    assert outcome.facts == []


def test_llm_extractor_returns_empty_on_provider_exception() -> None:
    class _FailingProvider:
        def generate(self, *, chat_session_id: object, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("Connection refused")

    extractor = LLMFactExtractor(
        provider=_FailingProvider(),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love hiking in the mountains.",
        assistant_message="That sounds wonderful!",
    )
    assert outcome.used_provider == "llm"
    assert outcome.fallback_reason == "RuntimeError"
    assert outcome.facts == []
    assert outcome.entities == []


def test_llm_extractor_handles_legacy_string_format() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider('["User loves pizza", "User has a cat"]'),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love pizza and I have a cat.",
        assistant_message="Nice!",
    )
    assert outcome.used_provider == "llm"
    assert len(outcome.facts) == 2
    assert outcome.facts[0].text == "User loves pizza"
    assert outcome.facts[0].subject == "User"
    assert outcome.entities == []


def test_llm_extractor_routes_companion_subject() -> None:
    """Companion-subject facts go to companion_facts, not facts."""
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '[{"subject": "Ari", "predicate": "comforted", "object": "User", '
            '"text": "Ari comforted User"}, '
            '{"subject": "User", "predicate": "felt better after talking to", "object": "Ari", '
            '"text": "User felt better after talking to Ari"}]'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="You really helped me feel better.",
        assistant_message="I'm glad!",
        companion_name="Ari",
    )
    assert len(outcome.facts) == 1
    assert outcome.facts[0].subject == "User"
    assert outcome.facts[0].text == "User felt better after talking to Ari"
    assert len(outcome.companion_facts) == 1
    assert outcome.companion_facts[0].subject == "Ari"
    assert outcome.companion_facts[0].text == "Ari comforted User"


def test_llm_extractor_preserves_subject_object_direction() -> None:
    """Regression: 'Sarah yelled at me' should not become 'User yelled at Sarah'."""
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '[{"subject": "User", "predicate": "was yelled at by", "object": "Sarah", '
            '"text": "User was yelled at by Sarah"}]'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="Sarah yelled at me yesterday.",
        assistant_message="That sounds really stressful.",
    )
    assert outcome.facts[0].subject == "User"
    assert outcome.facts[0].predicate == "was yelled at by"
    assert outcome.facts[0].object == "Sarah"


def test_llm_extractor_returns_entities() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '{"facts": [{"subject": "User", "predicate": "has sister", "object": "Sarah", '
            '"text": "User has a sister named Sarah"}], '
            '"entities": [{"name": "Sarah", "relationship": "sister", "aliases": ["sis"]}]}'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="My sister Sarah, we call her sis, is visiting.",
        assistant_message="How exciting!",
    )
    assert outcome.used_provider == "llm"
    assert len(outcome.facts) == 1
    assert len(outcome.entities) == 1
    assert outcome.entities[0].name == "Sarah"
    assert outcome.entities[0].relationship == "sister"
    assert outcome.entities[0].aliases == ["sis"]


def test_llm_extractor_wrapper_with_empty_entities() -> None:
    extractor = LLMFactExtractor(
        provider=_JsonProvider(
            '{"facts": [{"subject": "User", "predicate": "loves", "object": "pizza", '
            '"text": "User loves pizza"}], "entities": []}'
        ),
    )
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love pizza.",
        assistant_message="Yum!",
    )
    assert len(outcome.facts) == 1
    assert outcome.entities == []


# ---------------------------------------------------------------------------
# _parse_extraction_payload
# ---------------------------------------------------------------------------


def test_parse_extraction_wrapper_format() -> None:
    facts, entities = _parse_extraction_payload(
        '{"facts": [{"subject": "User", "predicate": "argued with", "object": "Sarah", '
        '"text": "User argued with Sarah"}], '
        '"entities": [{"name": "Sarah", "relationship": "friend", "aliases": ["S"]}]}'
    )
    assert len(facts) == 1
    assert facts[0].text == "User argued with Sarah"
    assert len(entities) == 1
    assert entities[0].name == "Sarah"
    assert entities[0].relationship == "friend"
    assert entities[0].aliases == ["S"]


def test_parse_extraction_backward_compat_array() -> None:
    facts, entities = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "is", "object": "tall", "text": "User is tall"}]'
    )
    assert len(facts) == 1
    assert facts[0].text == "User is tall"
    assert entities == []


def test_parse_extraction_legacy_string_array() -> None:
    facts, entities = _parse_extraction_payload('["User loves pizza", "User has a cat"]')
    assert len(facts) == 2
    assert facts[0].text == "User loves pizza"
    assert entities == []


def test_parse_extraction_filters_non_objects_in_facts() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "is", "object": "tall", "text": "User is tall"}, '
        '42, null, '
        '{"subject": "User", "predicate": "likes", "object": "tea", "text": "User likes tea"}]'
    )
    assert len(facts) == 2
    assert facts[0].text == "User is tall"
    assert facts[1].text == "User likes tea"


def test_parse_extraction_deduplicates_facts() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "likes", "object": "tea", "text": "User likes tea"}, '
        '{"subject": "User", "predicate": "enjoys", "object": "tea", "text": "User likes tea"}]'
    )
    assert len(facts) == 1


def test_parse_extraction_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        _parse_extraction_payload("not json at all")


def test_parse_extraction_synthesizes_text_from_triple() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "likes", "object": "tea"}]'
    )
    assert len(facts) == 1
    assert facts[0].text == "User likes tea"


def test_parse_extraction_skips_empty_entity_names() -> None:
    _, entities = _parse_extraction_payload(
        '{"facts": [], "entities": [{"name": "", "relationship": "friend", "aliases": []}, '
        '{"name": "Sarah", "relationship": "sister", "aliases": []}]}'
    )
    assert len(entities) == 1
    assert entities[0].name == "Sarah"


def test_parse_extraction_deduplicates_entities_by_name() -> None:
    _, entities = _parse_extraction_payload(
        '{"facts": [], "entities": ['
        '{"name": "Sarah", "relationship": "sister", "aliases": ["sis"]}, '
        '{"name": "Sarah", "relationship": "friend", "aliases": []}]}'
    )
    assert len(entities) == 1


def test_parse_extraction_fenced_wrapper() -> None:
    facts, entities = _parse_extraction_payload(
        '```json\n{"facts": [{"subject": "User", "predicate": "likes", '
        '"object": "tea", "text": "User likes tea"}], "entities": []}\n```'
    )
    assert len(facts) == 1
    assert entities == []


def test_parse_extraction_relation_type_defaults_to_relates_to() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "likes", "object": "tea", "text": "User likes tea"}]'
    )
    assert facts[0].relation_type == "RELATES_TO"


def test_parse_extraction_relation_type_parsed() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "likes", "object": "tea", '
        '"text": "User likes tea", "relation_type": "LIKES"}]'
    )
    assert facts[0].relation_type == "LIKES"


def test_parse_extraction_unknown_relation_type_falls_back() -> None:
    facts, _ = _parse_extraction_payload(
        '[{"subject": "User", "predicate": "likes", "object": "tea", '
        '"text": "User likes tea", "relation_type": "INVENTED_TYPE"}]'
    )
    assert facts[0].relation_type == "RELATES_TO"


# ---------------------------------------------------------------------------
# validate_facts
# ---------------------------------------------------------------------------


def test_validate_facts_filters_companion_as_subject() -> None:
    facts = [
        ExtractedFact(subject="Ari", predicate="said", object="hello", text="Ari said hello"),
        ExtractedFact(subject="User", predicate="greeted", object="Ari", text="User greeted Ari"),
    ]
    valid = validate_facts(facts, companion_name="Ari")
    assert len(valid) == 1
    assert valid[0].subject == "User"


def test_validate_facts_filters_empty_subject() -> None:
    facts = [
        ExtractedFact(subject="", predicate="likes", object="tea", text="likes tea"),
        ExtractedFact(subject="User", predicate="likes", object="tea", text="User likes tea"),
    ]
    valid = validate_facts(facts)
    assert len(valid) == 1
    assert valid[0].subject == "User"


def test_validate_facts_filters_empty_text() -> None:
    facts = [
        ExtractedFact(subject="User", predicate="likes", object="tea", text=""),
        ExtractedFact(subject="User", predicate="likes", object="tea", text="User likes tea"),
    ]
    valid = validate_facts(facts)
    assert len(valid) == 1


def test_validate_facts_deduplicates_by_text() -> None:
    facts = [
        ExtractedFact(subject="User", predicate="likes", object="tea", text="User likes tea"),
        ExtractedFact(subject="User", predicate="enjoys", object="tea", text="User likes tea"),
    ]
    valid = validate_facts(facts)
    assert len(valid) == 1


def test_validate_facts_companion_case_insensitive() -> None:
    facts = [
        ExtractedFact(subject="ari", predicate="said", object="hi", text="ari said hi"),
        ExtractedFact(subject="User", predicate="likes", object="Ari", text="User likes Ari"),
    ]
    valid = validate_facts(facts, companion_name="Ari")
    assert len(valid) == 1
    assert valid[0].subject == "User"


# ---------------------------------------------------------------------------
# build_fact_extractor factory
# ---------------------------------------------------------------------------


def test_build_fact_extractor_returns_noop_by_default() -> None:
    settings = Settings(inference_provider="mock", analysis_provider="heuristic")
    extractor = build_fact_extractor(settings)
    assert isinstance(extractor, _NoOpFactExtractor)


def test_build_fact_extractor_returns_llm_when_configured() -> None:
    settings = Settings(
        inference_provider="mock",
        analysis_provider="llm",
        analysis_model="test-model",
        analysis_base_url="http://localhost:1234/v1",
    )
    extractor = build_fact_extractor(settings)
    assert isinstance(extractor, LLMFactExtractor)
