"""Unit tests for the companion affect state system."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.agents import _parse_affect_response
from app.config import Settings
from app.main import create_app
from app.schemas import CompanionAffect, MonologueState
from app.services import _build_affect_block, _derive_mood, _heuristic_affect_update


# ---------------------------------------------------------------------------
# CompanionAffect defaults
# ---------------------------------------------------------------------------


def test_companion_affect_defaults() -> None:
    affect = CompanionAffect()
    assert affect.mood == "curious"
    assert affect.valence == 0.1
    assert affect.arousal == 0.3
    assert affect.comfort_level == 3.0
    assert affect.trust == 3.0
    assert affect.attraction == 3.0
    assert affect.engagement == 5.0
    assert affect.recent_triggers == []


# ---------------------------------------------------------------------------
# Heuristic affect updater
# ---------------------------------------------------------------------------


def test_heuristic_hostile_language_lowers_trust() -> None:
    current = CompanionAffect(trust=5.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="You are so stupid and useless!",
    )
    assert updated.trust < 5.0
    assert updated.attraction < current.attraction
    assert any("hostile" in t for t in updated.recent_triggers)


def test_heuristic_warm_language_raises_trust() -> None:
    current = CompanionAffect(trust=5.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="I really appreciate you being here.",
    )
    assert updated.trust > 5.0
    assert any("warm" in t for t in updated.recent_triggers)


def test_heuristic_positive_emotion_raises_engagement() -> None:
    current = CompanionAffect(engagement=5.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="positive",
        intent="statement",
        message_content="I'm so happy today!",
    )
    assert updated.engagement > 5.0


def test_heuristic_anxious_emotion_raises_arousal() -> None:
    current = CompanionAffect(arousal=0.3)
    updated = _heuristic_affect_update(
        current=current,
        emotion="anxious",
        intent="statement",
        message_content="I'm nervous about tomorrow.",
    )
    assert updated.arousal > 0.3


def test_heuristic_normal_interaction_slowly_builds_comfort() -> None:
    current = CompanionAffect(comfort_level=3.0, trust=3.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="Today I went to the store.",
    )
    assert updated.comfort_level > 3.0
    assert updated.trust > 3.0


# ---------------------------------------------------------------------------
# Mood derivation
# ---------------------------------------------------------------------------


def test_derive_mood_high_valence_high_arousal_is_excited() -> None:
    assert _derive_mood(valence=0.5, arousal=0.7, triggers=[]) == "excited"


def test_derive_mood_high_valence_low_arousal_is_fond() -> None:
    assert _derive_mood(valence=0.5, arousal=0.3, triggers=[]) == "fond"


def test_derive_mood_negative_valence_high_arousal_is_anxious() -> None:
    assert _derive_mood(valence=-0.5, arousal=0.7, triggers=[]) == "anxious"


def test_derive_mood_hostile_trigger_returns_hurt_or_withdrawn() -> None:
    result = _derive_mood(
        valence=-0.2, arousal=0.5, triggers=["hostile language detected"],
    )
    assert result in ("hurt", "withdrawn")


def test_derive_mood_default_is_curious() -> None:
    assert _derive_mood(valence=0.05, arousal=0.3, triggers=[]) == "curious"


# ---------------------------------------------------------------------------
# Affect block rendering
# ---------------------------------------------------------------------------


def test_build_affect_block_contains_key_fields() -> None:
    affect = CompanionAffect(mood="wary", trust=2.0, engagement=7.0)
    block = _build_affect_block(affect)
    assert "Current mood: wary" in block
    assert "Trust in user: 2.0/10" in block
    assert "Engagement: 7.0/10" in block


def test_build_affect_block_includes_triggers() -> None:
    affect = CompanionAffect(
        recent_triggers=["user seemed upset"],
    )
    block = _build_affect_block(affect)
    assert "user seemed upset" in block


# ---------------------------------------------------------------------------
# LLM affect response parsing
# ---------------------------------------------------------------------------


def test_parse_affect_response_valid_json() -> None:
    raw = '{"mood": "amused", "valence": 0.5, "arousal": 0.4, '
    raw += '"comfort_level": 6.0, "trust": 5.0, "attraction": 4.0, '
    raw += '"engagement": 7.0, "recent_triggers": ["user told a joke"]}'
    fallback = CompanionAffect()
    result = _parse_affect_response(raw, fallback=fallback)
    assert result.mood == "amused"
    assert result.valence == 0.5
    assert result.engagement == 7.0


def test_parse_affect_response_fenced_json() -> None:
    raw = 'Here is the update:\n```json\n{"mood": "fond", "valence": 0.3}\n```'
    fallback = CompanionAffect()
    result = _parse_affect_response(raw, fallback=fallback)
    assert result.mood == "fond"


def test_parse_affect_response_invalid_returns_fallback() -> None:
    raw = "This is not valid JSON at all."
    fallback = CompanionAffect(mood="wary")
    result = _parse_affect_response(raw, fallback=fallback)
    assert result.mood == "wary"


# ---------------------------------------------------------------------------
# Integration: affect appears in system prompt
# ---------------------------------------------------------------------------


def test_affect_state_appears_in_system_prompt_when_set() -> None:
    app = create_app(Settings(inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(
        f"/v1/sessions/{session_id}/seed",
        json={
            "seed": {
                "companion_name": "Ari",
                "backstory": "Ari is reflective.",
                "character_traits": ["curious"],
                "goals": ["build trust"],
                "relationship_setup": "Companion.",
            },
        },
    )

    # Pre-populate affect in the monologue store
    from uuid import UUID

    app.state.container.monologue_store.upsert(
        MonologueState(
            chat_session_id=UUID(session_id),
            affect=CompanionAffect(mood="wary", trust=2.0, valence=-0.2),
        )
    )

    response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "Hello"},
    )
    assert response.status_code == 200
    content = response.json()["assistant_message"]["content"]
    assert "Current mood: wary" in content
    assert "Trust in user: 2.0/10" in content
