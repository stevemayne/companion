"""Unit tests for the companion affect state system."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.agents import _parse_affect_response
from app.config import Settings
from app.main import create_app
from app.schemas import CompanionAffect, MonologueState
from app.services import (
    _build_affect_block,
    _build_user_context_block,
    _derive_mood,
    _extract_user_state,
    _heuristic_affect_update,
    _strip_leaked_state,
    _strip_sycophantic_closer,
)

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
    assert affect.shyness == 6.0
    assert affect.patience == 7.0
    assert affect.curiosity == 6.0
    assert affect.vulnerability == 2.0
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
    current = CompanionAffect(trust=3.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="I really appreciate you being here.",
    )
    assert updated.trust > 3.0
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


def test_heuristic_hostile_language_increases_shyness() -> None:
    current = CompanionAffect(shyness=5.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="You are so stupid!",
    )
    assert updated.shyness > 5.0
    assert updated.patience < current.patience


def test_heuristic_warm_language_decreases_shyness() -> None:
    current = CompanionAffect(shyness=6.0, vulnerability=2.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="statement",
        message_content="I really appreciate you being here.",
    )
    assert updated.shyness < 6.0
    assert updated.vulnerability > 2.0


def test_heuristic_question_increases_curiosity() -> None:
    current = CompanionAffect(curiosity=5.0)
    updated = _heuristic_affect_update(
        current=current,
        emotion="neutral",
        intent="question",
        message_content="What do you think about space travel?",
    )
    assert updated.curiosity > 5.0


def test_heuristic_decay_prevents_saturation() -> None:
    """After many neutral turns, values should stay near baseline, not max out."""
    state = CompanionAffect()
    for _ in range(100):
        state = _heuristic_affect_update(
            current=state,
            emotion="neutral",
            intent="statement",
            message_content="Just chatting normally.",
        )
    assert state.trust < 5.0, f"trust saturated to {state.trust}"
    assert state.comfort_level < 5.0, f"comfort saturated to {state.comfort_level}"
    assert state.engagement < 7.0, f"engagement saturated to {state.engagement}"


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
    affect = CompanionAffect(mood="wary", trust=2.0, engagement=7.0, shyness=8.0)
    block = _build_affect_block(affect)
    assert "Current mood: wary" in block
    assert "Trust in user: 2.0/10" in block
    assert "Engagement: 7.0/10" in block
    assert "Shyness: 8.0/10" in block
    assert "Patience:" in block
    assert "Curiosity:" in block
    assert "Vulnerability:" in block


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
    raw += '"engagement": 7.0, "shyness": 4.0, "patience": 8.0, '
    raw += '"curiosity": 7.0, "vulnerability": 3.0, '
    raw += '"recent_triggers": ["user told a joke"]}'
    fallback = CompanionAffect()
    result = _parse_affect_response(raw, fallback=fallback)
    assert result.mood == "amused"
    assert result.valence == 0.5
    assert result.engagement == 7.0
    assert result.shyness == 4.0
    assert result.curiosity == 7.0


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


# ---------------------------------------------------------------------------
# User state extraction
# ---------------------------------------------------------------------------


def test_extract_user_state_action_sentences() -> None:
    text = "I put on a smart suit. The weather is nice today."
    result = _extract_user_state(text)
    assert len(result) == 1
    assert "suit" in result[0]


def test_extract_user_state_let_me_form() -> None:
    text = "Let me put on something more comfortable."
    result = _extract_user_state(text)
    assert len(result) == 1


def test_extract_user_state_multiple_sentences() -> None:
    text = "I sit down on the couch. I adjust my tie. How are you?"
    result = _extract_user_state(text)
    assert len(result) == 2


def test_extract_user_state_ignores_non_physical() -> None:
    text = "I think you're really great. I love this song."
    result = _extract_user_state(text)
    assert len(result) == 0


def test_build_user_context_block_contains_items() -> None:
    block = _build_user_context_block(["I put on a smart suit", "I sit down"])
    assert "User's Described State" in block
    assert "I put on a smart suit" in block
    assert "I sit down" in block
    assert "THE USER" in block


def test_user_state_persisted_across_turns() -> None:
    app = create_app(Settings(
        inference_provider="mock", enable_background_agents=False,
    ))
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

    # First turn: user describes an action
    client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I put on a smart suit."},
    )

    # Second turn: user state should appear in system prompt
    resp = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "How do I look?"},
    )
    assert resp.status_code == 200
    content = resp.json()["assistant_message"]["content"]
    # The actual user state content should be present (the section header
    # gets stripped by _strip_leaked_state, but the item text remains).
    assert "smart suit" in content.lower()


# ---------------------------------------------------------------------------
# Leaked internal state stripping
# ---------------------------------------------------------------------------


def test_strip_leaked_state_bracketed_emotional_state() -> None:
    text = (
        "Those scales look amazing!\n\n"
        "[Emotional state: amazed 5.9/10; trust 4.7/10 (slightly skeptical)]"
    )
    result = _strip_leaked_state(text)
    assert "Emotional state" not in result
    assert "scales look amazing" in result


def test_strip_leaked_state_inline_metrics() -> None:
    text = "I'm curious about your day. trust: 4.7/10 engagement: 8.2/10"
    result = _strip_leaked_state(text)
    assert "4.7/10" not in result
    assert "8.2/10" not in result
    assert "curious about your day" in result


def test_strip_leaked_state_section_header() -> None:
    text = "Hello there!\n## Your Inner Emotional State (internal)\nHow are you?"
    result = _strip_leaked_state(text)
    assert "Inner Emotional State" not in result
    assert "Hello there" in result


def test_strip_leaked_state_detected_intent() -> None:
    text = "That sounds fun! Detected intent: question; emotion: positive"
    result = _strip_leaked_state(text)
    assert "Detected intent" not in result
    assert "sounds fun" in result


def test_strip_leaked_state_preserves_clean_response() -> None:
    text = "I love that idea! Tell me more about how it works."
    result = _strip_leaked_state(text)
    assert result == text


def test_strip_leaked_state_returns_empty_when_entirely_leaked() -> None:
    text = "[Emotional state: happy 8/10]"
    result = _strip_leaked_state(text)
    # If the entire response was leaked state, return empty so the
    # caller can detect and retry rather than echoing the leak.
    assert result == ""


# ---------------------------------------------------------------------------
# Sycophantic closer stripping
# ---------------------------------------------------------------------------


def test_strip_sycophantic_closer_im_here_for_you() -> None:
    text = "That sounds exciting! I'm always here for you."
    result = _strip_sycophantic_closer(text)
    assert "here for you" not in result
    assert "exciting" in result


def test_strip_sycophantic_closer_together_we_can() -> None:
    text = "What a great plan. Together, we can do anything!"
    result = _strip_sycophantic_closer(text)
    assert "Together" not in result
    assert "great plan" in result


def test_strip_sycophantic_closer_preserves_normal_ending() -> None:
    text = "What kind of music do you like?"
    result = _strip_sycophantic_closer(text)
    assert result == text


def test_strip_sycophantic_closer_wont_gut_response() -> None:
    text = "I'm here for you."
    result = _strip_sycophantic_closer(text)
    # Should return original since stripping would leave nothing
    assert result == text
