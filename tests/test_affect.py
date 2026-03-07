"""Unit tests for the companion affect state system."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.agents import _parse_affect_response, _parse_state_response
from app.config import Settings
from app.main import create_app
from app.schemas import CompanionAffect, MonologueState, WorldState
from app.services import (
    _build_affect_block,
    _build_user_context_block,
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
    assert affect.dominance == 0.4
    assert affect.trust == 3.0
    assert affect.closeness == 3.0
    assert affect.engagement == 5.0
    assert affect.recent_triggers == []


# ---------------------------------------------------------------------------
# Affect block rendering
# ---------------------------------------------------------------------------


def test_build_affect_block_contains_mood_and_directives() -> None:
    affect = CompanionAffect(mood="wary", trust=2.0, engagement=7.0, dominance=0.2)
    block = _build_affect_block(affect)
    assert "wary" in block
    # Low trust should produce a guarded directive
    assert "guarded" in block
    # Low dominance should produce tentative directive
    assert "tentative" in block
    # Should NOT contain raw numeric values
    assert "/10" not in block


def test_build_affect_block_no_raw_numbers() -> None:
    affect = CompanionAffect(
        mood="fond", valence=0.5, arousal=0.6, dominance=0.5,
        trust=5.0, closeness=5.0, engagement=5.0,
    )
    block = _build_affect_block(affect)
    # No /10 metrics should appear
    assert "/10" not in block
    # No numeric field labels
    assert "Comfort with user:" not in block
    assert "Shyness:" not in block
    assert "Patience:" not in block


def test_build_affect_block_includes_triggers() -> None:
    affect = CompanionAffect(
        recent_triggers=["user seemed upset"],
    )
    block = _build_affect_block(affect)
    assert "user seemed upset" in block


def test_build_affect_block_high_closeness_high_dominance() -> None:
    affect = CompanionAffect(
        mood="playful", closeness=8.0, dominance=0.8, trust=7.0,
    )
    block = _build_affect_block(affect)
    assert "bold" in block or "uninhibited" in block
    assert "intimate" in block or "affectionate" in block


def test_build_affect_block_low_engagement() -> None:
    affect = CompanionAffect(mood="bored", engagement=2.0)
    block = _build_affect_block(affect)
    assert "drifting" in block or "distracted" in block


# ---------------------------------------------------------------------------
# Affect clamping
# ---------------------------------------------------------------------------


def test_clamp_affect_limits_wild_swings() -> None:
    from app.agents import _clamp_affect
    previous = CompanionAffect(
        valence=0.1, arousal=0.3, dominance=0.4,
        trust=3.0, closeness=3.0, engagement=5.0,
    )
    # LLM tries to jump everything to max
    proposed = CompanionAffect(
        valence=1.0, arousal=1.0, dominance=1.0,
        trust=10.0, closeness=10.0, engagement=10.0,
    )
    result = _clamp_affect(proposed, previous)
    assert result.valence <= 0.1 + 0.3 + 0.001
    assert result.arousal <= 0.3 + 0.25 + 0.001
    assert result.dominance <= 0.4 + 0.15 + 0.001
    assert result.trust <= 3.0 + 0.8 + 0.001
    assert result.closeness <= 3.0 + 0.8 + 0.001
    assert result.engagement <= 5.0 + 1.5 + 0.001


def test_clamp_affect_allows_decreases() -> None:
    from app.agents import _clamp_affect
    previous = CompanionAffect(
        valence=0.5, arousal=0.8, dominance=0.7,
        trust=7.0, closeness=6.0, engagement=8.0,
    )
    proposed = CompanionAffect(
        valence=-1.0, arousal=0.0, dominance=0.0,
        trust=0.0, closeness=0.0, engagement=0.0,
    )
    result = _clamp_affect(proposed, previous)
    assert result.valence >= 0.5 - 0.3 - 0.001
    assert result.arousal >= 0.8 - 0.25 - 0.001
    assert result.dominance >= 0.7 - 0.15 - 0.001
    assert result.trust >= 7.0 - 0.8 - 0.001
    assert result.closeness >= 6.0 - 0.8 - 0.001
    assert result.engagement >= 8.0 - 1.5 - 0.001


# ---------------------------------------------------------------------------
# LLM affect response parsing
# ---------------------------------------------------------------------------


def test_parse_affect_response_valid_json() -> None:
    raw = (
        '{"mood": "amused", "valence": 0.5, "arousal": 0.4, '
        '"dominance": 0.6, "trust": 5.0, "closeness": 4.0, '
        '"engagement": 7.0, '
        '"recent_triggers": ["user told a joke"]}'
    )
    fallback = CompanionAffect()
    result = _parse_affect_response(raw, fallback=fallback)
    assert result.mood == "amused"
    assert result.valence == 0.5
    assert result.engagement == 7.0
    assert result.dominance == 0.6
    assert result.closeness == 4.0


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
# LLM state response parsing (affect + world)
# ---------------------------------------------------------------------------


def test_parse_state_response_valid_json_with_world() -> None:
    import json
    proposed = {
        "mood": "fond", "valence": 0.4, "arousal": 0.3,
        "dominance": 0.5, "trust": 4.0, "closeness": 5.0,
        "engagement": 6.0,
    }
    raw = json.dumps({
        **proposed,
        "recent_triggers": ["warm conversation"],
        "world": {
            "self_state": {"clothing": "blue cardigan"},
            "user_state": {"clothing": "smart suit", "position": "sitting on the couch"},
            "other_characters": {},
            "environment": "living room",
            "time_of_day": "evening",
            "recent_events": [],
        },
        "internal_monologue": "",
    })
    # Use a fallback matching the proposed values so the clamp is a no-op —
    # this test is validating parsing, not clamping.
    affect, world, monologue = _parse_state_response(
        raw, fallback_affect=CompanionAffect(**proposed),
    )
    assert affect.mood == "fond"
    assert affect.trust == 4.0
    assert world.user_state.clothing == "smart suit"
    assert world.user_state.position == "sitting on the couch"
    assert world.self_state.clothing == "blue cardigan"
    assert world.environment == "living room"
    assert monologue == ""


def test_parse_state_response_legacy_user_state_converted() -> None:
    fallback = CompanionAffect(
        mood="fond", valence=0.4, arousal=0.3,
        dominance=0.5, trust=4.0, closeness=5.0, engagement=6.0,
    )
    raw = (
        '{"mood": "fond", "valence": 0.4, "arousal": 0.3, '
        '"dominance": 0.5, "trust": 4.0, "closeness": 5.0, '
        '"engagement": 6.0, '
        '"recent_triggers": ["warm conversation"], '
        '"user_state": ["wearing a smart suit", "sitting on the couch"]}'
    )
    affect, world, monologue = _parse_state_response(
        raw, fallback_affect=fallback,
    )
    assert affect.mood == "fond"
    # Legacy user_state list gets converted to appearance entries
    assert "wearing a smart suit" in world.user_state.appearance
    assert "sitting on the couch" in world.user_state.appearance


def test_parse_state_response_with_monologue() -> None:
    import json
    fallback = CompanionAffect(
        mood="fond", valence=0.4, arousal=0.3,
        dominance=0.5, trust=4.0, closeness=5.0, engagement=6.0,
    )
    raw = json.dumps({
        "mood": "fond", "valence": 0.4, "arousal": 0.3,
        "dominance": 0.5, "trust": 4.0, "closeness": 5.0,
        "engagement": 6.0,
        "recent_triggers": ["warm conversation"],
        "world": {
            "self_state": {}, "user_state": {"position": "sitting on the couch"},
            "other_characters": {}, "recent_events": [],
        },
        "internal_monologue": "I feel really connected right now.",
    })
    _, _, monologue = _parse_state_response(
        raw, fallback_affect=fallback,
    )
    assert monologue == "I feel really connected right now."


def test_parse_state_response_missing_world_uses_fallback() -> None:
    raw = '{"mood": "curious", "valence": 0.1}'
    from app.schemas import CharacterState
    fallback_world = WorldState(
        user_state=CharacterState(clothing="a hat"),
    )
    affect, world, _ = _parse_state_response(
        raw,
        fallback_affect=CompanionAffect(),
        fallback_world=fallback_world,
    )
    assert affect.mood == "curious"
    assert world.user_state.clothing == "a hat"


def test_parse_state_response_invalid_json_uses_fallbacks() -> None:
    raw = "not json"
    fallback_affect = CompanionAffect(mood="wary")
    fallback_world = WorldState(environment="bedroom")
    affect, world, _ = _parse_state_response(
        raw,
        fallback_affect=fallback_affect,
        fallback_world=fallback_world,
    )
    assert affect.mood == "wary"
    assert world.environment == "bedroom"


def test_parse_state_response_with_other_characters() -> None:
    import json
    fallback = CompanionAffect(
        mood="curious", valence=0.2, arousal=0.3,
        dominance=0.5, trust=4.0, closeness=3.0, engagement=6.0,
    )
    raw = json.dumps({
        "mood": "curious", "valence": 0.2, "arousal": 0.3,
        "dominance": 0.5, "trust": 4.0, "closeness": 3.0,
        "engagement": 6.0,
        "recent_triggers": [],
        "world": {
            "self_state": {},
            "user_state": {},
            "other_characters": {
                "Emma": {"clothing": "sundress", "activity": "playing with Rex"},
            },
            "environment": "living room",
            "time_of_day": None,
            "recent_events": ["Emma arrived with Rex"],
        },
        "internal_monologue": "It's nice having company.",
    })
    _, world, _ = _parse_state_response(
        raw, fallback_affect=fallback,
    )
    assert "Emma" in world.other_characters
    assert world.other_characters["Emma"].clothing == "sundress"
    assert world.other_characters["Emma"].activity == "playing with Rex"
    assert "Emma arrived with Rex" in world.recent_events


# ---------------------------------------------------------------------------
# Integration: affect appears in system prompt
# ---------------------------------------------------------------------------


def test_affect_state_appears_in_system_prompt_when_set() -> None:
    app = create_app(Settings(inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    seed_resp = client.post(
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
    companion_id = seed_resp.json()["companion_id"]

    # Pre-populate affect in the monologue store, using the companion_id
    # from the seed so that the ScopedMonologueStore will find it.
    from uuid import UUID

    app.state.container.monologue_store.upsert(
        MonologueState(
            chat_session_id=UUID(session_id),
            companion_id=UUID(companion_id),
            affect=CompanionAffect(mood="wary", trust=2.0, valence=-0.2),
        )
    )

    response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "Hello"},
    )
    assert response.status_code == 200
    content = response.json()["assistant_message"]["content"]
    # The mock provider echoes the full system prompt.  Post-processing
    # strips section headers like "## Your Inner Emotional State" but
    # the directive text survives.  With trust=2.0 the directive says
    # "guarded" which is plain prose and not stripped.
    assert "guarded" in content


# ---------------------------------------------------------------------------
# User context block rendering
# ---------------------------------------------------------------------------


def test_build_user_context_block_contains_items() -> None:
    block = _build_user_context_block(["I put on a smart suit", "I sit down"])
    assert "User's Described State" in block
    assert "I put on a smart suit" in block
    assert "I sit down" in block
    assert "THE USER" in block


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
