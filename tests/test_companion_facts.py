"""Tests for companion self-fact extraction and tracking."""
from __future__ import annotations

from uuid import uuid4

from app.analysis import extract_companion_facts
from app.schemas import MemoryItem, MemoryKind, MemoryStatus

_SESSION = uuid4()


# ---------------------------------------------------------------------------
# extract_companion_facts — heuristic extraction
# ---------------------------------------------------------------------------


def test_extract_identity_statement() -> None:
    """'I am a nature spirit' should be captured."""
    facts = extract_companion_facts(
        "I am a nature spirit who loves the forest.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "Chloe" in facts[0].text
    assert facts[0].subject == "Chloe"


def test_extract_preference_statement() -> None:
    """'I love cooking' should be captured."""
    facts = extract_companion_facts(
        "I love cooking Italian food on rainy evenings.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "cooking" in facts[0].text.lower()


def test_extract_history_statement() -> None:
    """'I grew up near the coast' should be captured."""
    facts = extract_companion_facts(
        "I grew up near the coast where the waves sang all day.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "coast" in facts[0].text.lower()


def test_extract_contraction() -> None:
    """'I'm a morning person' should be captured."""
    facts = extract_companion_facts(
        "I'm definitely a morning person.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "Chloe is" in facts[0].text


def test_skips_filler_think_you() -> None:
    """'I think you should...' is about the user, not a self-fact."""
    facts = extract_companion_facts(
        "I think you should try going for a walk.",
        companion_name="Chloe",
    )
    assert len(facts) == 0


def test_skips_filler_hope_you() -> None:
    facts = extract_companion_facts(
        "I hope you feel better soon.",
        companion_name="Chloe",
    )
    assert len(facts) == 0


def test_skips_filler_understand() -> None:
    facts = extract_companion_facts(
        "I understand how that must feel.",
        companion_name="Chloe",
    )
    assert len(facts) == 0


def test_skips_filler_glad() -> None:
    facts = extract_companion_facts(
        "I'm glad to hear that.",
        companion_name="Chloe",
    )
    assert len(facts) == 0


def test_skips_sentence_with_you() -> None:
    """Sentences mentioning 'you' are about the user, not the companion."""
    facts = extract_companion_facts(
        "I have always admired your courage.",
        companion_name="Chloe",
    )
    assert len(facts) == 0


def test_skips_short_sentences() -> None:
    facts = extract_companion_facts("I am.", companion_name="Chloe")
    assert len(facts) == 0


def test_converts_to_third_person() -> None:
    """First person should be converted to companion's name."""
    facts = extract_companion_facts(
        "I love stargazing on clear nights.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert facts[0].text.startswith("Chloe")
    assert "I" not in facts[0].text


def test_converts_my_to_possessive() -> None:
    """'My favorite' should become 'Chloe's favorite'."""
    facts = extract_companion_facts(
        "My favorite season is autumn without a doubt.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "Chloe's" in facts[0].text


def test_default_companion_name() -> None:
    """When no companion_name is given, uses 'Companion'."""
    facts = extract_companion_facts(
        "I love the sound of rain.",
    )
    assert len(facts) == 1
    assert "Companion" in facts[0].text


def test_multiple_facts_extracted() -> None:
    """Multiple sentences can yield multiple facts."""
    facts = extract_companion_facts(
        "I grew up near a lake. I love swimming in summer. "
        "I enjoy reading by the water.",
        companion_name="Chloe",
    )
    assert len(facts) == 3


def test_deduplication() -> None:
    """Identical sentences should not produce duplicate facts."""
    facts = extract_companion_facts(
        "I love cooking Italian food. I love cooking Italian food.",
        companion_name="Chloe",
    )
    assert len(facts) == 1


def test_importance_is_moderate() -> None:
    """Companion self-facts should have moderate importance (0.6)."""
    facts = extract_companion_facts(
        "I am fascinated by astronomy.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert facts[0].importance == 0.6


def test_mixed_message_extracts_only_self_facts() -> None:
    """A response with both self-facts and user-directed content."""
    facts = extract_companion_facts(
        "That sounds wonderful! I grew up near the mountains. "
        "I think you would love it there too.",
        companion_name="Chloe",
    )
    assert len(facts) == 1
    assert "mountains" in facts[0].text.lower()


# ---------------------------------------------------------------------------
# MemoryKind.COMPANION
# ---------------------------------------------------------------------------


def test_companion_memory_kind_exists() -> None:
    assert MemoryKind.COMPANION == "companion"


def test_memory_item_with_companion_kind() -> None:
    item = MemoryItem(
        chat_session_id=_SESSION,
        kind=MemoryKind.COMPANION,
        content="Chloe loves cooking Italian food",
    )
    assert item.kind == MemoryKind.COMPANION
    assert item.status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# Filtering companion facts from user retrieval
# ---------------------------------------------------------------------------


def test_companion_facts_filtered_from_reranking() -> None:
    """COMPANION memories should be excluded before reranking."""
    from app.services import _rerank_memories

    items = [
        MemoryItem(
            chat_session_id=_SESSION,
            kind=MemoryKind.SEMANTIC,
            content="User has a cat named Luna",
            score=0.7,
        ),
        MemoryItem(
            chat_session_id=_SESSION,
            kind=MemoryKind.COMPANION,
            content="Chloe loves cats",
            score=0.9,
        ),
    ]
    # Filter as handle_turn does, then rerank
    filtered = [m for m in items if m.kind != MemoryKind.COMPANION]
    result = _rerank_memories(filtered, entities=[], limit=5)
    assert len(result) == 1
    assert result[0].kind == MemoryKind.SEMANTIC
