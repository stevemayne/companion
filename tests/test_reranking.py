"""Tests for heuristic memory reranking."""
from __future__ import annotations

from uuid import uuid4

from app.schemas import MemoryItem, MemoryKind
from app.services import _rerank_memories

_SESSION = uuid4()


def _mem(
    content: str,
    *,
    score: float = 0.5,
    importance: float = 0.5,
    access_count: int = 0,
) -> MemoryItem:
    return MemoryItem(
        chat_session_id=_SESSION,
        kind=MemoryKind.SEMANTIC,
        content=content,
        score=score,
        importance=importance,
        access_count=access_count,
    )


def test_rerank_boosts_entity_match() -> None:
    """Memory mentioning detected entity outranks higher-base-score memory."""
    items = [
        _mem("User enjoys hiking in the mountains", score=0.8),
        _mem("User has a sister named Sarah", score=0.6),
    ]
    result = _rerank_memories(items, entities=["Sarah"], limit=5)
    assert result[0].content == "User has a sister named Sarah"


def test_rerank_entity_case_insensitive() -> None:
    items = [
        _mem("User mentioned sarah last week", score=0.5),
    ]
    result = _rerank_memories(items, entities=["Sarah"], limit=5)
    # Should match despite case difference
    assert len(result) == 1
    assert result[0].score > 0.5  # boosted


def test_rerank_access_boost() -> None:
    """Frequently accessed memory scores higher than equally scored one."""
    items = [
        _mem("Fact A", score=0.5, access_count=0),
        _mem("Fact B", score=0.5, access_count=10),
    ]
    result = _rerank_memories(items, entities=[], limit=5)
    assert result[0].content == "Fact B"
    assert result[0].score > result[1].score


def test_rerank_access_boost_capped() -> None:
    """Access boost caps at 10 — 100 accesses gives same boost as 10."""
    items = [
        _mem("Fact A", score=0.5, access_count=10),
        _mem("Fact B", score=0.5, access_count=100),
    ]
    result = _rerank_memories(items, entities=[], limit=5)
    assert result[0].score == result[1].score


def test_rerank_respects_limit() -> None:
    items = [_mem(f"Memory {i}", score=0.5) for i in range(10)]
    result = _rerank_memories(items, entities=[], limit=3)
    assert len(result) == 3


def test_rerank_empty_entities() -> None:
    """No entities: ordering follows base_score (with access boost)."""
    items = [
        _mem("Low score", score=0.3),
        _mem("High score", score=0.9),
        _mem("Mid score", score=0.6),
    ]
    result = _rerank_memories(items, entities=[], limit=5)
    assert result[0].content == "High score"
    assert result[1].content == "Mid score"
    assert result[2].content == "Low score"


def test_rerank_empty_items() -> None:
    result = _rerank_memories([], entities=["Sarah"], limit=5)
    assert result == []


def test_rerank_updates_score() -> None:
    """Returned items should have updated score fields."""
    items = [
        _mem("User has a cat named Luna", score=0.5, access_count=5),
    ]
    result = _rerank_memories(items, entities=["Luna"], limit=5)
    # entity_boost=1.5, access_boost=1.0 + 5*0.02 = 1.1
    expected = 0.5 * 1.5 * 1.1
    assert abs(result[0].score - expected) < 0.001
