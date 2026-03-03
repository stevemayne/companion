"""Tests for Phase 1: importance scoring, weighted retrieval, and access tracking."""
from __future__ import annotations

import math
from uuid import uuid4

from app.analysis import HeuristicFactExtractor, _heuristic_importance, _parse_facts_list
from app.embedding import MockEmbeddingProvider
from app.schemas import MemoryItem, MemoryKind, MemoryStatus
from app.services import InMemoryVectorStore, _deduplicate_memories

# ---------------------------------------------------------------------------
# Heuristic importance scoring
# ---------------------------------------------------------------------------


def test_high_importance_for_relationship_keywords() -> None:
    assert _heuristic_importance("I love hiking in the mountains") == 0.8
    assert _heuristic_importance("My sister is visiting next week") == 0.8


def test_medium_importance_for_preference_keywords() -> None:
    assert _heuristic_importance("I enjoy cooking Italian food") == 0.6
    assert _heuristic_importance("I work at a startup downtown") == 0.6


def test_low_importance_for_transient_statements() -> None:
    assert _heuristic_importance("I went to the store") == 0.4


# ---------------------------------------------------------------------------
# LLM extraction payload parsing with importance
# ---------------------------------------------------------------------------


def test_parse_facts_list_extracts_importance() -> None:
    raw = [
        {
            "subject": "User",
            "predicate": "has sister",
            "object": "Sarah",
            "text": "User has a sister named Sarah",
            "importance": 0.8,
        },
        {
            "subject": "User",
            "predicate": "went to",
            "object": "store",
            "text": "User went to the store",
            "importance": 0.3,
        },
    ]
    facts = _parse_facts_list(raw)
    assert len(facts) == 2
    assert facts[0].importance == 0.8
    assert facts[1].importance == 0.3


def test_parse_facts_list_defaults_importance_when_missing() -> None:
    raw = [
        {"subject": "User", "predicate": "likes", "object": "cats", "text": "User likes cats"},
    ]
    facts = _parse_facts_list(raw)
    assert facts[0].importance == 0.5


def test_parse_facts_list_clamps_importance() -> None:
    raw = [
        {"subject": "User", "predicate": "x", "object": "y", "text": "a", "importance": 1.5},
        {"subject": "User", "predicate": "x", "object": "y", "text": "b", "importance": -0.3},
    ]
    facts = _parse_facts_list(raw)
    assert facts[0].importance == 1.0
    assert facts[1].importance == 0.0


# ---------------------------------------------------------------------------
# Heuristic fact extractor carries importance through
# ---------------------------------------------------------------------------


def test_heuristic_extractor_assigns_importance() -> None:
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message="I love my sister Sarah and I went to the park.",
        assistant_message="That sounds nice!",
    )
    assert len(outcome.facts) >= 1
    # "I love my sister Sarah" should get high importance
    love_facts = [
        f for f in outcome.facts
        if "love" in f.text.lower() or "sister" in f.text.lower()
    ]
    assert love_facts
    assert love_facts[0].importance == 0.8


# ---------------------------------------------------------------------------
# MemoryItem schema
# ---------------------------------------------------------------------------


def test_memory_item_defaults() -> None:
    item = MemoryItem(
        chat_session_id=uuid4(),
        kind=MemoryKind.SEMANTIC,
        content="User likes cats",
    )
    assert item.importance == 0.5
    assert item.access_count == 0
    assert item.last_accessed is None
    assert item.status == MemoryStatus.ACTIVE
    assert item.source_turn_id is None


def test_memory_item_with_importance() -> None:
    item = MemoryItem(
        chat_session_id=uuid4(),
        kind=MemoryKind.SEMANTIC,
        content="User's sister is Sarah",
        importance=0.8,
    )
    assert item.importance == 0.8


# ---------------------------------------------------------------------------
# Importance-weighted retrieval (InMemoryVectorStore)
# ---------------------------------------------------------------------------


def test_weighted_retrieval_prefers_important_memories() -> None:
    """When two memories have similar cosine similarity to a query,
    the one with higher importance should rank higher because
    final_score = cosine_sim * importance * recency_factor."""
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()

    # Same content, only importance differs — cosine sim is identical
    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a pet cat named Luna",
        importance=0.2,
    ))
    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a pet cat named Luna",
        importance=0.9,
    ))

    results = store.query_similar(
        chat_session_id=session_id,
        query="User has a pet cat named Luna",
        limit=2,
    )
    assert len(results) == 2
    # Both have identical cosine sim (same content), so importance decides
    assert results[0].importance == 0.9
    assert results[1].importance == 0.2


def test_archived_memories_excluded_from_retrieval() -> None:
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()

    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User used to like dogs",
        status=MemoryStatus.ARCHIVED,
    ))
    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User likes cats now",
        status=MemoryStatus.ACTIVE,
    ))

    results = store.query_similar(
        chat_session_id=session_id,
        query="pets animals",
        limit=10,
    )
    assert all(r.status == MemoryStatus.ACTIVE for r in results)


# ---------------------------------------------------------------------------
# Access tracking (InMemoryVectorStore)
# ---------------------------------------------------------------------------


def test_update_access_increments_count() -> None:
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()
    memory_id = uuid4()

    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        memory_id=memory_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a pet cat",
    ))

    store.update_access(memory_id=memory_id)
    store.update_access(memory_id=memory_id)

    items = store.list_memories(chat_session_id=session_id)
    assert len(items) == 1
    assert items[0].access_count == 2
    assert items[0].last_accessed is not None


# ---------------------------------------------------------------------------
# Deduplication prefers higher importance
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_higher_importance() -> None:
    session_id = uuid4()
    low = MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User loves cats a lot",
        importance=0.3,
    )
    high = MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User loves cats a lot indeed",
        importance=0.9,
    )
    # The two overlap heavily — only the higher-importance one should survive
    result = _deduplicate_memories([low, high], history_text="hello world")
    assert len(result) == 1
    assert result[0].importance == 0.9


# ---------------------------------------------------------------------------
# Recency decay math
# ---------------------------------------------------------------------------


def test_recency_decay_reduces_old_memory_score() -> None:
    # exp(-0.01 * 70) ≈ 0.497 — after 70 days, score is halved
    decay_70 = math.exp(-0.01 * 70)
    assert 0.49 < decay_70 < 0.51

    # Fresh memory has no decay
    decay_0 = math.exp(-0.01 * 0)
    assert decay_0 == 1.0
