"""Tests for Phase 2: consolidation loop."""
from __future__ import annotations

import json
from uuid import uuid4

from app.consolidation import (
    ConsolidationAgent,
    _heuristic_consolidate,
    _parse_consolidation_response,
    _token_overlap,
)
from app.embedding import MockEmbeddingProvider
from app.schemas import MemoryItem, MemoryKind, MemoryStatus, Message
from app.services import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Token overlap utility
# ---------------------------------------------------------------------------


def test_token_overlap_identical() -> None:
    assert _token_overlap("the cat sat", "the cat sat") == 1.0


def test_token_overlap_partial() -> None:
    overlap = _token_overlap("the cat sat", "the cat ran away")
    # "the" and "cat" overlap = 2/3
    assert abs(overlap - 2 / 3) < 0.01


def test_token_overlap_none() -> None:
    assert _token_overlap("hello world", "foo bar") == 0.0


def test_token_overlap_empty() -> None:
    assert _token_overlap("", "something") == 0.0


# ---------------------------------------------------------------------------
# Heuristic consolidation
# ---------------------------------------------------------------------------


def test_heuristic_reinforces_overlapping_memories() -> None:
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="I love my cat Luna"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.6,
        ),
    ]
    result = _heuristic_consolidate(messages=messages, existing_memories=existing)
    assert len(result.reinforced) == 1
    assert result.reinforced[0].old_importance == 0.6
    assert result.reinforced[0].new_importance == 0.7


def test_heuristic_skips_non_overlapping_memories() -> None:
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="I went to the store"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.6,
        ),
    ]
    result = _heuristic_consolidate(messages=messages, existing_memories=existing)
    assert len(result.reinforced) == 0


def test_heuristic_skips_archived_memories() -> None:
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="I love my cat Luna"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.6,
            status=MemoryStatus.ARCHIVED,
        ),
    ]
    result = _heuristic_consolidate(messages=messages, existing_memories=existing)
    assert len(result.reinforced) == 0


def test_heuristic_caps_importance_at_1() -> None:
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="User loves cat Luna"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.95,
        ),
    ]
    result = _heuristic_consolidate(messages=messages, existing_memories=existing)
    assert len(result.reinforced) == 1
    assert result.reinforced[0].new_importance == 1.0


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def test_parse_consolidation_response_reinforce() -> None:
    session_id = uuid4()
    mem_id = uuid4()
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            memory_id=mem_id,
            kind=MemoryKind.SEMANTIC,
            content="User has a cat",
            importance=0.5,
        ),
    ]
    raw = json.dumps({
        "reinforce": [{"memory_id": str(mem_id), "new_importance": 0.7}],
    })
    result = _parse_consolidation_response(raw, existing_memories=existing)
    assert len(result.reinforced) == 1
    assert result.reinforced[0].memory_id == mem_id
    assert result.reinforced[0].new_importance == 0.7


def test_parse_consolidation_response_supersede() -> None:
    session_id = uuid4()
    mem_id = uuid4()
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            memory_id=mem_id,
            kind=MemoryKind.SEMANTIC,
            content="User has a cat named Luna",
            importance=0.6,
        ),
    ]
    raw = json.dumps({
        "supersede": [{
            "memory_id": str(mem_id),
            "reason": "Cat passed away",
            "replacement_text": "User had a cat named Luna who passed away",
        }],
    })
    result = _parse_consolidation_response(raw, existing_memories=existing)
    assert len(result.superseded) == 1
    assert result.superseded[0].memory_id == mem_id
    assert result.superseded[0].replacement_text == "User had a cat named Luna who passed away"


def test_parse_consolidation_response_new_facts() -> None:
    raw = json.dumps({
        "new_facts": [
            {"text": "User recently adopted a dog", "importance": 0.7},
        ],
    })
    result = _parse_consolidation_response(raw, existing_memories=[])
    assert len(result.new_facts) == 1
    assert result.new_facts[0].text == "User recently adopted a dog"
    assert result.new_facts[0].importance == 0.7


def test_parse_consolidation_response_clamps_importance() -> None:
    session_id = uuid4()
    mem_id = uuid4()
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            memory_id=mem_id,
            kind=MemoryKind.SEMANTIC,
            content="test",
            importance=0.5,
        ),
    ]
    raw = json.dumps({
        "reinforce": [{"memory_id": str(mem_id), "new_importance": 1.5}],
        "new_facts": [{"text": "fact", "importance": -0.3}],
    })
    result = _parse_consolidation_response(raw, existing_memories=existing)
    assert result.reinforced[0].new_importance == 1.0
    assert result.new_facts[0].importance == 0.0


def test_parse_consolidation_response_ignores_unknown_memory_ids() -> None:
    raw = json.dumps({
        "reinforce": [{"memory_id": str(uuid4()), "new_importance": 0.8}],
        "supersede": [{"memory_id": str(uuid4()), "reason": "gone"}],
    })
    result = _parse_consolidation_response(raw, existing_memories=[])
    assert len(result.reinforced) == 0
    assert len(result.superseded) == 0


def test_parse_consolidation_response_fenced_json() -> None:
    raw = "```json\n{\"new_facts\": [{\"text\": \"test fact\", \"importance\": 0.5}]}\n```"
    result = _parse_consolidation_response(raw, existing_memories=[])
    assert len(result.new_facts) == 1


# ---------------------------------------------------------------------------
# ConsolidationAgent integration
# ---------------------------------------------------------------------------


def test_consolidation_agent_heuristic_no_provider() -> None:
    """Without an LLM provider, agent falls back to heuristic."""
    agent = ConsolidationAgent(provider=None)
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="I love my cat Luna"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.5,
        ),
    ]
    result = agent.consolidate_session(
        chat_session_id=session_id,
        messages=messages,
        existing_memories=existing,
    )
    assert result.provider == "heuristic"
    assert len(result.reinforced) == 1


def test_consolidation_agent_empty_messages() -> None:
    agent = ConsolidationAgent(provider=None)
    result = agent.consolidate_session(
        chat_session_id=uuid4(),
        messages=[],
        existing_memories=[],
    )
    assert result.reinforced == []
    assert result.superseded == []
    assert result.new_facts == []


def test_consolidation_repeated_mentions_increase_importance() -> None:
    """Repeated conversation about the same topic should reinforce the memory."""
    session_id = uuid4()
    agent = ConsolidationAgent(provider=None)
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User has sister named Sarah",
            importance=0.6,
        ),
    ]
    # User message re-states the same fact with high token overlap
    messages = [
        Message(
            chat_session_id=session_id, role="user",
            content="User has a sister named Sarah who is visiting",
        ),
    ]
    r1 = agent.consolidate_session(
        chat_session_id=session_id, messages=messages,
        existing_memories=existing,
    )
    assert len(r1.reinforced) == 1
    assert r1.reinforced[0].new_importance > existing[0].importance


# ---------------------------------------------------------------------------
# Consolidation idempotency
# ---------------------------------------------------------------------------


def test_consolidation_is_idempotent() -> None:
    """Running heuristic consolidation twice on the same data yields same result."""
    session_id = uuid4()
    messages = [
        Message(chat_session_id=session_id, role="user", content="I love my cat Luna"),
    ]
    existing = [
        MemoryItem(
            chat_session_id=session_id,
            kind=MemoryKind.SEMANTIC,
            content="User loves cat Luna",
            importance=0.6,
        ),
    ]
    r1 = _heuristic_consolidate(messages=messages, existing_memories=existing)
    r2 = _heuristic_consolidate(messages=messages, existing_memories=existing)
    assert len(r1.reinforced) == len(r2.reinforced)
    for a, b in zip(r1.reinforced, r2.reinforced, strict=True):
        assert a.new_importance == b.new_importance


# ---------------------------------------------------------------------------
# update_memory integration with InMemoryVectorStore
# ---------------------------------------------------------------------------


def test_update_memory_changes_importance() -> None:
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()
    memory_id = uuid4()

    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        memory_id=memory_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a cat",
        importance=0.5,
    ))

    store.update_memory(memory_id=memory_id, importance=0.8)
    items = store.list_memories(chat_session_id=session_id)
    assert items[0].importance == 0.8


def test_update_memory_changes_status() -> None:
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()
    memory_id = uuid4()

    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        memory_id=memory_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a cat",
        importance=0.5,
    ))

    store.update_memory(memory_id=memory_id, status=MemoryStatus.SUPERSEDED)
    items = store.list_memories(chat_session_id=session_id)
    assert items[0].status == MemoryStatus.SUPERSEDED


def test_superseded_memories_excluded_from_query() -> None:
    """After superseding, the memory should not appear in query results."""
    embedder = MockEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(embedder=embedder)
    session_id = uuid4()
    memory_id = uuid4()

    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        memory_id=memory_id,
        kind=MemoryKind.SEMANTIC,
        content="User has a cat named Luna",
        importance=0.5,
    ))
    store.upsert_memory(MemoryItem(
        chat_session_id=session_id,
        kind=MemoryKind.SEMANTIC,
        content="User had a cat named Luna who passed away",
        importance=0.6,
    ))

    store.update_memory(memory_id=memory_id, status=MemoryStatus.SUPERSEDED)

    results = store.query_similar(
        chat_session_id=session_id,
        query="cat Luna",
        limit=10,
    )
    for r in results:
        assert r.memory_id != memory_id
        assert r.status == MemoryStatus.ACTIVE
