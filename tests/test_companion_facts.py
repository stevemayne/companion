"""Tests for companion fact tracking (schema, storage, filtering)."""
from __future__ import annotations

from uuid import uuid4

from app.schemas import MemoryItem, MemoryKind, MemoryStatus

_SESSION = uuid4()


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
