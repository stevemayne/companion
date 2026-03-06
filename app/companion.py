"""CompanionContext — the self-contained black box for a single companion.

A CompanionContext bundles identity (seed), emotional state (monologue/affect),
semantic memory, and knowledge graph behind a uniform interface.  The
orchestrator receives one of these and operates on it without knowing about
sessions or other companions.

ScopedStore wrappers automatically filter all reads/writes to a single
(session_id, companion_id) partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from app.schemas import (
    CompanionAffect,
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    MemoryStatus,
    MonologueState,
    SessionSeedContext,
)

if TYPE_CHECKING:
    from app.services import GraphStore, MonologueStore, VectorStore


# ---------------------------------------------------------------------------
# Scoped store wrappers
# ---------------------------------------------------------------------------

class ScopedVectorStore:
    """Wraps a VectorStore, pinning all operations to one companion."""

    def __init__(
        self, inner: VectorStore, session_id: UUID, companion_id: UUID,
    ) -> None:
        self._inner = inner
        self._session_id = session_id
        self._companion_id = companion_id

    def upsert_memory(self, item: MemoryItem) -> None:
        scoped = item.model_copy(update={
            "chat_session_id": self._session_id,
            "companion_id": self._companion_id,
        })
        self._inner.upsert_memory(scoped)

    def query_similar(self, *, query: str, limit: int = 10) -> list[MemoryItem]:
        return self._inner.query_similar(
            chat_session_id=self._session_id,
            companion_id=self._companion_id,
            query=query,
            limit=limit,
        )

    def list_memories(self) -> list[MemoryItem]:
        return self._inner.list_memories(
            chat_session_id=self._session_id,
            companion_id=self._companion_id,
        )

    def update_access(self, *, memory_id: UUID) -> None:
        self._inner.update_access(memory_id=memory_id)

    def update_memory(
        self,
        *,
        memory_id: UUID,
        importance: float | None = None,
        status: MemoryStatus | None = None,
    ) -> None:
        self._inner.update_memory(
            memory_id=memory_id, importance=importance, status=status,
        )


class ScopedGraphStore:
    """Wraps a GraphStore, pinning all operations to one companion."""

    def __init__(
        self, inner: GraphStore, session_id: UUID, companion_id: UUID,
    ) -> None:
        self._inner = inner
        self._session_id = session_id
        self._companion_id = companion_id

    def upsert_relation(self, relation: GraphRelation) -> None:
        scoped = relation.model_copy(update={
            "chat_session_id": self._session_id,
            "companion_id": self._companion_id,
        })
        self._inner.upsert_relation(scoped)

    def get_related(self, *, entity: str, limit: int = 10) -> list[GraphRelation]:
        return self._inner.get_related(
            chat_session_id=self._session_id,
            companion_id=self._companion_id,
            entity=entity,
            limit=limit,
        )

    def list_relations(self) -> list[GraphRelation]:
        return self._inner.list_relations(
            chat_session_id=self._session_id,
            companion_id=self._companion_id,
        )


class ScopedMonologueStore:
    """Wraps a MonologueStore, pinning all operations to one companion."""

    def __init__(
        self, inner: MonologueStore, session_id: UUID, companion_id: UUID,
    ) -> None:
        self._inner = inner
        self._session_id = session_id
        self._companion_id = companion_id

    def get(self) -> MonologueState | None:
        return self._inner.get(
            chat_session_id=self._session_id,
            companion_id=self._companion_id,
        )

    def upsert(self, state: MonologueState) -> MonologueState:
        scoped = state.model_copy(update={
            "chat_session_id": self._session_id,
            "companion_id": self._companion_id,
        })
        return self._inner.upsert(scoped)


# ---------------------------------------------------------------------------
# CompanionContext — the black box
# ---------------------------------------------------------------------------

@dataclass
class CompanionContext:
    """Everything the orchestrator needs to run one companion's turn.

    The orchestrator calls methods on ``memories``, ``graph``, and
    ``monologue`` without knowing about session IDs or companion IDs —
    the scoped wrappers handle that.
    """

    companion_id: UUID
    session_id: UUID
    seed: CompanionSeed
    user_description: str | None
    memories: ScopedVectorStore
    graph: ScopedGraphStore
    monologue: ScopedMonologueStore

    @property
    def name(self) -> str:
        return self.seed.companion_name

    @property
    def affect(self) -> CompanionAffect:
        state = self.monologue.get()
        return state.affect if state else CompanionAffect()


def build_companion_context(
    *,
    seed_context: SessionSeedContext,
    vector_store: VectorStore,
    graph_store: GraphStore,
    monologue_store: MonologueStore,
) -> CompanionContext:
    """Construct a CompanionContext from a seed and the shared stores."""
    return CompanionContext(
        companion_id=seed_context.companion_id,
        session_id=seed_context.chat_session_id,
        seed=seed_context.seed,
        user_description=seed_context.user_description,
        memories=ScopedVectorStore(
            vector_store,
            seed_context.chat_session_id,
            seed_context.companion_id,
        ),
        graph=ScopedGraphStore(
            graph_store,
            seed_context.chat_session_id,
            seed_context.companion_id,
        ),
        monologue=ScopedMonologueStore(
            monologue_store,
            seed_context.chat_session_id,
            seed_context.companion_id,
        ),
    )
