from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.schemas import (
    GraphRelation,
    MemoryItem,
    Message,
    MonologueState,
    SessionSeedContext,
)


class ModelProvider(Protocol):
    def generate(self, *, chat_session_id: UUID, prompt: str) -> str: ...


class EpisodicStore(Protocol):
    def append_message(self, message: Message) -> None: ...

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]: ...


class VectorStore(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]: ...


class GraphStore(Protocol):
    def upsert_relation(self, relation: GraphRelation) -> None: ...

    def get_related(
        self, *, chat_session_id: UUID, entity: str, limit: int = 10
    ) -> list[GraphRelation]: ...


class SeedContextStore(Protocol):
    def create_seed(self, context: SessionSeedContext) -> SessionSeedContext: ...

    def update_seed(self, context: SessionSeedContext) -> SessionSeedContext: ...

    def get_seed(self, *, chat_session_id: UUID) -> SessionSeedContext | None: ...


class MonologueStore(Protocol):
    def get(self, *, chat_session_id: UUID) -> MonologueState | None: ...

    def upsert(self, state: MonologueState) -> MonologueState: ...


class Orchestrator(Protocol):
    def handle_turn(self, message: Message) -> Message: ...
