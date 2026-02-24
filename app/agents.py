from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from uuid import UUID

from app.schemas import GraphRelation, MemoryItem, MemoryKind, Message, MonologueState


@dataclass
class AgentMetrics:
    extraction_jobs: int = 0
    reflector_jobs: int = 0
    failures: int = 0


class EpisodicStore(Protocol):
    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]: ...


class VectorStore(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...


class GraphStore(Protocol):
    def upsert_relation(self, relation: GraphRelation) -> None: ...


class MonologueStore(Protocol):
    def get(self, *, chat_session_id: UUID) -> MonologueState | None: ...

    def upsert(self, state: MonologueState) -> MonologueState: ...


class BackgroundAgentDispatcher:
    def __init__(
        self,
        *,
        episodic_store: EpisodicStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        monologue_store: MonologueStore,
        enabled: bool = True,
    ) -> None:
        self._episodic_store = episodic_store
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._monologue_store = monologue_store
        self._enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aether-agents")
        self._lock = Lock()
        self._metrics: dict[UUID, AgentMetrics] = {}
        self._futures: set[Future[None]] = set()

    def enqueue_turn(
        self, *, chat_session_id: UUID, user_message: str, assistant_message: str
    ) -> None:
        if not self._enabled:
            return
        self._submit(
            self._run_extraction,
            chat_session_id,
            user_message,
            assistant_message,
        )
        self._submit(self._run_reflector, chat_session_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def get_metrics(self, *, chat_session_id: UUID) -> AgentMetrics:
        with self._lock:
            return self._metrics.get(chat_session_id, AgentMetrics())

    def _submit(self, fn: Callable[..., None], *args: object) -> None:
        future: Future[None] = self._executor.submit(fn, *args)
        with self._lock:
            self._futures.add(future)

        def _cleanup(done: Future[None]) -> None:
            with self._lock:
                self._futures.discard(done)

        future.add_done_callback(_cleanup)

    def _run_extraction(
        self, chat_session_id: UUID, user_message: str, assistant_message: str
    ) -> None:
        try:
            self._vector_store.upsert_memory(
                MemoryItem(
                    chat_session_id=chat_session_id,
                    kind=MemoryKind.SEMANTIC,
                    content=assistant_message,
                    score=0.8,
                )
            )
            entities = self._extract_entities(f"{user_message} {assistant_message}")
            for entity in entities:
                self._graph_store.upsert_relation(
                    GraphRelation(
                        chat_session_id=chat_session_id,
                        source="assistant",
                        relation="ACKNOWLEDGED_IN_REPLY",
                        target=entity,
                    )
                )
            self._increment(chat_session_id=chat_session_id, extraction_jobs=1)
        except Exception:
            self._increment(chat_session_id=chat_session_id, failures=1)

    def _run_reflector(self, chat_session_id: UUID) -> None:
        try:
            recent = self._episodic_store.get_recent_messages(
                chat_session_id=chat_session_id,
                limit=6,
            )
            summary = " | ".join(f"{msg.role}:{msg.content}" for msg in recent[-3:]) or "none"
            current = self._monologue_store.get(chat_session_id=chat_session_id)
            prefix = current.internal_monologue if current is not None else ""
            combined = f"{prefix} | reflector_summary={summary}".strip(" |")
            self._monologue_store.upsert(
                MonologueState(
                    chat_session_id=chat_session_id,
                    internal_monologue=combined,
                )
            )
            self._increment(chat_session_id=chat_session_id, reflector_jobs=1)
        except Exception:
            self._increment(chat_session_id=chat_session_id, failures=1)

    def _increment(
        self,
        *,
        chat_session_id: UUID,
        extraction_jobs: int = 0,
        reflector_jobs: int = 0,
        failures: int = 0,
    ) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(chat_session_id, AgentMetrics())
            metrics.extraction_jobs += extraction_jobs
            metrics.reflector_jobs += reflector_jobs
            metrics.failures += failures

    def _extract_entities(self, text: str) -> list[str]:
        entities = [
            token.strip(",.!?;:")
            for token in text.split()
            if token[:1].isupper() and len(token.strip(",.!?;:")) > 1
        ]
        deduped: list[str] = []
        for entity in entities:
            if entity not in deduped:
                deduped.append(entity)
        return deduped
