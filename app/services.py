from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol
from uuid import UUID, uuid4

from fastapi import HTTPException, status

from app.agents import BackgroundAgentDispatcher
from app.analysis import IntentAnalyzer, build_intent_analyzer
from app.api_models import (
    ChatRequest,
    ChatResponse,
    MemoryResponse,
    SeedContextUpsertRequest,
    SessionListResponse,
    SessionSummary,
)
from app.config import Settings
from app.debug_trace import DebugTraceStore, build_trace_base, sanitize_debug_text
from app.inference import build_inference_provider
from app.prompting import build_companion_system_prompt
from app.schemas import (
    GraphRelation,
    MemoryItem,
    MemoryKind,
    Message,
    MonologueState,
    PreprocessResult,
    SessionActivity,
    SessionSeedContext,
)
from app.store_adapters import (
    Neo4jGraphStore,
    PostgresEpisodicStore,
    PostgresSeedContextStore,
    QdrantVectorStore,
)

logger = logging.getLogger(__name__)

class ModelProvider(Protocol):
    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str: ...


class EpisodicStore(Protocol):
    def append_message(self, message: Message) -> None: ...

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]: ...

    def list_session_activity(self, *, limit: int = 50) -> list[SessionActivity]: ...


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
    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None: ...

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]: ...

class InMemoryEpisodicStore:
    def __init__(self) -> None:
        self._messages: dict[UUID, list[Message]] = {}
        self._lock = Lock()

    def append_message(self, message: Message) -> None:
        with self._lock:
            bucket = self._messages.setdefault(message.chat_session_id, [])
            bucket.append(message)

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]:
        with self._lock:
            messages = list(self._messages.get(chat_session_id, []))
        return messages[-limit:]

    def list_session_activity(self, *, limit: int = 50) -> list[SessionActivity]:
        with self._lock:
            buckets = list(self._messages.items())
        activity: list[SessionActivity] = []
        for chat_session_id, messages in buckets:
            if not messages:
                continue
            sorted_messages = sorted(messages, key=lambda item: item.created_at)
            activity.append(
                SessionActivity(
                    chat_session_id=chat_session_id,
                    created_at=sorted_messages[0].created_at,
                    updated_at=sorted_messages[-1].created_at,
                    message_count=len(sorted_messages),
                )
            )
        activity.sort(key=lambda item: item.updated_at, reverse=True)
        return activity[:limit]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[UUID, list[MemoryItem]] = {}
        self._lock = Lock()

    def upsert_memory(self, item: MemoryItem) -> None:
        with self._lock:
            bucket = self._items.setdefault(item.chat_session_id, [])
            bucket.append(item)

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]:
        query_terms = {token.lower() for token in query.split()}
        scored: list[tuple[int, MemoryItem]] = []
        with self._lock:
            items = list(self._items.get(chat_session_id, []))
        for item in items:
            item_terms = {token.lower() for token in item.content.split()}
            overlap = len(query_terms.intersection(item_terms))
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda candidate: candidate[0], reverse=True)
        return [item for _, item in scored[:limit]]


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._relations: dict[UUID, list[GraphRelation]] = {}
        self._lock = Lock()

    def upsert_relation(self, relation: GraphRelation) -> None:
        with self._lock:
            bucket = self._relations.setdefault(relation.chat_session_id, [])
            bucket.append(relation)

    def get_related(
        self, *, chat_session_id: UUID, entity: str, limit: int = 10
    ) -> list[GraphRelation]:
        entity_lc = entity.lower()
        with self._lock:
            relations = list(self._relations.get(chat_session_id, []))
        related = [
            relation
            for relation in relations
            if relation.source.lower() == entity_lc or relation.target.lower() == entity_lc
        ]
        return related[:limit]


class InMemoryMonologueStore:
    def __init__(self) -> None:
        self._states: dict[UUID, MonologueState] = {}
        self._lock = Lock()

    def get(self, *, chat_session_id: UUID) -> MonologueState | None:
        with self._lock:
            return self._states.get(chat_session_id)

    def upsert(self, state: MonologueState) -> MonologueState:
        with self._lock:
            self._states[state.chat_session_id] = state
        return state


class InMemorySeedContextStore:
    def __init__(self) -> None:
        self._seeds: dict[UUID, SessionSeedContext] = {}
        self._lock = Lock()

    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        with self._lock:
            exists = chat_session_id in self._seeds
        if exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Seed context already exists for session.",
            )
        seed_context = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        with self._lock:
            self._seeds[chat_session_id] = seed_context
        return seed_context

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        with self._lock:
            existing = self._seeds.get(chat_session_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seed context not found for session.",
            )
        updated = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=existing.version + 1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        with self._lock:
            self._seeds[chat_session_id] = updated
        return updated

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None:
        with self._lock:
            return self._seeds.get(chat_session_id)

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]:
        with self._lock:
            contexts = list(self._seeds.values())
        contexts.sort(key=lambda item: item.updated_at, reverse=True)
        return contexts[:limit]


@dataclass
class CognitiveOrchestrator:
    episodic_store: EpisodicStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore
    seed_store: SeedContextStore
    model_provider: ModelProvider
    intent_analyzer: IntentAnalyzer

    def handle_turn(self, message: Message) -> tuple[Message, dict[str, Any]]:
        analysis = self.intent_analyzer.analyze(
            chat_session_id=message.chat_session_id,
            content=message.content,
        )
        preprocess = analysis.preprocess
        semantic_context = self.vector_store.query_similar(
            chat_session_id=message.chat_session_id,
            query=message.content,
            limit=5,
        )
        graph_context = self._graph_context(
            chat_session_id=message.chat_session_id,
            entities=preprocess.entities,
        )
        messages = self._assemble_messages(
            chat_session_id=message.chat_session_id,
            user_message=message,
            preprocess=preprocess,
            semantic_context=semantic_context,
            graph_context=graph_context,
        )
        response = self.model_provider.generate(
            chat_session_id=message.chat_session_id,
            messages=messages,
        )
        response = self._enforce_seeded_identity(
            chat_session_id=message.chat_session_id,
            response=response,
        )

        assistant_message = Message(
            chat_session_id=message.chat_session_id,
            message_id=uuid4(),
            role="assistant",
            content=response,
        )
        self.episodic_store.append_message(assistant_message)
        writes = self._postprocess(message=message, preprocess=preprocess)
        trace = {
            "preprocess": {
                "intent": preprocess.intent,
                "emotion": preprocess.emotion,
                "entities": preprocess.entities,
                "analysis": analysis.as_trace(),
            },
            "retrieval": {
                "semantic_items": [item.content for item in semantic_context],
                "graph_relations": [
                    f"{rel.source}-{rel.relation}->{rel.target}" for rel in graph_context
                ],
            },
            "prompt": {
                "summary": self._summarize_messages(messages),
                "messages": [
                    {"role": m["role"], "content": sanitize_debug_text(m["content"])}
                    for m in messages
                ],
            },
            "provider": {
                "name": type(self.model_provider).__name__,
            },
            "writes": writes,
        }
        return assistant_message, trace

    def _graph_context(self, *, chat_session_id: UUID, entities: list[str]) -> list[GraphRelation]:
        related: list[GraphRelation] = []
        for entity in entities:
            related.extend(
                self.graph_store.get_related(
                    chat_session_id=chat_session_id,
                    entity=entity,
                    limit=3,
                )
            )
        return related

    def _assemble_messages(
        self,
        *,
        chat_session_id: UUID,
        user_message: Message,
        preprocess: PreprocessResult,
        semantic_context: list[MemoryItem],
        graph_context: list[GraphRelation],
    ) -> list[dict[str, str]]:
        recent_messages = self.episodic_store.get_recent_messages(
            chat_session_id=chat_session_id,
            limit=6,
        )
        monologue = self.monologue_store.get(chat_session_id=chat_session_id)
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)
        companion_system_prompt = build_companion_system_prompt(seed_context)

        semantic_excerpt = " | ".join(item.content for item in semantic_context)
        graph_excerpt = " | ".join(
            f"{rel.source}-{rel.relation}->{rel.target}" for rel in graph_context
        )

        if monologue is not None and monologue.internal_monologue:
            monologue_text = monologue.internal_monologue
        else:
            monologue_text = None

        context_parts: list[str] = []
        if monologue_text:
            context_parts.append(f"Internal reflection: {monologue_text}")
        if semantic_excerpt:
            context_parts.append(f"Relevant memories: {semantic_excerpt}")
        if graph_excerpt:
            context_parts.append(f"Relationships: {graph_excerpt}")
        context_parts.append(
            f"Detected intent: {preprocess.intent}; emotion: {preprocess.emotion}"
        )

        system_content = companion_system_prompt
        if context_parts:
            system_content += (
                "\n\n## Session Context (internal — never include this in your response)\n"
                + "\n".join(context_parts)
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        # Add conversation history as proper turns, excluding the current
        # user message (which was already appended to the store before this
        # method is called).
        history = [m for m in recent_messages if m.message_id != user_message.message_id]
        for msg in history[-4:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message.content})
        return messages

    def _enforce_seeded_identity(self, *, chat_session_id: UUID, response: str) -> str:
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)
        if seed_context is None:
            return response

        companion_name = seed_context.seed.companion_name
        updated = re.sub(
            r"\bmy name is assistant\b",
            f"my name is {companion_name}",
            response,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bi am assistant\b",
            f"I am {companion_name}",
            updated,
            flags=re.IGNORECASE,
        )
        return updated

    def _postprocess(self, *, message: Message, preprocess: PreprocessResult) -> dict[str, Any]:
        semantic_item = MemoryItem(
            chat_session_id=message.chat_session_id,
            kind=MemoryKind.SEMANTIC,
            content=message.content,
            score=1.0,
        )
        self.vector_store.upsert_memory(semantic_item)
        graph_writes: list[str] = []
        for entity in preprocess.entities:
            relation = GraphRelation(
                chat_session_id=message.chat_session_id,
                source="user",
                relation="MENTIONED_IN_SESSION",
                target=entity,
            )
            self.graph_store.upsert_relation(relation)
            graph_writes.append(f"{relation.source}-{relation.relation}->{relation.target}")

        reflection = (
            f"Focus on a {preprocess.emotion} user; "
            f"intent={preprocess.intent}; "
            f"entities={','.join(preprocess.entities) or 'none'}"
        )
        monologue_state = MonologueState(
            chat_session_id=message.chat_session_id,
            internal_monologue=reflection,
        )
        self.monologue_store.upsert(monologue_state)
        return {
            "semantic_upserts": [semantic_item.content],
            "graph_upserts": graph_writes,
            "monologue": monologue_state.internal_monologue,
        }

    def _summarize_messages(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"][:80]
            parts.append(f"{role}: {content}")
        return " | ".join(parts[:4])


@dataclass
class ChatService:
    episodic_store: EpisodicStore
    seed_store: SeedContextStore
    orchestrator: CognitiveOrchestrator
    idempotency_cache: dict[str, Message]
    agent_dispatcher: BackgroundAgentDispatcher
    debug_store: DebugTraceStore

    def run_chat(
        self,
        *,
        request: ChatRequest,
        idempotency_key: str | None,
        safety_transforms: list[str] | None = None,
    ) -> ChatResponse:
        cache_key = self._cache_key(request.chat_session_id, idempotency_key)
        if cache_key is not None and cache_key in self.idempotency_cache:
            cached = self.idempotency_cache[cache_key]
            replay_trace = build_trace_base(chat_session_id=request.chat_session_id)
            replay_trace.update(
                {
                    "idempotency_replay": True,
                    "seed_version": self._seed_version(request.chat_session_id),
                    "safety_transforms": safety_transforms or [],
                    "user_message": sanitize_debug_text(request.message),
                    "assistant_message": sanitize_debug_text(cached.content),
                    "turn_trace": {"note": "idempotency replay; no new inference"},
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            self.debug_store.add_trace(chat_session_id=request.chat_session_id, trace=replay_trace)
            return ChatResponse(
                chat_session_id=request.chat_session_id,
                assistant_message=cached,
                idempotency_replay=True,
                seed_version=self._seed_version(request.chat_session_id),
            )

        user_message = Message(
            chat_session_id=request.chat_session_id,
            role="user",
            content=request.message,
        )
        self.episodic_store.append_message(user_message)

        assistant_message, trace = self.orchestrator.handle_turn(user_message)
        self.agent_dispatcher.enqueue_turn(
            chat_session_id=request.chat_session_id,
            user_message=user_message.content,
            assistant_message=assistant_message.content,
        )

        if cache_key is not None:
            self.idempotency_cache[cache_key] = assistant_message
        trace_payload = build_trace_base(chat_session_id=request.chat_session_id)
        trace_payload.update(
            {
                "idempotency_replay": False,
                "seed_version": self._seed_version(request.chat_session_id),
                "safety_transforms": safety_transforms or [],
                "user_message": sanitize_debug_text(request.message),
                "assistant_message": sanitize_debug_text(assistant_message.content),
                "turn_trace": trace,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self.debug_store.add_trace(chat_session_id=request.chat_session_id, trace=trace_payload)

        return ChatResponse(
            chat_session_id=request.chat_session_id,
            assistant_message=assistant_message,
            idempotency_replay=False,
            seed_version=self._seed_version(request.chat_session_id),
        )

    def get_memory(self, *, chat_session_id: UUID) -> MemoryResponse:
        return MemoryResponse(
            chat_session_id=chat_session_id,
            messages=self.episodic_store.get_recent_messages(
                chat_session_id=chat_session_id,
                limit=50,
            ),
            seed_context=self.seed_store.get(chat_session_id=chat_session_id),
        )

    def list_sessions(self, *, limit: int = 50) -> SessionListResponse:
        by_session: dict[UUID, SessionSummary] = {}
        for item in self.episodic_store.list_session_activity(limit=limit):
            by_session[item.chat_session_id] = SessionSummary(
                chat_session_id=item.chat_session_id,
                created_at=item.created_at,
                updated_at=item.updated_at,
                message_count=item.message_count,
                companion_name=None,
            )

        for seed in self.seed_store.list_seed_contexts(limit=limit):
            existing = by_session.get(seed.chat_session_id)
            if existing is None:
                by_session[seed.chat_session_id] = SessionSummary(
                    chat_session_id=seed.chat_session_id,
                    created_at=seed.created_at,
                    updated_at=seed.updated_at,
                    message_count=0,
                    companion_name=seed.seed.companion_name,
                )
                continue
            existing.created_at = min(existing.created_at, seed.created_at)
            existing.updated_at = max(existing.updated_at, seed.updated_at)
            existing.companion_name = seed.seed.companion_name

        sessions = sorted(
            by_session.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )[:limit]
        return SessionListResponse(sessions=sessions)

    def _seed_version(self, chat_session_id: UUID) -> int | None:
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)
        if seed_context is None:
            return None
        return seed_context.version

    def _cache_key(self, chat_session_id: UUID, idempotency_key: str | None) -> str | None:
        if idempotency_key is None:
            return None
        normalized = idempotency_key.strip()
        if not normalized:
            return None
        return f"{chat_session_id}:{normalized}"


@dataclass
class AppContainer:
    episodic_store: EpisodicStore
    seed_store: SeedContextStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore
    agent_dispatcher: BackgroundAgentDispatcher
    debug_store: DebugTraceStore
    orchestrator: CognitiveOrchestrator
    chat_service: ChatService


def _external_stores_from_settings(
    settings: Settings,
) -> tuple[EpisodicStore, VectorStore, GraphStore, SeedContextStore]:
    episodic = PostgresEpisodicStore(dsn=settings.postgres_dsn)
    vector = QdrantVectorStore(url=settings.qdrant_url)
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    seed = PostgresSeedContextStore(dsn=settings.postgres_dsn)
    episodic.ensure_schema()
    vector.ensure_schema()
    graph.ensure_schema()
    seed.ensure_schema()
    logger.info("External stores initialized.")
    return episodic, vector, graph, seed


def build_container(settings: Settings) -> AppContainer:
    monologue_store = InMemoryMonologueStore()
    model_provider = build_inference_provider(settings)
    intent_analyzer = build_intent_analyzer(settings)
    debug_store = DebugTraceStore(
        enabled=settings.debug_tracing,
        limit_per_session=settings.debug_trace_limit,
    )

    if settings.use_external_stores:
        episodic_store, vector_store, graph_store, seed_store = _external_stores_from_settings(
            settings
        )
    else:
        episodic_store = InMemoryEpisodicStore()
        vector_store = InMemoryVectorStore()
        graph_store = InMemoryGraphStore()
        seed_store = InMemorySeedContextStore()

    agent_dispatcher = BackgroundAgentDispatcher(
        episodic_store=episodic_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        enabled=settings.enable_background_agents,
    )

    orchestrator = CognitiveOrchestrator(
        episodic_store=episodic_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        seed_store=seed_store,
        model_provider=model_provider,
        intent_analyzer=intent_analyzer,
    )
    chat_service = ChatService(
        episodic_store=episodic_store,
        seed_store=seed_store,
        orchestrator=orchestrator,
        idempotency_cache={},
        agent_dispatcher=agent_dispatcher,
        debug_store=debug_store,
    )
    return AppContainer(
        episodic_store=episodic_store,
        seed_store=seed_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        agent_dispatcher=agent_dispatcher,
        debug_store=debug_store,
        orchestrator=orchestrator,
        chat_service=chat_service,
    )
