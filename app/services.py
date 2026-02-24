from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from uuid import UUID, uuid4

from fastapi import HTTPException, status

from app.agents import BackgroundAgentDispatcher
from app.api_models import ChatRequest, ChatResponse, MemoryResponse, SeedContextUpsertRequest
from app.config import Settings
from app.inference import build_inference_provider
from app.schemas import (
    GraphRelation,
    MemoryItem,
    MemoryKind,
    Message,
    MonologueState,
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
    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None: ...


@dataclass
class PreprocessResult:
    intent: str
    emotion: str
    entities: list[str]


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
            notes=payload.notes,
        )
        with self._lock:
            self._seeds[chat_session_id] = updated
        return updated

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None:
        with self._lock:
            return self._seeds.get(chat_session_id)


@dataclass
class CognitiveOrchestrator:
    episodic_store: EpisodicStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore
    seed_store: SeedContextStore
    model_provider: ModelProvider

    def handle_turn(self, message: Message) -> Message:
        preprocess = self._preprocess(message.content)
        semantic_context = self.vector_store.query_similar(
            chat_session_id=message.chat_session_id,
            query=message.content,
            limit=5,
        )
        graph_context = self._graph_context(
            chat_session_id=message.chat_session_id,
            entities=preprocess.entities,
        )
        prompt = self._assemble_context(
            chat_session_id=message.chat_session_id,
            user_message=message,
            preprocess=preprocess,
            semantic_context=semantic_context,
            graph_context=graph_context,
        )
        response = self.model_provider.generate(
            chat_session_id=message.chat_session_id,
            prompt=prompt,
        )

        assistant_message = Message(
            chat_session_id=message.chat_session_id,
            message_id=uuid4(),
            role="assistant",
            content=response,
        )
        self.episodic_store.append_message(assistant_message)
        self._postprocess(message=message, preprocess=preprocess)
        return assistant_message

    def _preprocess(self, content: str) -> PreprocessResult:
        lowered = content.lower()
        if any(term in lowered for term in ("nervous", "anxious", "worried", "scared")):
            emotion = "anxious"
        elif any(term in lowered for term in ("happy", "excited", "great", "good")):
            emotion = "positive"
        else:
            emotion = "neutral"

        if "?" in content:
            intent = "question"
        elif any(term in lowered for term in ("i am", "i'm", "i feel", "today")):
            intent = "status_update"
        else:
            intent = "statement"

        entities = [
            token.strip(",.!?;:")
            for token in content.split()
            if token[:1].isupper() and len(token.strip(",.!?;:")) > 1
        ]
        return PreprocessResult(intent=intent, emotion=emotion, entities=entities)

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

    def _assemble_context(
        self,
        *,
        chat_session_id: UUID,
        user_message: Message,
        preprocess: PreprocessResult,
        semantic_context: list[MemoryItem],
        graph_context: list[GraphRelation],
    ) -> str:
        recent_messages = self.episodic_store.get_recent_messages(
            chat_session_id=chat_session_id,
            limit=6,
        )
        monologue = self.monologue_store.get(chat_session_id=chat_session_id)
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)

        recent_excerpt = " | ".join(f"{msg.role}:{msg.content}" for msg in recent_messages[-4:])
        semantic_excerpt = " | ".join(item.content for item in semantic_context) or "none"
        graph_excerpt = (
            " | ".join(
                f"{relation.source}-{relation.relation}->{relation.target}"
                for relation in graph_context
            )
            or "none"
        )

        if monologue is not None and monologue.internal_monologue:
            monologue_text = monologue.internal_monologue
        else:
            monologue_text = "No prior monologue."

        seed_text = "none"
        if seed_context is not None:
            seed_text = (
                f"Companion={seed_context.seed.companion_name}; "
                f"Traits={','.join(seed_context.seed.character_traits)}; "
                f"Setup={seed_context.seed.relationship_setup}"
            )

        return (
            f"seed={seed_text}\n"
            f"internal_monologue={monologue_text}\n"
            f"intent={preprocess.intent}; emotion={preprocess.emotion}; "
            f"entities={','.join(preprocess.entities) or 'none'}\n"
            f"episodic_context={recent_excerpt or 'none'}\n"
            f"semantic_context={semantic_excerpt}\n"
            f"graph_context={graph_excerpt}\n"
            f"user_message={user_message.content}"
        )

    def _postprocess(self, *, message: Message, preprocess: PreprocessResult) -> None:
        self.vector_store.upsert_memory(
            MemoryItem(
                chat_session_id=message.chat_session_id,
                kind=MemoryKind.SEMANTIC,
                content=message.content,
                score=1.0,
            )
        )
        for entity in preprocess.entities:
            self.graph_store.upsert_relation(
                GraphRelation(
                    chat_session_id=message.chat_session_id,
                    source="user",
                    relation="MENTIONED_IN_SESSION",
                    target=entity,
                )
            )

        reflection = (
            f"Focus on a {preprocess.emotion} user; "
            f"intent={preprocess.intent}; "
            f"entities={','.join(preprocess.entities) or 'none'}"
        )
        self.monologue_store.upsert(
            MonologueState(
                chat_session_id=message.chat_session_id,
                internal_monologue=reflection,
            )
        )


@dataclass
class ChatService:
    episodic_store: EpisodicStore
    seed_store: SeedContextStore
    orchestrator: CognitiveOrchestrator
    idempotency_cache: dict[str, Message]
    agent_dispatcher: BackgroundAgentDispatcher

    def run_chat(self, *, request: ChatRequest, idempotency_key: str | None) -> ChatResponse:
        cache_key = self._cache_key(request.chat_session_id, idempotency_key)
        if cache_key is not None and cache_key in self.idempotency_cache:
            cached = self.idempotency_cache[cache_key]
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

        assistant_message = self.orchestrator.handle_turn(user_message)
        self.agent_dispatcher.enqueue_turn(
            chat_session_id=request.chat_session_id,
            user_message=user_message.content,
            assistant_message=assistant_message.content,
        )

        if cache_key is not None:
            self.idempotency_cache[cache_key] = assistant_message

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
    )
    chat_service = ChatService(
        episodic_store=episodic_store,
        seed_store=seed_store,
        orchestrator=orchestrator,
        idempotency_cache={},
        agent_dispatcher=agent_dispatcher,
    )
    return AppContainer(
        episodic_store=episodic_store,
        seed_store=seed_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        agent_dispatcher=agent_dispatcher,
        orchestrator=orchestrator,
        chat_service=chat_service,
    )
