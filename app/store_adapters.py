from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
from fastapi import HTTPException, status
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.api_models import SeedContextUpsertRequest
from app.embedding import EmbeddingProvider
from app.schemas import (
    CharacterDef,
    CompanionAffect,
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Message,
    MonologueState,
    SessionActivity,
    SessionSeedContext,
)


def _uuid_text(value: UUID) -> str:
    return str(value)


class PostgresEpisodicStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def append_message(self, message: Message) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO episodic_messages (
                      chat_session_id,
                      message_id,
                      role,
                      name,
                      content,
                      created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        _uuid_text(message.chat_session_id),
                        _uuid_text(message.message_id),
                        message.role,
                        message.name,
                        message.content,
                        message.created_at,
                    ),
                )
            conn.commit()

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chat_session_id, message_id, role, name, content, created_at
                    FROM episodic_messages
                    WHERE chat_session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (_uuid_text(chat_session_id), limit),
                )
                rows = cur.fetchall()
        rows.reverse()
        return [
            Message(
                chat_session_id=row[0],
                message_id=row[1],
                role=row[2],
                name=row[3],
                content=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def list_session_activity(self, *, limit: int = 50) -> list[SessionActivity]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      chat_session_id,
                      MIN(created_at) AS created_at,
                      MAX(created_at) AS updated_at,
                      COUNT(*) AS message_count
                    FROM episodic_messages
                    GROUP BY chat_session_id
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            SessionActivity(
                chat_session_id=row[0],
                created_at=row[1],
                updated_at=row[2],
                message_count=int(row[3]),
            )
            for row in rows
        ]


class PostgresMonologueStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def get(
        self, *, chat_session_id: UUID, character_name: str | None = None,
    ) -> MonologueState | None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                if character_name is None:
                    cur.execute(
                        """
                        SELECT chat_session_id, internal_monologue, affect,
                               user_state, updated_at, character_name
                        FROM monologue_states
                        WHERE chat_session_id = %s
                          AND character_name IS NULL
                        """,
                        (_uuid_text(chat_session_id),),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chat_session_id, internal_monologue, affect,
                               user_state, updated_at, character_name
                        FROM monologue_states
                        WHERE chat_session_id = %s
                          AND character_name = %s
                        """,
                        (_uuid_text(chat_session_id), character_name),
                    )
                row = cur.fetchone()
        if row is None:
            return None
        affect_raw = row[2]
        affect = CompanionAffect.model_validate(affect_raw if affect_raw else {})
        user_state_raw = row[3]
        user_state = user_state_raw if isinstance(user_state_raw, list) else []
        return MonologueState(
            chat_session_id=row[0],
            internal_monologue=row[1],
            affect=affect,
            user_state=user_state,
            updated_at=row[4],
            character_name=row[5],
        )

    def upsert(self, state: MonologueState) -> MonologueState:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monologue_states (
                      chat_session_id,
                      character_name,
                      internal_monologue,
                      affect,
                      user_state,
                      updated_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (chat_session_id, COALESCE(character_name, ''))
                    DO UPDATE SET
                      internal_monologue = EXCLUDED.internal_monologue,
                      affect = EXCLUDED.affect,
                      user_state = EXCLUDED.user_state,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (
                        _uuid_text(state.chat_session_id),
                        state.character_name,
                        state.internal_monologue,
                        json.dumps(state.affect.model_dump()),
                        json.dumps(state.user_state),
                        state.updated_at,
                    ),
                )
            conn.commit()
        return state


class QdrantVectorStore:
    COLLECTION_NAME = "aether_semantic_memory"

    def __init__(
        self, url: str, embedder: EmbeddingProvider, dimensions: int = 1536,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._embedder = embedder
        self._dimensions = dimensions

    def ensure_schema(self) -> None:
        existing = {
            collection.name
            for collection in self._client.get_collections().collections
        }
        if self.COLLECTION_NAME in existing:
            return
        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=self._dimensions,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def upsert_memory(self, item: MemoryItem) -> None:
        payload: dict[str, Any] = {
            "chat_session_id": _uuid_text(item.chat_session_id),
            "kind": item.kind.value,
            "content": item.content,
            "score": item.score,
            "importance": item.importance,
            "access_count": item.access_count,
            "last_accessed": item.last_accessed.isoformat() if item.last_accessed else None,
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
        }
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=_uuid_text(item.memory_id),
                    vector=self._embedder.embed(item.content),
                    payload=payload,
                )
            ],
        )

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]:
        # Fetch more candidates than requested so we can re-rank after
        # applying importance and recency weighting.
        fetch_limit = min(limit * 3, 100)
        points = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=self._embedder.embed(query),
            limit=fetch_limit,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chat_session_id",
                        match=qdrant_models.MatchValue(value=_uuid_text(chat_session_id)),
                    ),
                    qdrant_models.FieldCondition(
                        key="status",
                        match=qdrant_models.MatchValue(value=MemoryStatus.ACTIVE.value),
                    ),
                ]
            ),
        ).points

        now = datetime.now(UTC)
        scored: list[tuple[float, MemoryItem]] = []
        for point in points:
            payload = point.payload or {}
            kind_raw = str(payload.get("kind", MemoryKind.SEMANTIC.value))
            kind = MemoryKind(kind_raw)
            importance = float(payload.get("importance", 0.5))
            access_count = int(payload.get("access_count", 0))
            last_accessed_raw = payload.get("last_accessed")
            last_accessed = (
                datetime.fromisoformat(last_accessed_raw)
                if last_accessed_raw
                else None
            )
            created_at_raw = payload.get("created_at")
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if created_at_raw
                else now
            )
            status_raw = str(payload.get("status", MemoryStatus.ACTIVE.value))
            try:
                mem_status = MemoryStatus(status_raw)
            except ValueError:
                mem_status = MemoryStatus.ACTIVE

            cosine_sim = float(point.score) if point.score is not None else 0.0
            recency_ref = last_accessed or created_at
            days_since = max(0.0, (now - recency_ref).total_seconds() / 86400)
            recency_factor = math.exp(-0.01 * days_since)
            final_score = cosine_sim * importance * recency_factor

            item = MemoryItem(
                chat_session_id=chat_session_id,
                memory_id=UUID(str(point.id)) if point.id else None,
                kind=kind,
                content=str(payload.get("content", "")),
                score=final_score,
                importance=importance,
                access_count=access_count,
                last_accessed=last_accessed,
                status=mem_status,
                created_at=created_at,
            )
            scored.append((final_score, item))

        scored.sort(key=lambda c: c[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def update_access(self, *, memory_id: UUID) -> None:
        """Increment access_count and update last_accessed for a memory."""
        records, _ = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.HasIdCondition(
                        has_id=[_uuid_text(memory_id)],
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return
        record = records[0]
        payload = dict(record.payload or {})
        payload["access_count"] = int(payload.get("access_count", 0)) + 1
        payload["last_accessed"] = datetime.now(UTC).isoformat()
        self._client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=payload,
            points=[_uuid_text(memory_id)],
        )

    def update_memory(
        self, *, memory_id: UUID, importance: float | None = None,
        status: MemoryStatus | None = None,
    ) -> None:
        """Update importance and/or status on an existing memory point."""
        updates: dict[str, object] = {}
        if importance is not None:
            updates["importance"] = importance
        if status is not None:
            updates["status"] = status.value
        if not updates:
            return
        self._client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=updates,
            points=[_uuid_text(memory_id)],
        )

    def list_memories(self, *, chat_session_id: UUID) -> list[MemoryItem]:
        records, _next_offset = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chat_session_id",
                        match=qdrant_models.MatchValue(value=_uuid_text(chat_session_id)),
                    )
                ]
            ),
            limit=1000,
        )
        items: list[MemoryItem] = []
        for record in records:
            payload = record.payload or {}
            kind_raw = str(payload.get("kind", MemoryKind.SEMANTIC.value))
            kind = MemoryKind(kind_raw)
            last_accessed_raw = payload.get("last_accessed")
            created_at_raw = payload.get("created_at")
            status_raw = str(payload.get("status", MemoryStatus.ACTIVE.value))
            try:
                mem_status = MemoryStatus(status_raw)
            except ValueError:
                mem_status = MemoryStatus.ACTIVE
            items.append(
                MemoryItem(
                    chat_session_id=chat_session_id,
                    memory_id=UUID(str(record.id)) if record.id else None,
                    kind=kind,
                    content=str(payload.get("content", "")),
                    importance=float(payload.get("importance", 0.5)),
                    access_count=int(payload.get("access_count", 0)),
                    last_accessed=(
                        datetime.fromisoformat(last_accessed_raw)
                        if last_accessed_raw
                        else None
                    ),
                    status=mem_status,
                    created_at=(
                        datetime.fromisoformat(created_at_raw)
                        if created_at_raw
                        else datetime.now(UTC)
                    ),
                )
            )
        return items


def _parse_characters(raw: Any) -> list[CharacterDef]:
    if raw is None:
        return []
    items = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(items, list):
        return []
    return [CharacterDef.model_validate(c) for c in items]


class PostgresSeedContextStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        if self.get(chat_session_id=chat_session_id) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Seed context already exists for session.",
            )
        context = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        self._upsert_row(context)
        return context

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        existing = self.get(chat_session_id=chat_session_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seed context not found for session.",
            )
        context = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=existing.version + 1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
            created_at=existing.created_at,
        )
        self._upsert_row(context)
        return context

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      created_at,
                      updated_at,
                      characters
                    FROM session_seed_contexts
                    WHERE chat_session_id = %s
                    """,
                    (_uuid_text(chat_session_id),),
                )
                row = cur.fetchone()
        if row is None:
            return None
        raw_traits = row[3]
        raw_goals = row[4]
        traits = json.loads(raw_traits) if isinstance(raw_traits, str) else list(raw_traits)
        goals = json.loads(raw_goals) if isinstance(raw_goals, str) else list(raw_goals)
        raw_chars = row[10]
        characters = _parse_characters(raw_chars)
        return SessionSeedContext(
            chat_session_id=chat_session_id,
            version=row[0],
            seed=CompanionSeed(
                companion_name=row[1],
                backstory=row[2],
                character_traits=traits,
                goals=goals,
                relationship_setup=row[5],
                characters=characters,
            ),
            user_description=row[6],
            notes=row[7],
            created_at=row[8],
            updated_at=row[9],
        )

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      chat_session_id,
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      created_at,
                      updated_at,
                      characters
                    FROM session_seed_contexts
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        contexts: list[SessionSeedContext] = []
        for row in rows:
            raw_traits = row[4]
            raw_goals = row[5]
            traits = json.loads(raw_traits) if isinstance(raw_traits, str) else list(raw_traits)
            goals = json.loads(raw_goals) if isinstance(raw_goals, str) else list(raw_goals)
            characters = _parse_characters(row[11])
            contexts.append(
                SessionSeedContext(
                    chat_session_id=row[0],
                    version=row[1],
                    seed=CompanionSeed(
                        companion_name=row[2],
                        backstory=row[3],
                        character_traits=traits,
                        goals=goals,
                        relationship_setup=row[6],
                        characters=characters,
                    ),
                    user_description=row[7],
                    notes=row[8],
                    created_at=row[9],
                    updated_at=row[10],
                )
            )
        return contexts

    def _upsert_row(self, context: SessionSeedContext) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_seed_contexts (
                      chat_session_id,
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      characters,
                      created_at,
                      updated_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (chat_session_id)
                    DO UPDATE SET
                      version = EXCLUDED.version,
                      companion_name = EXCLUDED.companion_name,
                      backstory = EXCLUDED.backstory,
                      character_traits = EXCLUDED.character_traits,
                      goals = EXCLUDED.goals,
                      relationship_setup = EXCLUDED.relationship_setup,
                      user_description = EXCLUDED.user_description,
                      notes = EXCLUDED.notes,
                      characters = EXCLUDED.characters,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (
                        _uuid_text(context.chat_session_id),
                        context.version,
                        context.seed.companion_name,
                        context.seed.backstory,
                        json.dumps(context.seed.character_traits),
                        json.dumps(context.seed.goals),
                        context.seed.relationship_setup,
                        context.user_description,
                        context.notes,
                        json.dumps([c.model_dump() for c in context.seed.characters]),
                        context.created_at,
                        context.updated_at,
                    ),
                )
            conn.commit()


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            session.run(
                """
                CREATE INDEX relation_session_idx IF NOT EXISTS
                FOR ()-[r:RELATES_TO]-()
                ON (r.chat_session_id)
                """
            )

    def upsert_relation(self, relation: GraphRelation) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $source, chat_session_id: $chat_session_id})
                MERGE (t:Entity {name: $target, chat_session_id: $chat_session_id})
                MERGE (s)-[r:RELATES_TO {
                  chat_session_id: $chat_session_id,
                  relation: $relation
                }]->(t)
                SET r.confidence = $confidence
                """,
                source=relation.source,
                target=relation.target,
                relation=relation.relation,
                confidence=relation.confidence,
                chat_session_id=_uuid_text(relation.chat_session_id),
            )

    def get_related(
        self, *, chat_session_id: UUID, entity: str, limit: int = 10
    ) -> list[GraphRelation]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {chat_session_id: $chat_session_id})
                -[r:RELATES_TO {chat_session_id: $chat_session_id}]->
                (t:Entity {chat_session_id: $chat_session_id})
                WHERE toLower(s.name) = toLower($entity)
                   OR toLower(t.name) = toLower($entity)
                RETURN s.name AS source,
                       r.relation AS relation,
                       t.name AS target,
                       r.confidence AS confidence
                LIMIT $limit
                """,
                chat_session_id=_uuid_text(chat_session_id),
                entity=entity,
                limit=limit,
            )
            return [
                GraphRelation(
                    chat_session_id=chat_session_id,
                    source=record["source"],
                    relation=record["relation"],
                    target=record["target"],
                    confidence=float(record["confidence"]),
                )
                for record in result
            ]

    def list_relations(self, *, chat_session_id: UUID) -> list[GraphRelation]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {chat_session_id: $chat_session_id})
                -[r:RELATES_TO {chat_session_id: $chat_session_id}]->
                (t:Entity {chat_session_id: $chat_session_id})
                RETURN s.name AS source,
                       r.relation AS relation,
                       t.name AS target,
                       r.confidence AS confidence
                """,
                chat_session_id=_uuid_text(chat_session_id),
            )
            return [
                GraphRelation(
                    chat_session_id=chat_session_id,
                    source=record["source"],
                    relation=record["relation"],
                    target=record["target"],
                    confidence=float(record["confidence"]),
                )
                for record in result
            ]
