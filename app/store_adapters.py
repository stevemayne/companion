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
    WorldState,
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
                      speaker_id,
                      speaker_name,
                      content,
                      created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        _uuid_text(message.chat_session_id),
                        _uuid_text(message.message_id),
                        message.role,
                        _uuid_text(message.speaker_id) if message.speaker_id else None,
                        message.speaker_name,
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
                    SELECT chat_session_id, message_id, role, speaker_id,
                           speaker_name, content, created_at
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
                speaker_id=row[3],
                speaker_name=row[4],
                content=row[5],
                created_at=row[6],
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
    _DEFAULT_COMPANION_ID = "00000000-0000-0000-0000-000000000001"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def get(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> MonologueState | None:
        cid = _uuid_text(companion_id) if companion_id else self._DEFAULT_COMPANION_ID
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chat_session_id, companion_id, internal_monologue,
                           affect, user_state, updated_at, world
                    FROM monologue_states
                    WHERE chat_session_id = %s AND companion_id = %s
                    """,
                    (_uuid_text(chat_session_id), cid),
                )
                row = cur.fetchone()
        if row is None:
            return None
        affect_raw = row[3]
        affect = CompanionAffect.model_validate(affect_raw if affect_raw else {})
        user_state_raw = row[4]
        user_state = user_state_raw if isinstance(user_state_raw, list) else []
        world_raw = row[6]
        world = WorldState.model_validate(world_raw) if isinstance(world_raw, dict) and world_raw else WorldState()
        return MonologueState(
            chat_session_id=row[0],
            companion_id=row[1],
            internal_monologue=row[2],
            affect=affect,
            user_state=user_state,
            world=world,
            updated_at=row[5],
        )

    def upsert(self, state: MonologueState) -> MonologueState:
        cid = (
            _uuid_text(state.companion_id)
            if state.companion_id
            else self._DEFAULT_COMPANION_ID
        )
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monologue_states (
                      chat_session_id,
                      companion_id,
                      internal_monologue,
                      affect,
                      user_state,
                      world,
                      updated_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (chat_session_id, companion_id)
                    DO UPDATE SET
                      internal_monologue = EXCLUDED.internal_monologue,
                      affect = EXCLUDED.affect,
                      user_state = EXCLUDED.user_state,
                      world = EXCLUDED.world,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (
                        _uuid_text(state.chat_session_id),
                        cid,
                        state.internal_monologue,
                        json.dumps(state.affect.model_dump()),
                        json.dumps(state.user_state),
                        json.dumps(state.world.model_dump()),
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
            info = self._client.get_collection(self.COLLECTION_NAME)
            current_dim = getattr(info.config.params.vectors, "size", None)
            if current_dim != self._dimensions:
                self._client.delete_collection(self.COLLECTION_NAME)
            else:
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
            "companion_id": _uuid_text(item.companion_id) if item.companion_id else None,
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
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
        query: str, limit: int = 10,
    ) -> list[MemoryItem]:
        # Fetch more candidates than requested so we can re-rank after
        # applying importance and recency weighting.
        fetch_limit = min(limit * 3, 100)
        must_filters = [
            qdrant_models.FieldCondition(
                key="chat_session_id",
                match=qdrant_models.MatchValue(value=_uuid_text(chat_session_id)),
            ),
            qdrant_models.FieldCondition(
                key="status",
                match=qdrant_models.MatchValue(value=MemoryStatus.ACTIVE.value),
            ),
        ]
        if companion_id is not None:
            must_filters.append(
                qdrant_models.FieldCondition(
                    key="companion_id",
                    match=qdrant_models.MatchValue(value=_uuid_text(companion_id)),
                )
            )
        points = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=self._embedder.embed(query),
            limit=fetch_limit,
            query_filter=qdrant_models.Filter(must=must_filters),
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

            raw_cid = payload.get("companion_id")
            item = MemoryItem(
                chat_session_id=chat_session_id,
                companion_id=UUID(raw_cid) if raw_cid else None,
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

    def list_memories(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> list[MemoryItem]:
        must_filters = [
            qdrant_models.FieldCondition(
                key="chat_session_id",
                match=qdrant_models.MatchValue(value=_uuid_text(chat_session_id)),
            ),
        ]
        if companion_id is not None:
            must_filters.append(
                qdrant_models.FieldCondition(
                    key="companion_id",
                    match=qdrant_models.MatchValue(value=_uuid_text(companion_id)),
                )
            )
        records, _next_offset = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=qdrant_models.Filter(must=must_filters),
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
            raw_cid = payload.get("companion_id")
            items.append(
                MemoryItem(
                    chat_session_id=chat_session_id,
                    companion_id=UUID(raw_cid) if raw_cid else None,
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


class PostgresSeedContextStore:
    _DEFAULT_COMPANION_ID = "00000000-0000-0000-0000-000000000001"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest,
        companion_id: UUID | None = None,
    ) -> SessionSeedContext:
        from uuid import uuid4 as _uuid4
        cid = companion_id or _uuid4()
        context = SessionSeedContext(
            chat_session_id=chat_session_id,
            companion_id=cid,
            version=1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        self._upsert_row(context)
        return context

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest,
        companion_id: UUID | None = None,
    ) -> SessionSeedContext:
        existing = self.get(
            chat_session_id=chat_session_id, companion_id=companion_id,
        )
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seed context not found.",
            )
        context = SessionSeedContext(
            chat_session_id=chat_session_id,
            companion_id=existing.companion_id,
            version=existing.version + 1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
            created_at=existing.created_at,
        )
        self._upsert_row(context)
        return context

    def get(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> SessionSeedContext | None:
        sid = _uuid_text(chat_session_id)
        if companion_id is not None:
            where = "WHERE chat_session_id = %s AND companion_id = %s"
            params = (sid, _uuid_text(companion_id))
        else:
            # Backward compat: return first companion for session
            where = "WHERE chat_session_id = %s ORDER BY created_at LIMIT 1"
            params = (sid,)
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                      companion_id,
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      created_at,
                      updated_at
                    FROM session_seed_contexts
                    {where}
                    """,
                    params,
                )
                row = cur.fetchone()
        if row is None:
            return None
        raw_traits = row[4]
        raw_goals = row[5]
        traits = json.loads(raw_traits) if isinstance(raw_traits, str) else list(raw_traits)
        goals = json.loads(raw_goals) if isinstance(raw_goals, str) else list(raw_goals)
        return SessionSeedContext(
            chat_session_id=chat_session_id,
            companion_id=row[0],
            version=row[1],
            seed=CompanionSeed(
                companion_name=row[2],
                backstory=row[3],
                character_traits=traits,
                goals=goals,
                relationship_setup=row[6],
            ),
            user_description=row[7],
            notes=row[8],
            created_at=row[9],
            updated_at=row[10],
        )

    def list_for_session(
        self, *, chat_session_id: UUID,
    ) -> list[SessionSeedContext]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      companion_id, version, companion_name, backstory,
                      character_traits, goals, relationship_setup,
                      user_description, notes, created_at, updated_at
                    FROM session_seed_contexts
                    WHERE chat_session_id = %s
                    ORDER BY created_at
                    """,
                    (_uuid_text(chat_session_id),),
                )
                rows = cur.fetchall()
        return [self._row_to_context(chat_session_id, row) for row in rows]

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      chat_session_id,
                      companion_id,
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      created_at,
                      updated_at
                    FROM session_seed_contexts
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        contexts: list[SessionSeedContext] = []
        for row in rows:
            raw_traits = row[5]
            raw_goals = row[6]
            traits = json.loads(raw_traits) if isinstance(raw_traits, str) else list(raw_traits)
            goals = json.loads(raw_goals) if isinstance(raw_goals, str) else list(raw_goals)
            contexts.append(
                SessionSeedContext(
                    chat_session_id=row[0],
                    companion_id=row[1],
                    version=row[2],
                    seed=CompanionSeed(
                        companion_name=row[3],
                        backstory=row[4],
                        character_traits=traits,
                        goals=goals,
                        relationship_setup=row[7],
                    ),
                    user_description=row[8],
                    notes=row[9],
                    created_at=row[10],
                    updated_at=row[11],
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
                      companion_id,
                      version,
                      companion_name,
                      backstory,
                      character_traits,
                      goals,
                      relationship_setup,
                      user_description,
                      notes,
                      created_at,
                      updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                    ON CONFLICT (chat_session_id, companion_id)
                    DO UPDATE SET
                      version = EXCLUDED.version,
                      companion_name = EXCLUDED.companion_name,
                      backstory = EXCLUDED.backstory,
                      character_traits = EXCLUDED.character_traits,
                      goals = EXCLUDED.goals,
                      relationship_setup = EXCLUDED.relationship_setup,
                      user_description = EXCLUDED.user_description,
                      notes = EXCLUDED.notes,
                      updated_at = EXCLUDED.updated_at
                    """,
                    (
                        _uuid_text(context.chat_session_id),
                        _uuid_text(context.companion_id),
                        context.version,
                        context.seed.companion_name,
                        context.seed.backstory,
                        json.dumps(context.seed.character_traits),
                        json.dumps(context.seed.goals),
                        context.seed.relationship_setup,
                        context.user_description,
                        context.notes,
                        context.created_at,
                        context.updated_at,
                    ),
                )
            conn.commit()

    def _row_to_context(
        self, chat_session_id: UUID, row: tuple,
    ) -> SessionSeedContext:
        raw_traits = row[4]
        raw_goals = row[5]
        traits = json.loads(raw_traits) if isinstance(raw_traits, str) else list(raw_traits)
        goals = json.loads(raw_goals) if isinstance(raw_goals, str) else list(raw_goals)
        return SessionSeedContext(
            chat_session_id=chat_session_id,
            companion_id=row[0],
            version=row[1],
            seed=CompanionSeed(
                companion_name=row[2],
                backstory=row[3],
                character_traits=traits,
                goals=goals,
                relationship_setup=row[6],
            ),
            user_description=row[7],
            notes=row[8],
            created_at=row[9],
            updated_at=row[10],
        )


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
            # Backfill companion_id on legacy nodes/relationships that predate
            # the multi-companion architecture.
            session.run(
                'MATCH (e:Entity) WHERE e.companion_id IS NULL SET e.companion_id = ""'
            )
            session.run(
                'MATCH ()-[r:RELATES_TO]-() WHERE r.companion_id IS NULL SET r.companion_id = ""'
            )

    def upsert_relation(self, relation: GraphRelation) -> None:
        cid = _uuid_text(relation.companion_id) if relation.companion_id else ""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $source, chat_session_id: $sid, companion_id: $cid})
                MERGE (t:Entity {name: $target, chat_session_id: $sid, companion_id: $cid})
                MERGE (s)-[r:RELATES_TO {
                  chat_session_id: $sid,
                  companion_id: $cid,
                  relation: $relation
                }]->(t)
                SET r.confidence = $confidence
                """,
                source=relation.source,
                target=relation.target,
                relation=relation.relation,
                confidence=relation.confidence,
                sid=_uuid_text(relation.chat_session_id),
                cid=cid,
            )

    def get_related(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
        entity: str, limit: int = 10,
    ) -> list[GraphRelation]:
        sid = _uuid_text(chat_session_id)
        cid = _uuid_text(companion_id) if companion_id else ""
        cid_clause = "AND s.companion_id = $cid AND r.companion_id = $cid" if companion_id else ""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (s:Entity {{chat_session_id: $sid}})
                -[r:RELATES_TO {{chat_session_id: $sid}}]->
                (t:Entity {{chat_session_id: $sid}})
                WHERE (toLower(s.name) = toLower($entity)
                   OR toLower(t.name) = toLower($entity))
                {cid_clause}
                RETURN s.name AS source,
                       r.relation AS relation,
                       t.name AS target,
                       r.confidence AS confidence
                LIMIT $limit
                """,
                sid=sid, cid=cid, entity=entity, limit=limit,
            )
            return [
                GraphRelation(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                    source=record["source"],
                    relation=record["relation"],
                    target=record["target"],
                    confidence=float(record["confidence"]),
                )
                for record in result
            ]

    def list_relations(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> list[GraphRelation]:
        sid = _uuid_text(chat_session_id)
        cid = _uuid_text(companion_id) if companion_id else ""
        cid_clause = "AND s.companion_id = $cid AND r.companion_id = $cid" if companion_id else ""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (s:Entity {{chat_session_id: $sid}})
                -[r:RELATES_TO {{chat_session_id: $sid}}]->
                (t:Entity {{chat_session_id: $sid}})
                WHERE true {cid_clause}
                RETURN s.name AS source,
                       r.relation AS relation,
                       t.name AS target,
                       r.confidence AS confidence
                """,
                sid=sid, cid=cid,
            )
            return [
                GraphRelation(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                    source=record["source"],
                    relation=record["relation"],
                    target=record["target"],
                    confidence=float(record["confidence"]),
                )
                for record in result
            ]
