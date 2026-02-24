from __future__ import annotations

import hashlib
import json
import struct
from typing import Any
from uuid import UUID

import psycopg
from fastapi import HTTPException, status
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.api_models import SeedContextUpsertRequest
from app.schemas import (
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    MemoryKind,
    Message,
    SessionSeedContext,
)


def _uuid_text(value: UUID) -> str:
    return str(value)


def embed_text(text: str, size: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    floats: list[float] = []
    for idx in range(size):
        start = (idx * 4) % len(digest)
        chunk = digest[start : start + 4]
        if len(chunk) < 4:
            chunk = (chunk + digest)[:4]
        value = struct.unpack("!I", chunk)[0]
        floats.append((value % 1000) / 1000.0)
    return floats


class PostgresEpisodicStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodic_messages (
                      chat_session_id UUID NOT NULL,
                      message_id UUID PRIMARY KEY,
                      role TEXT NOT NULL,
                      content TEXT NOT NULL,
                      created_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_episodic_session_created
                    ON episodic_messages(chat_session_id, created_at DESC);
                    """
                )
            conn.commit()

    def append_message(self, message: Message) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO episodic_messages (
                      chat_session_id,
                      message_id,
                      role,
                      content,
                      created_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        _uuid_text(message.chat_session_id),
                        _uuid_text(message.message_id),
                        message.role,
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
                    SELECT chat_session_id, message_id, role, content, created_at
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
                content=row[3],
                created_at=row[4],
            )
            for row in rows
        ]


class QdrantVectorStore:
    COLLECTION_NAME = "aether_semantic_memory"

    def __init__(self, url: str) -> None:
        self._client = QdrantClient(url=url)

    def ensure_schema(self) -> None:
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self.COLLECTION_NAME in existing:
            return
        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=16,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def upsert_memory(self, item: MemoryItem) -> None:
        payload: dict[str, Any] = {
            "chat_session_id": _uuid_text(item.chat_session_id),
            "kind": item.kind.value,
            "content": item.content,
            "score": item.score,
            "created_at": item.created_at.isoformat(),
        }
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=_uuid_text(item.memory_id),
                    vector=embed_text(item.content),
                    payload=payload,
                )
            ],
        )

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]:
        points = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embed_text(query),
            limit=limit,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chat_session_id",
                        match=qdrant_models.MatchValue(value=_uuid_text(chat_session_id)),
                    )
                ]
            ),
        ).points

        items: list[MemoryItem] = []
        for point in points:
            payload = point.payload or {}
            kind_raw = str(payload.get("kind", MemoryKind.SEMANTIC.value))
            kind = MemoryKind(kind_raw)
            items.append(
                MemoryItem(
                    chat_session_id=chat_session_id,
                    kind=kind,
                    content=str(payload.get("content", "")),
                    score=float(point.score) if point.score is not None else None,
                )
            )
        return items


class PostgresSeedContextStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_seed_contexts (
                      chat_session_id UUID PRIMARY KEY,
                      version INTEGER NOT NULL,
                      companion_name TEXT NOT NULL,
                      backstory TEXT NOT NULL,
                      character_traits JSONB NOT NULL,
                      goals JSONB NOT NULL,
                      relationship_setup TEXT NOT NULL,
                      notes TEXT,
                      created_at TIMESTAMPTZ NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
            conn.commit()

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
                      notes,
                      created_at,
                      updated_at
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
        return SessionSeedContext(
            chat_session_id=chat_session_id,
            version=row[0],
            seed=CompanionSeed(
                companion_name=row[1],
                backstory=row[2],
                character_traits=traits,
                goals=goals,
                relationship_setup=row[5],
            ),
            notes=row[6],
            created_at=row[7],
            updated_at=row[8],
        )

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
                      notes,
                      created_at,
                      updated_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (chat_session_id)
                    DO UPDATE SET
                      version = EXCLUDED.version,
                      companion_name = EXCLUDED.companion_name,
                      backstory = EXCLUDED.backstory,
                      character_traits = EXCLUDED.character_traits,
                      goals = EXCLUDED.goals,
                      relationship_setup = EXCLUDED.relationship_setup,
                      notes = EXCLUDED.notes,
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
                        context.notes,
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
