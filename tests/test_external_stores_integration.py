from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.api_models import SeedContextUpsertRequest
from app.schemas import CompanionSeed, GraphRelation, MemoryItem, MemoryKind, Message
from app.store_adapters import (
    Neo4jGraphStore,
    PostgresEpisodicStore,
    PostgresSeedContextStore,
    QdrantVectorStore,
)

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"


pytestmark = pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_INTEGRATION=1 to run")


def test_postgres_and_qdrant_and_neo4j_session_partitioning() -> None:
    postgres = PostgresEpisodicStore(
        dsn=os.getenv("POSTGRES_DSN", "postgresql://aether:aether@localhost:5432/aether")
    )
    qdrant = QdrantVectorStore(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    neo4j = Neo4jGraphStore(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    seed_store = PostgresSeedContextStore(
        dsn=os.getenv("POSTGRES_DSN", "postgresql://aether:aether@localhost:5432/aether")
    )

    postgres.ensure_schema()
    qdrant.ensure_schema()
    neo4j.ensure_schema()
    seed_store.ensure_schema()

    session_a = uuid4()
    session_b = uuid4()

    postgres.append_message(Message(chat_session_id=session_a, role="user", content="alpha memory"))
    postgres.append_message(Message(chat_session_id=session_b, role="user", content="beta memory"))

    msgs_a = postgres.get_recent_messages(chat_session_id=session_a)
    msgs_b = postgres.get_recent_messages(chat_session_id=session_b)

    assert len(msgs_a) >= 1
    assert len(msgs_b) >= 1
    assert all(msg.chat_session_id == session_a for msg in msgs_a)
    assert all(msg.chat_session_id == session_b for msg in msgs_b)

    qdrant.upsert_memory(
        MemoryItem(
            chat_session_id=session_a,
            kind=MemoryKind.SEMANTIC,
            content="Sarah conflict memory",
        )
    )
    qdrant.upsert_memory(
        MemoryItem(
            chat_session_id=session_b,
            kind=MemoryKind.SEMANTIC,
            content="Different person memory",
        )
    )

    vector_a = qdrant.query_similar(chat_session_id=session_a, query="Sarah", limit=10)
    vector_b = qdrant.query_similar(chat_session_id=session_b, query="Sarah", limit=10)

    assert all(item.chat_session_id == session_a for item in vector_a)
    assert all(item.chat_session_id == session_b for item in vector_b)

    neo4j.upsert_relation(
        GraphRelation(
            chat_session_id=session_a,
            source="Sarah",
            relation="IS_SISTER_OF",
            target="User",
        )
    )
    neo4j.upsert_relation(
        GraphRelation(
            chat_session_id=session_b,
            source="Jordan",
            relation="IS_FRIEND_OF",
            target="User",
        )
    )

    graph_a = neo4j.get_related(chat_session_id=session_a, entity="Sarah")
    graph_b = neo4j.get_related(chat_session_id=session_b, entity="Sarah")

    assert len(graph_a) >= 1
    assert all(rel.chat_session_id == session_a for rel in graph_a)
    assert graph_b == []

    created = seed_store.create(
        chat_session_id=session_a,
        payload=SeedContextUpsertRequest(
            seed=CompanionSeed(
                companion_name="Ari",
                backstory="Ari is reflective.",
                character_traits=["calm"],
                goals=["build trust"],
                relationship_setup="Long-term companion.",
            ),
            notes="initial",
        ),
    )
    updated = seed_store.update(
        chat_session_id=session_a,
        payload=SeedContextUpsertRequest(
            seed=CompanionSeed(
                companion_name="Ari v2",
                backstory="Ari evolves.",
                character_traits=["calm", "curious"],
                goals=["build trust", "be consistent"],
                relationship_setup="Long-term companion.",
            ),
            notes="updated",
        ),
    )
    fetched = seed_store.get(chat_session_id=session_a)

    assert created.version == 1
    assert updated.version == 2
    assert fetched is not None
    assert fetched.version == 2
    assert fetched.seed.companion_name == "Ari v2"
