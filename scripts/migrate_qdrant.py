"""Migrate Qdrant collection from old 16-dim hash vectors to real embeddings.

This script:
1. Reads all existing points from the aether_semantic_memory collection
2. Drops and recreates the collection with the configured vector dimensions
3. Re-embeds all existing memories using the configured embedding provider
4. Upserts the re-embedded points back into the new collection

Usage:
    uv run python scripts/migrate_qdrant.py [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qdrant_models  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embedding import MockEmbeddingProvider, OpenAICompatibleEmbeddingProvider  # noqa: E402

COLLECTION = "aether_semantic_memory"


def _build_embedder(settings: Settings):
    provider = settings.embedding_provider.strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    return MockEmbeddingProvider(dimensions=64)


def migrate(*, dry_run: bool = False) -> int:
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url)
    embedder = _build_embedder(settings)

    # Use actual embedder dimensions: mock produces 64-dim, real provider
    # uses the configured embedding_dimensions.
    if isinstance(embedder, MockEmbeddingProvider):
        dimensions = 64
    else:
        dimensions = settings.embedding_dimensions

    # Check if collection exists
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        print(f"Collection '{COLLECTION}' does not exist. Creating fresh.")
        if not dry_run:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=dimensions,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            print(f"Created collection with {dimensions}-dim vectors.")
        return 0

    # Read all existing points
    print(f"Reading existing points from '{COLLECTION}'...")
    all_points = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    print(f"Found {len(all_points)} existing points.")

    if dry_run:
        print("[DRY RUN] Would drop and recreate collection, then re-embed all points.")
        return 0

    # Drop and recreate
    print(f"Dropping collection '{COLLECTION}'...")
    client.delete_collection(collection_name=COLLECTION)

    print(f"Creating collection with {dimensions}-dim vectors...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qdrant_models.VectorParams(
            size=dimensions,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    if not all_points:
        print("No points to migrate. Done.")
        return 0

    # Re-embed and upsert in batches
    batch_size = 50
    total_migrated = 0
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i : i + batch_size]
        texts = [str(p.payload.get("content", "")) for p in batch if p.payload]
        if not texts:
            continue

        vectors = embedder.embed_batch(texts)

        points = []
        for point, vec in zip(batch, vectors, strict=True):
            points.append(
                qdrant_models.PointStruct(
                    id=point.id,
                    vector=vec,
                    payload=point.payload,
                )
            )

        client.upsert(collection_name=COLLECTION, points=points)
        total_migrated += len(points)
        print(f"  Migrated {total_migrated}/{len(all_points)} points...")

    print(
        f"\nMigration complete: {total_migrated} points "
        f"re-embedded with {dimensions}-dim vectors."
    )
    return 0


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    raise SystemExit(migrate(dry_run=dry_run))
