from __future__ import annotations

import json
import math

import httpx
import pytest

from app.embedding import (
    MockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# MockEmbeddingProvider
# ---------------------------------------------------------------------------


def test_mock_embed_returns_correct_dimensions() -> None:
    provider = MockEmbeddingProvider(dimensions=32)
    vec = provider.embed("hello world")
    assert len(vec) == 32


def test_mock_embed_is_deterministic() -> None:
    provider = MockEmbeddingProvider()
    assert provider.embed("test") == provider.embed("test")


def test_mock_embed_similar_texts_score_higher() -> None:
    provider = MockEmbeddingProvider()
    v_cat = provider.embed("I love my cat")
    v_kitten = provider.embed("I love my kitten")
    v_finance = provider.embed("quarterly earnings report revenue")

    sim_related = _cosine_sim(v_cat, v_kitten)
    sim_unrelated = _cosine_sim(v_cat, v_finance)
    assert sim_related > sim_unrelated


def test_mock_embed_is_normalized() -> None:
    provider = MockEmbeddingProvider()
    vec = provider.embed("some text here")
    magnitude = math.sqrt(sum(v * v for v in vec))
    assert abs(magnitude - 1.0) < 1e-6


def test_mock_embed_batch() -> None:
    provider = MockEmbeddingProvider()
    texts = ["hello", "world"]
    batch = provider.embed_batch(texts)
    assert len(batch) == 2
    assert batch[0] == provider.embed("hello")
    assert batch[1] == provider.embed("world")


# ---------------------------------------------------------------------------
# OpenAICompatibleEmbeddingProvider
# ---------------------------------------------------------------------------


def _embedding_response(vectors: list[list[float]]) -> dict[str, object]:
    return {
        "data": [
            {"index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
    }


def test_openai_embed_single() -> None:
    fake_vec = [0.1, 0.2, 0.3]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-test"
        assert body["input"] == ["hello"]
        return httpx.Response(
            200, json=_embedding_response([fake_vec]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="text-embedding-test",
        client=client,
    )
    result = provider.embed("hello")
    assert result == fake_vec


def test_openai_embed_batch() -> None:
    vecs = [[0.1, 0.2], [0.3, 0.4]]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert len(body["input"]) == 2
        return httpx.Response(
            200, json=_embedding_response(vecs),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="m",
        client=client,
    )
    result = provider.embed_batch(["a", "b"])
    assert result == vecs


def test_openai_embed_passes_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["dimensions"] == 256
        return httpx.Response(
            200, json=_embedding_response([[0.0] * 256]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="m",
        dimensions=256,
        client=client,
    )
    provider.embed("test")


def test_openai_embed_passes_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(
            200, json=_embedding_response([[0.0]]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="m",
        api_key="sk-test",
        client=client,
    )
    provider.embed("test")


def test_openai_embed_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="m",
        client=client,
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.embed("test")


def test_openai_embed_raises_on_mismatched_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_embedding_response([[0.1], [0.2]]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://example.com/v1",
        model="m",
        client=client,
    )
    with pytest.raises(ValueError, match="Expected 1 embeddings"):
        provider.embed("single input")
