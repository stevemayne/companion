from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _seed_payload(name: str) -> dict[str, object]:
    return {
        "seed": {
            "companion_name": name,
            "backstory": "Backstory",
            "character_traits": ["warm"],
            "goals": ["trust"],
            "relationship_setup": "Companion"
        },
        "notes": "seed"
    }


def test_debug_tracing_disabled_returns_403() -> None:
    app = create_app(Settings(debug_tracing=False, inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    response = client.get(f"/v1/debug/{session_id}")

    assert response.status_code == 403

    client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "hello"},
    )
    assert app.state.container.debug_store.list_traces(chat_session_id=UUID(session_id)) == []


def test_debug_trace_contains_retrieval_and_writes() -> None:
    app = create_app(Settings(debug_tracing=True, inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload("Ari"))
    client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I feel anxious about Sarah"},
    )

    debug = client.get(f"/v1/debug/{session_id}")
    assert debug.status_code == 200
    payload = debug.json()
    assert payload["count"] >= 1

    trace = payload["traces"][-1]
    assert trace["chat_session_id"] == session_id
    assert "preprocess" in trace["turn_trace"]
    assert "retrieval" in trace["turn_trace"]
    assert "writes" in trace["turn_trace"]
    assert "characters" in trace["turn_trace"]
    assert "semantic_upserts" in trace["turn_trace"]["writes"]


def test_debug_traces_are_session_isolated() -> None:
    app = create_app(Settings(debug_tracing=True, inference_provider="mock"))
    client = TestClient(app)
    session_a = str(uuid4())
    session_b = str(uuid4())

    client.post(
        "/v1/chat",
        json={"chat_session_id": session_a, "message": "session a message"},
    )
    client.post(
        "/v1/chat",
        json={"chat_session_id": session_b, "message": "session b message"},
    )

    debug_a = client.get(f"/v1/debug/{session_a}").json()["traces"]
    debug_b = client.get(f"/v1/debug/{session_b}").json()["traces"]

    assert all(item["chat_session_id"] == session_a for item in debug_a)
    assert all(item["chat_session_id"] == session_b for item in debug_b)
    assert len(debug_a) >= 1
    assert len(debug_b) >= 1


def test_question_word_is_not_extracted_as_entity() -> None:
    app = create_app(Settings(debug_tracing=True, inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "What are you wearing?"},
    )

    trace = client.get(f"/v1/debug/{session_id}").json()["traces"][-1]
    assert trace["turn_trace"]["preprocess"]["intent"] == "question"
    assert trace["turn_trace"]["preprocess"]["entities"] == []
