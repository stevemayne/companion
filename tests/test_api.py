from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app


def _seed_payload(name: str = "Ari") -> dict[str, object]:
    return {
        "seed": {
            "companion_name": name,
            "backstory": "Ari is a thoughtful long-term companion.",
            "character_traits": ["curious", "calm"],
            "goals": ["build trust"],
            "relationship_setup": "Close friend and reflective coach.",
        },
        "notes": "Initial seed",
    }


def test_chat_and_memory_are_session_scoped() -> None:
    client = TestClient(create_app())
    session_a = str(uuid4())
    session_b = str(uuid4())

    response_a = client.post("/v1/chat", json={"chat_session_id": session_a, "message": "hello"})
    response_b = client.post("/v1/chat", json={"chat_session_id": session_b, "message": "hi"})

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    memory_a = client.get(f"/v1/memory/{session_a}")
    memory_b = client.get(f"/v1/memory/{session_b}")

    assert memory_a.status_code == 200
    assert memory_b.status_code == 200
    assert len(memory_a.json()["messages"]) == 2
    assert len(memory_b.json()["messages"]) == 2
    assert all(item["chat_session_id"] == session_a for item in memory_a.json()["messages"])
    assert all(item["chat_session_id"] == session_b for item in memory_b.json()["messages"])


def test_chat_idempotency_key_replays_response() -> None:
    client = TestClient(create_app())
    session_id = str(uuid4())
    headers = {"Idempotency-Key": "abc-123"}
    payload = {"chat_session_id": session_id, "message": "Are you there?"}

    first = client.post("/v1/chat", json=payload, headers=headers)
    second = client.post("/v1/chat", json=payload, headers=headers)
    memory = client.get(f"/v1/memory/{session_id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["idempotency_replay"] is True
    assert (
        second.json()["assistant_message"]["message_id"]
        == first.json()["assistant_message"]["message_id"]
    )
    assert len(memory.json()["messages"]) == 2


def test_seed_lifecycle_and_versioning() -> None:
    client = TestClient(create_app())
    session_id = str(uuid4())

    created = client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload(name="Ari"))
    updated = client.put(f"/v1/sessions/{session_id}/seed", json=_seed_payload(name="Nova"))
    fetched = client.get(f"/v1/sessions/{session_id}/seed")

    assert created.status_code == 201
    assert created.json()["version"] == 1
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert fetched.status_code == 200
    assert fetched.json()["seed"]["companion_name"] == "Nova"


def test_seed_is_applied_from_first_turn() -> None:
    client = TestClient(create_app())
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload(name="Ari"))
    chat_response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I feel anxious"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["seed_version"] == 1
    assert "You are Ari" in chat_response.json()["assistant_message"]["content"]


def test_sessions_list_returns_recent_activity_with_seed_metadata() -> None:
    client = TestClient(create_app())
    session_a = str(uuid4())
    session_b = str(uuid4())

    client.post(f"/v1/sessions/{session_a}/seed", json=_seed_payload(name="Ari"))
    client.post("/v1/chat", json={"chat_session_id": session_b, "message": "hello from b"})
    client.post("/v1/chat", json={"chat_session_id": session_a, "message": "hello from a"})

    response = client.get("/v1/sessions?limit=10")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    ids = [item["chat_session_id"] for item in sessions]
    assert session_a in ids
    assert session_b in ids

    by_id = {item["chat_session_id"]: item for item in sessions}
    assert by_id[session_a]["companion_name"] == "Ari"
    assert by_id[session_a]["message_count"] >= 1
    assert by_id[session_b]["message_count"] >= 1
