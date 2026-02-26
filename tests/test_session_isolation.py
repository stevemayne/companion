from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.main import create_app


def _seed_payload(name: str) -> dict[str, object]:
    return {
        "seed": {
            "companion_name": name,
            "backstory": f"{name} backstory",
            "character_traits": ["calm"],
            "goals": ["support"],
            "relationship_setup": "Companion"
        }
    }


def test_seed_context_isolated_between_sessions() -> None:
    client = TestClient(create_app())
    session_a = str(uuid4())
    session_b = str(uuid4())

    client.post(f"/v1/sessions/{session_a}/seed", json=_seed_payload("Ari"))
    client.post(f"/v1/sessions/{session_b}/seed", json=_seed_payload("Nova"))

    res_a = client.post(
        "/v1/chat",
        json={"chat_session_id": session_a, "message": "hello"},
    )
    res_b = client.post(
        "/v1/chat",
        json={"chat_session_id": session_b, "message": "hello"},
    )

    assert res_a.status_code == 200
    assert res_b.status_code == 200
    assert "You are Ari" in res_a.json()["assistant_message"]["content"]
    assert "You are Nova" in res_b.json()["assistant_message"]["content"]
    assert "You are Nova" not in res_a.json()["assistant_message"]["content"]


def test_monologue_isolated_between_sessions() -> None:
    app = create_app()
    client = TestClient(app)
    session_a = str(uuid4())
    session_b = str(uuid4())

    client.post(
        "/v1/chat",
        json={"chat_session_id": session_a, "message": "I feel anxious about dinner"},
    )
    client.post(
        "/v1/chat",
        json={"chat_session_id": session_b, "message": "I feel great today"},
    )

    mono_a = app.state.container.monologue_store.get(chat_session_id=UUID(session_a))
    mono_b = app.state.container.monologue_store.get(chat_session_id=UUID(session_b))

    assert mono_a is not None
    assert mono_b is not None
    assert "anxious" in mono_a.internal_monologue
    assert "positive" in mono_b.internal_monologue
