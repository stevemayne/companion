from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.main import create_app


def _seed_payload() -> dict[str, object]:
    return {
        "seed": {
            "companion_name": "Ari",
            "backstory": "Ari is reflective and warm.",
            "character_traits": ["curious", "supportive"],
            "goals": ["build trust"],
            "relationship_setup": "Long-term companion.",
        }
    }


def test_orchestrator_persists_monologue_per_session() -> None:
    app = create_app()
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload())
    response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I'm nervous about dinner with Sarah."},
    )

    assert response.status_code == 200
    unrelated = app.state.container.monologue_store.get(chat_session_id=uuid4())
    assert unrelated is None

    persisted = app.state.container.monologue_store.get(
        chat_session_id=UUID(response.json()["chat_session_id"])
    )
    assert persisted is not None
    assert "anxious" in persisted.internal_monologue
    assert "Sarah" in persisted.internal_monologue


def test_orchestrator_uses_semantic_and_graph_context_on_follow_up() -> None:
    app = create_app()
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload())
    first = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I argued with Sarah yesterday."},
    )
    second = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I am nervous to see Sarah tonight."},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    content = second.json()["assistant_message"]["content"]
    # Context is now in the system message within the structured messages array.
    # The mock response from turn 1 may also appear in semantic memories (via
    # the background extraction agent), so we check for substring presence
    # rather than an exact prefix match.
    assert "Relevant memories:" in content
    assert "I argued with Sarah yesterday." in content
    assert "user-MENTIONED_IN_SESSION->Sarah" in content
    assert "Internal reflection: Focus on a neutral user" in content
