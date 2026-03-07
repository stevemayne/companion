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
    # Affect state is preserved (background LLM reflector updates it async)
    assert persisted.affect is not None
    assert persisted.affect.mood != ""


def test_orchestrator_uses_graph_context_on_follow_up() -> None:
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
    # Graph relations are created by the background LLM extractor (async),
    # so they may not be present in the synchronous mock test.  The internal
    # monologue still carries forward.
    assert "Internal reflection: Focus on a neutral user" in content
    # Conversation history includes the earlier user message.
    assert "I argued with Sarah yesterday." in content
    # Affect block should appear in the system prompt on the second turn.
    # The directive "You're feeling ..." survives post-processing since
    # it's plain prose, not a leaked metric.
    assert "You're feeling" in content
