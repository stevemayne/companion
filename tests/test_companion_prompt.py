from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _seed_payload() -> dict[str, object]:
    return {
        "seed": {
            "companion_name": "Ari",
            "backstory": "Ari is a long-term companion who knows the user well.",
            "character_traits": ["warm", "playful"],
            "goals": ["build trust", "maintain continuity"],
            "relationship_setup": "Close confidant and emotional support partner.",
        },
        "notes": "companion prompt test",
    }


def test_companion_prompt_includes_seeded_identity_and_relational_context_every_turn() -> None:
    client = TestClient(create_app(Settings(inference_provider="mock")))
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload())

    first = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "Hello"},
    )
    second = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "What's your name?"},
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_content = first.json()["assistant_message"]["content"]
    second_content = second.json()["assistant_message"]["content"]

    assert "You are Ari" in first_content
    assert "Relationship setup: Close confidant and emotional support partner." in first_content
    assert "Primary goals in this conversation: build trust, maintain continuity." in first_content

    assert "You are Ari" in second_content
    assert "Relationship setup: Close confidant and emotional support partner." in second_content
    assert "Primary goals in this conversation: build trust, maintain continuity." in second_content


def test_seeded_identity_rewrites_assistant_name_fallback() -> None:
    app = create_app(Settings(inference_provider="mock"))
    client = TestClient(app)
    session_id = str(uuid4())

    client.post(f"/v1/sessions/{session_id}/seed", json=_seed_payload())

    class FallbackNameProvider:
        def generate(self, *, chat_session_id: object, prompt: str) -> str:
            del chat_session_id, prompt
            return "My name is Assistant."

    app.state.container.orchestrator.model_provider = FallbackNameProvider()

    response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "What's your name?"},
    )

    assert response.status_code == 200
    content = response.json()["assistant_message"]["content"]
    assert "my name is ari" in content.lower()
    assert "my name is assistant" not in content.lower()
