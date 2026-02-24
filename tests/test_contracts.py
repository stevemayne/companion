from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import CompanionSeed, Entity, Message, SessionSeedContext


def test_message_requires_chat_session_id() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content="hello")


def test_entity_requires_chat_session_id() -> None:
    with pytest.raises(ValidationError):
        Entity(name="Sarah", entity_type="person")


def test_session_seed_context_supports_versioned_seed_data() -> None:
    session_id = uuid4()
    context = SessionSeedContext(
        chat_session_id=session_id,
        version=2,
        seed=CompanionSeed(
            companion_name="Ari",
            backstory="Ari is a thoughtful long-term companion.",
            character_traits=["curious", "calm"],
            goals=["build trust", "stay consistent"],
            relationship_setup="Close friend and reflective coach.",
        ),
    )

    assert context.chat_session_id == session_id
    assert context.version == 2
    assert context.seed.companion_name == "Ari"
