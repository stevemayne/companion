from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    chat_session_id: UUID
    message_id: UUID = Field(default_factory=uuid4)
    role: Role
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    REFLECTIVE = "reflective"


class MemoryItem(BaseModel):
    chat_session_id: UUID
    memory_id: UUID = Field(default_factory=uuid4)
    kind: MemoryKind
    content: str = Field(min_length=1)
    score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Entity(BaseModel):
    chat_session_id: UUID
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    summary: str | None = None


class GraphRelation(BaseModel):
    chat_session_id: UUID
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MonologueState(BaseModel):
    chat_session_id: UUID
    internal_monologue: str = Field(default="")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CompanionSeed(BaseModel):
    companion_name: str = Field(min_length=1)
    backstory: str = Field(min_length=1)
    character_traits: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    relationship_setup: str = Field(min_length=1)


class SessionSeedContext(BaseModel):
    chat_session_id: UUID
    version: int = Field(default=1, ge=1)
    seed: CompanionSeed
    user_description: str | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionActivity(BaseModel):
    chat_session_id: UUID
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)


class PreprocessResult(BaseModel):
    intent: str
    emotion: str
    entities: list[str] = Field(default_factory=list)
