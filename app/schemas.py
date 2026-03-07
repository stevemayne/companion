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
    speaker_id: UUID | None = None
    speaker_name: str | None = None
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    REFLECTIVE = "reflective"
    COMPANION = "companion"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class MemoryItem(BaseModel):
    chat_session_id: UUID
    companion_id: UUID | None = None
    memory_id: UUID = Field(default_factory=uuid4)
    kind: MemoryKind
    content: str = Field(min_length=1)
    score: float | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    last_accessed: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    source_turn_id: UUID | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Entity(BaseModel):
    chat_session_id: UUID
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    summary: str | None = None


class GraphRelation(BaseModel):
    chat_session_id: UUID
    companion_id: UUID | None = None
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CompanionAffect(BaseModel):
    mood: str = Field(default="curious")
    valence: float = Field(default=0.1, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.3, ge=0.0, le=1.0)
    dominance: float = Field(default=0.4, ge=0.0, le=1.0)
    trust: float = Field(default=3.0, ge=0.0, le=10.0)
    closeness: float = Field(default=3.0, ge=0.0, le=10.0)
    engagement: float = Field(default=5.0, ge=0.0, le=10.0)
    recent_triggers: list[str] = Field(default_factory=list)


class CharacterState(BaseModel):
    """Physical/locational state of a character as perceived by the observer."""

    clothing: str | None = None
    location: str | None = None
    activity: str | None = None
    position: str | None = None
    appearance: list[str] = Field(default_factory=list)
    mood_apparent: str | None = None


class WorldState(BaseModel):
    """A companion's subjective perception of the current scene."""

    self_state: CharacterState = Field(default_factory=CharacterState)
    user_state: CharacterState = Field(default_factory=CharacterState)
    other_characters: dict[str, CharacterState] = Field(default_factory=dict)
    environment: str | None = None
    time_of_day: str | None = None
    recent_events: list[str] = Field(default_factory=list)


class MonologueState(BaseModel):
    chat_session_id: UUID
    companion_id: UUID | None = None
    internal_monologue: str = Field(default="")
    affect: CompanionAffect = Field(default_factory=CompanionAffect)
    world: WorldState = Field(default_factory=WorldState)
    user_state: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CompanionSeed(BaseModel):
    companion_name: str = Field(min_length=1)
    user_name: str | None = None
    backstory: str = Field(min_length=1)
    character_traits: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    relationship_setup: str = Field(min_length=1)


class SessionSeedContext(BaseModel):
    chat_session_id: UUID
    companion_id: UUID = Field(default_factory=uuid4)
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
