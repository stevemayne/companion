from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas import (
    CompanionAffect,
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    Message,
    SessionSeedContext,
)


class ChatRequest(BaseModel):
    chat_session_id: UUID
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    chat_session_id: UUID
    assistant_message: Message
    affect: CompanionAffect | None = None
    idempotency_replay: bool = False
    seed_version: int | None = None


class SeedContextUpsertRequest(BaseModel):
    seed: CompanionSeed
    user_description: str | None = None
    notes: str | None = None


class MemoryResponse(BaseModel):
    chat_session_id: UUID
    messages: list[Message]
    seed_context: SessionSeedContext | None = None


class KnowledgeResponse(BaseModel):
    chat_session_id: UUID
    facts: list[MemoryItem]
    graph: list[GraphRelation]
    monologue: str | None = None
    affect: CompanionAffect | None = None


class SessionSummary(BaseModel):
    chat_session_id: UUID
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    companion_name: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody
