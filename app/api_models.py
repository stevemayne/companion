from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas import CompanionSeed, Message, SessionSeedContext


class ChatRequest(BaseModel):
    chat_session_id: UUID
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    chat_session_id: UUID
    assistant_message: Message
    idempotency_replay: bool = False
    seed_version: int | None = None


class SeedContextUpsertRequest(BaseModel):
    seed: CompanionSeed
    notes: str | None = None


class MemoryResponse(BaseModel):
    chat_session_id: UUID
    messages: list[Message]
    seed_context: SessionSeedContext | None = None


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody
