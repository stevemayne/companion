"""Consolidation loop — periodically reviews recent conversation and refines memories.

For each existing memory the consolidator decides:
- **reinforce**: bump importance (the fact was mentioned again)
- **contradict**: mark superseded and create a replacement memory
- **ignore**: no change needed

New durable facts discovered during consolidation are stored with kind=REFLECTIVE.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol
from uuid import UUID

from app.schemas import MemoryItem, MemoryStatus, Message

logger = logging.getLogger(__name__)


class ConsolidationModelProvider(Protocol):
    def generate(
        self, *, chat_session_id: UUID, messages: list[dict[str, str]]
    ) -> str: ...


class ConsolidationVectorStore(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]: ...

    def list_memories(self, *, chat_session_id: UUID) -> list[MemoryItem]: ...


class ConsolidationEpisodicStore(Protocol):
    def get_recent_messages(
        self, *, chat_session_id: UUID, limit: int = 50
    ) -> list[Message]: ...


# ── result types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReinforcedMemory:
    memory_id: UUID
    old_importance: float
    new_importance: float


@dataclass(frozen=True)
class SupersededMemory:
    memory_id: UUID
    reason: str
    replacement_text: str | None = None


@dataclass(frozen=True)
class NewFact:
    text: str
    importance: float


@dataclass(frozen=True)
class ConsolidationResult:
    reinforced: list[ReinforcedMemory] = field(default_factory=list)
    superseded: list[SupersededMemory] = field(default_factory=list)
    new_facts: list[NewFact] = field(default_factory=list)
    provider: str = "heuristic"
    latency_ms: float = 0.0


# ── heuristic consolidation ──────────────────────────────────────────


def _token_overlap(a: str, b: str) -> float:
    """Fraction of tokens in *a* that also appear in *b*."""
    tokens_a = {t.lower() for t in a.split()}
    tokens_b = {t.lower() for t in b.split()}
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _heuristic_consolidate(
    *,
    messages: list[Message],
    existing_memories: list[MemoryItem],
) -> ConsolidationResult:
    """Cheap consolidation: if recent conversation text overlaps significantly
    with an existing memory, bump its importance (reinforce)."""
    start = perf_counter()
    conversation_text = " ".join(m.content for m in messages if m.role == "user")
    reinforced: list[ReinforcedMemory] = []

    for mem in existing_memories:
        if mem.status != MemoryStatus.ACTIVE:
            continue
        overlap = _token_overlap(mem.content, conversation_text)
        if overlap >= 0.5:
            bump = min(1.0, mem.importance + 0.1)
            if bump > mem.importance:
                reinforced.append(ReinforcedMemory(
                    memory_id=mem.memory_id,
                    old_importance=mem.importance,
                    new_importance=bump,
                ))

    return ConsolidationResult(
        reinforced=reinforced,
        provider="heuristic",
        latency_ms=(perf_counter() - start) * 1000,
    )


# ── LLM consolidation ────────────────────────────────────────────────


_CONSOLIDATION_PROMPT = """\
You are a memory consolidation system. Given recent conversation turns and \
the user's existing memories, decide what to do with each memory and whether \
any new durable facts should be extracted.

## Existing memories
{memories_block}

## Recent conversation
{conversation_block}

## Instructions
For each existing memory, choose ONE action:
- **reinforce**: The conversation confirms or repeats this fact. \
Return the memory_id and a new importance (current + 0.05 to 0.15, max 1.0).
- **supersede**: The conversation contradicts or updates this fact. \
Return the memory_id, a reason, and optional replacement_text.
- **ignore**: No relevant mention. Omit from output.

Also extract any NEW durable facts about the USER that are not already \
captured in existing memories. Each new fact needs text and importance (0.0-1.0).

CRITICAL: Only extract facts about the USER based on what they said. \
NEVER extract facts about the assistant/companion — their opinions, \
feelings, preferences, or actions are NOT user facts. If the assistant \
said "I love hiking", that is NOT a fact about the user.

## Output format
Return strict JSON:
{{
  "reinforce": [
    {{"memory_id": "...", "new_importance": 0.75}}
  ],
  "supersede": [
    {{"memory_id": "...", "reason": "...", "replacement_text": "..."}}
  ],
  "new_facts": [
    {{"text": "...", "importance": 0.6}}
  ]
}}

Omit empty arrays. Return only JSON, no explanation.\
"""


def _build_consolidation_prompt(
    *,
    messages: list[Message],
    existing_memories: list[MemoryItem],
) -> str:
    mem_lines: list[str] = []
    for mem in existing_memories:
        if mem.status != MemoryStatus.ACTIVE:
            continue
        mem_lines.append(
            f"- id={mem.memory_id} importance={mem.importance:.2f}: {mem.content}"
        )
    memories_block = "\n".join(mem_lines) if mem_lines else "(none)"

    conv_lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            conv_lines.append(f"USER: {msg.content}")
    conversation_block = "\n".join(conv_lines) if conv_lines else "(none)"

    return _CONSOLIDATION_PROMPT.format(
        memories_block=memories_block,
        conversation_block=conversation_block,
    )


def _parse_consolidation_response(
    raw: str,
    *,
    existing_memories: list[MemoryItem],
) -> ConsolidationResult:
    """Parse LLM JSON response into a ConsolidationResult."""
    candidate = raw.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    data = json.loads(candidate)
    mem_by_id = {str(m.memory_id): m for m in existing_memories}

    reinforced: list[ReinforcedMemory] = []
    for item in data.get("reinforce", []):
        mid = str(item.get("memory_id", ""))
        if mid not in mem_by_id:
            continue
        new_imp = max(0.0, min(1.0, float(item.get("new_importance", 0.5))))
        reinforced.append(ReinforcedMemory(
            memory_id=mem_by_id[mid].memory_id,
            old_importance=mem_by_id[mid].importance,
            new_importance=new_imp,
        ))

    superseded: list[SupersededMemory] = []
    for item in data.get("supersede", []):
        mid = str(item.get("memory_id", ""))
        if mid not in mem_by_id:
            continue
        superseded.append(SupersededMemory(
            memory_id=mem_by_id[mid].memory_id,
            reason=str(item.get("reason", "")),
            replacement_text=item.get("replacement_text"),
        ))

    new_facts: list[NewFact] = []
    for item in data.get("new_facts", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
        new_facts.append(NewFact(text=text, importance=importance))

    return ConsolidationResult(
        reinforced=reinforced,
        superseded=superseded,
        new_facts=new_facts,
        provider="llm",
    )


# ── main agent ────────────────────────────────────────────────────────


class ConsolidationAgent:
    def __init__(
        self,
        *,
        provider: ConsolidationModelProvider | None = None,
    ) -> None:
        self._provider = provider

    def consolidate_session(
        self,
        *,
        chat_session_id: UUID,
        messages: list[Message],
        existing_memories: list[MemoryItem],
    ) -> ConsolidationResult:
        if not messages:
            return ConsolidationResult()

        if self._provider is None or not existing_memories:
            return _heuristic_consolidate(
                messages=messages,
                existing_memories=existing_memories,
            )

        start = perf_counter()
        prompt = _build_consolidation_prompt(
            messages=messages,
            existing_memories=existing_memories,
        )
        try:
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[{"role": "user", "content": prompt}],
            )
            result = _parse_consolidation_response(
                raw, existing_memories=existing_memories,
            )
            return ConsolidationResult(
                reinforced=result.reinforced,
                superseded=result.superseded,
                new_facts=result.new_facts,
                provider="llm",
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception:
            logger.warning(
                "LLM consolidation failed for session %s, falling back to heuristic",
                chat_session_id,
                exc_info=True,
            )
            return _heuristic_consolidate(
                messages=messages,
                existing_memories=existing_memories,
            )
