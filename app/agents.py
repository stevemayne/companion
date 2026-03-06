from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from uuid import UUID

from app.analysis import ExtractionOutcome
from app.consolidation import ConsolidationAgent
from app.debug_trace import DebugTraceStore, build_trace_base
from app.schemas import (
    CompanionAffect,
    GraphRelation,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Message,
    MonologueState,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    extraction_jobs: int = 0
    reflector_jobs: int = 0
    consolidation_jobs: int = 0
    failures: int = 0


class EpisodicStore(Protocol):
    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]: ...


class VectorStore(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...

    def list_memories(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> list[MemoryItem]: ...

    def update_memory(
        self, *, memory_id: UUID, importance: float | None = None,
        status: MemoryStatus | None = None,
    ) -> None: ...


class GraphStore(Protocol):
    def upsert_relation(self, relation: GraphRelation) -> None: ...


class MonologueStore(Protocol):
    def get(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> MonologueState | None: ...

    def upsert(self, state: MonologueState) -> MonologueState: ...


class FactExtractor(Protocol):
    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
    ) -> ExtractionOutcome: ...


class AffectRefiner(Protocol):
    def generate(
        self, *, chat_session_id: UUID, messages: list[dict[str, str]]
    ) -> str: ...


class BackgroundAgentDispatcher:
    def __init__(
        self,
        *,
        episodic_store: EpisodicStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        monologue_store: MonologueStore,
        fact_extractor: FactExtractor,
        consolidation_agent: ConsolidationAgent | None = None,
        consolidation_interval: int = 10,
        consolidation_message_window: int = 20,
        affect_refiner: AffectRefiner | None = None,
        debug_store: DebugTraceStore,
        enabled: bool = True,
    ) -> None:
        self._episodic_store = episodic_store
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._monologue_store = monologue_store
        self._fact_extractor = fact_extractor
        self._consolidation_agent = consolidation_agent
        self._consolidation_interval = consolidation_interval
        self._consolidation_message_window = consolidation_message_window
        self._affect_refiner = affect_refiner
        self._debug_store = debug_store
        self._enabled = enabled
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="aether-agents",
        )
        self._lock = Lock()
        self._metrics: dict[UUID, AgentMetrics] = {}
        self._turn_counts: dict[UUID, int] = {}
        self._futures: set[Future[None]] = set()

    def enqueue_turn(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
        companion_id: UUID | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._submit(
            self._run_extraction,
            chat_session_id,
            user_message,
            assistant_message,
            companion_name,
            companion_id,
        )
        self._submit(self._run_reflector, chat_session_id, companion_id)

        if self._consolidation_agent is not None:
            with self._lock:
                count = self._turn_counts.get(chat_session_id, 0) + 1
                self._turn_counts[chat_session_id] = count
            if count % self._consolidation_interval == 0:
                self._submit(self._run_consolidation, chat_session_id, companion_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def get_metrics(self, *, chat_session_id: UUID) -> AgentMetrics:
        with self._lock:
            return self._metrics.get(chat_session_id, AgentMetrics())

    def _submit(self, fn: Callable[..., None], *args: object) -> None:
        future: Future[None] = self._executor.submit(fn, *args)
        with self._lock:
            self._futures.add(future)

        def _cleanup(done: Future[None]) -> None:
            with self._lock:
                self._futures.discard(done)

        future.add_done_callback(_cleanup)

    def _run_extraction(
        self,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None,
        companion_id: UUID | None = None,
    ) -> None:
        try:
            outcome = self._fact_extractor.extract(
                chat_session_id=chat_session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                companion_name=companion_name,
            )
            for fact in outcome.facts:
                self._vector_store.upsert_memory(
                    MemoryItem(
                        chat_session_id=chat_session_id,
                        companion_id=companion_id,
                        kind=MemoryKind.SEMANTIC,
                        content=fact.text,
                        importance=fact.importance,
                    )
                )

            for em in outcome.entities:
                if em.relationship:
                    self._graph_store.upsert_relation(
                        GraphRelation(
                            chat_session_id=chat_session_id,
                            companion_id=companion_id,
                            source=em.owner,
                            relation=f"HAS_{em.relationship.upper().replace(' ', '_')}",
                            target=em.name,
                        )
                    )
                    for alias in em.aliases:
                        self._graph_store.upsert_relation(
                            GraphRelation(
                                chat_session_id=chat_session_id,
                                companion_id=companion_id,
                                source=em.name,
                                relation="ALSO_KNOWN_AS",
                                target=alias,
                            )
                        )

            for fact in outcome.companion_facts:
                self._vector_store.upsert_memory(
                    MemoryItem(
                        chat_session_id=chat_session_id,
                        companion_id=companion_id,
                        kind=MemoryKind.COMPANION,
                        content=fact.text,
                        importance=fact.importance,
                    )
                )

            trace = build_trace_base(chat_session_id=chat_session_id)
            trace.update({
                "agent": "extraction",
                "provider": outcome.used_provider,
                "fallback_reason": outcome.fallback_reason,
                "latency_ms": round(outcome.latency_ms, 2),
                "facts": [f.text for f in outcome.facts],
                "structured_facts": [
                    {
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "object": f.object,
                        "text": f.text,
                    }
                    for f in outcome.facts
                ],
                "entities": [
                    {
                        "name": e.name, "relationship": e.relationship,
                        "owner": e.owner, "entity_type": e.entity_type,
                        "aliases": e.aliases,
                    }
                    for e in outcome.entities
                ],
                "companion_facts": [f.text for f in outcome.companion_facts],
            })
            self._debug_store.add_trace(chat_session_id=chat_session_id, trace=trace)

            self._increment(chat_session_id=chat_session_id, extraction_jobs=1)
        except Exception:
            self._increment(chat_session_id=chat_session_id, failures=1)

    def _run_reflector(
        self, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> None:
        try:
            current = self._monologue_store.get(
                chat_session_id=chat_session_id,
                companion_id=companion_id,
            )
            current_affect = (
                current.affect if current is not None else CompanionAffect()
            )
            current_monologue = (
                current.internal_monologue if current is not None else ""
            )
            current_user_state = (
                current.user_state if current is not None else []
            )

            refined_affect = current_affect
            refined_user_state = current_user_state

            if self._affect_refiner is not None:
                recent = self._episodic_store.get_recent_messages(
                    chat_session_id=chat_session_id,
                    limit=6,
                )
                if recent:
                    refined_affect, refined_user_state = (
                        self._llm_refine_state(
                            chat_session_id=chat_session_id,
                            recent_messages=recent,
                            current_affect=current_affect,
                            current_user_state=current_user_state,
                        )
                    )

            self._monologue_store.upsert(
                MonologueState(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                    internal_monologue=current_monologue,
                    affect=refined_affect,
                    user_state=refined_user_state,
                )
            )
            self._increment(
                chat_session_id=chat_session_id, reflector_jobs=1,
            )
        except Exception:
            self._increment(
                chat_session_id=chat_session_id, failures=1,
            )

    def _run_consolidation(
        self, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> None:
        assert self._consolidation_agent is not None
        try:
            messages = self._episodic_store.get_recent_messages(
                chat_session_id=chat_session_id,
                limit=self._consolidation_message_window,
            )
            existing = [
                m for m in self._vector_store.list_memories(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                )
                if m.kind != MemoryKind.COMPANION
            ]
            result = self._consolidation_agent.consolidate_session(
                chat_session_id=chat_session_id,
                messages=messages,
                existing_memories=existing,
            )

            for r in result.reinforced:
                self._vector_store.update_memory(
                    memory_id=r.memory_id,
                    importance=r.new_importance,
                )
            for s in result.superseded:
                self._vector_store.update_memory(
                    memory_id=s.memory_id,
                    status=MemoryStatus.SUPERSEDED,
                )
                if s.replacement_text:
                    self._vector_store.upsert_memory(MemoryItem(
                        chat_session_id=chat_session_id,
                        companion_id=companion_id,
                        kind=MemoryKind.REFLECTIVE,
                        content=s.replacement_text,
                        importance=0.6,
                    ))
            for nf in result.new_facts:
                self._vector_store.upsert_memory(MemoryItem(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                    kind=MemoryKind.REFLECTIVE,
                    content=nf.text,
                    importance=nf.importance,
                ))

            trace = build_trace_base(chat_session_id=chat_session_id)
            trace.update({
                "agent": "consolidation",
                "provider": result.provider,
                "latency_ms": round(result.latency_ms, 2),
                "reinforced": len(result.reinforced),
                "superseded": len(result.superseded),
                "new_facts": len(result.new_facts),
            })
            self._debug_store.add_trace(
                chat_session_id=chat_session_id, trace=trace,
            )
            self._increment(
                chat_session_id=chat_session_id, consolidation_jobs=1,
            )
        except Exception:
            logger.warning(
                "Consolidation failed for session %s",
                chat_session_id, exc_info=True,
            )
            self._increment(chat_session_id=chat_session_id, failures=1)

    def _llm_refine_state(
        self,
        *,
        chat_session_id: UUID,
        recent_messages: list[Message],
        current_affect: CompanionAffect,
        current_user_state: list[str],
    ) -> tuple[CompanionAffect, list[str]]:
        """Call the analysis LLM to refine affect and extract user state."""
        conversation_excerpt = "\n".join(
            f"{msg.role.upper()}: {msg.content}"
            for msg in recent_messages[-6:]
        )
        current_json = current_affect.model_dump_json()
        state_json = json.dumps(current_user_state)
        prompt = (
            "You are an affect-state analyser for a companion AI. "
            "Given the recent conversation and the companion's current "
            "internal state, return a revised affect state and the "
            "user's current physical state as strict JSON.\n\n"
            "## Current companion affect\n"
            f"{current_json}\n\n"
            "## Current user state\n"
            f"{state_json}\n\n"
            "## Recent conversation\n"
            f"{conversation_excerpt}\n\n"
            "## Output format\n"
            "Return a single JSON object with exactly these keys:\n"
            "  mood (string — one of: curious, wary, anxious, amused, "
            "frustrated, concerned, excited, hurt, withdrawn, fond, "
            "playful)\n"
            "  valence (float -1.0 to 1.0)\n"
            "  arousal (float 0.0 to 1.0)\n"
            "  comfort_level (float 0 to 10)\n"
            "  trust (float 0 to 10)\n"
            "  attraction (float 0 to 10)\n"
            "  engagement (float 0 to 10)\n"
            "  shyness (float 0 to 10, high=reserved/hesitant)\n"
            "  patience (float 0 to 10)\n"
            "  curiosity (float 0 to 10)\n"
            "  vulnerability (float 0 to 10, willingness to share "
            "deeper feelings)\n"
            "  recent_triggers (list of up to 3 short strings explaining "
            "what changed)\n"
            "  user_state (list of up to 8 short strings describing "
            "the user's current physical state, appearance, clothing, "
            "actions, or location — extracted from the USER's messages "
            "only. Merge with the existing state: keep still-relevant "
            "entries, drop outdated ones, add new observations.)\n\n"
            "Adjust affect values modestly — this is a refinement, "
            "not a reset. Return only JSON, no explanation."
        )
        assert self._affect_refiner is not None
        raw = self._affect_refiner.generate(
            chat_session_id=chat_session_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_state_response(
            raw,
            fallback_affect=current_affect,
            fallback_user_state=current_user_state,
        )

    def _increment(
        self,
        *,
        chat_session_id: UUID,
        extraction_jobs: int = 0,
        reflector_jobs: int = 0,
        consolidation_jobs: int = 0,
        failures: int = 0,
    ) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(chat_session_id, AgentMetrics())
            metrics.extraction_jobs += extraction_jobs
            metrics.reflector_jobs += reflector_jobs
            metrics.consolidation_jobs += consolidation_jobs
            metrics.failures += failures


def _parse_affect_response(
    raw: str, *, fallback: CompanionAffect,
) -> CompanionAffect:
    """Parse LLM-returned affect JSON, returning fallback on any error."""
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
    try:
        data = json.loads(candidate)
        return CompanionAffect.model_validate(data)
    except Exception:
        return fallback


def _parse_state_response(
    raw: str,
    *,
    fallback_affect: CompanionAffect,
    fallback_user_state: list[str],
) -> tuple[CompanionAffect, list[str]]:
    """Parse LLM state response containing affect + user_state."""
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
    try:
        data = json.loads(candidate)
    except Exception:
        return fallback_affect, fallback_user_state

    # Extract user_state before validating affect (which would reject
    # the extra key).
    raw_user_state = data.pop("user_state", None)
    user_state = fallback_user_state
    if isinstance(raw_user_state, list):
        user_state = [
            str(s).strip() for s in raw_user_state
            if isinstance(s, str) and str(s).strip()
        ][:8]

    try:
        affect = CompanionAffect.model_validate(data)
    except Exception:
        affect = fallback_affect

    return affect, user_state

