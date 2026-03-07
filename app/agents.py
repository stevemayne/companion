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
    CharacterState,
    CompanionAffect,
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Message,
    MonologueState,
    SessionSeedContext,
    WorldState,
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


class SeedContextStore(Protocol):
    def get(
        self, *, chat_session_id: UUID, companion_id: UUID | None = None,
    ) -> SessionSeedContext | None: ...


class FactExtractor(Protocol):
    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
        user_name: str | None = None,
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
        seed_store: SeedContextStore | None = None,
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
        self._seed_store = seed_store
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
        user_name: str | None = None,
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
            user_name,
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
        user_name: str | None = None,
    ) -> None:
        try:
            outcome = self._fact_extractor.extract(
                chat_session_id=chat_session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                companion_name=companion_name,
                user_name=user_name,
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
                            relation=em.relationship.upper().replace(' ', '_'),
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
            current_world = (
                current.world if current is not None else WorldState()
            )

            refined_affect = current_affect
            refined_world = current_world
            refined_monologue = current_monologue
            # On the very first reflector run the LLM needs to
            # establish baseline values from the seed context (e.g.
            # high trust/closeness for a long-term partner).  Skip
            # clamping so the LLM can set appropriate initial values
            # in one shot rather than ramping over many turns.
            is_bootstrap = current is None

            if self._affect_refiner is not None:
                recent = self._episodic_store.get_recent_messages(
                    chat_session_id=chat_session_id,
                    limit=6,
                )
                if recent:
                    seed: CompanionSeed | None = None
                    if self._seed_store is not None:
                        seed_context = self._seed_store.get(
                            chat_session_id=chat_session_id,
                            companion_id=companion_id,
                        )
                        if seed_context is not None:
                            seed = seed_context.seed
                    refined_affect, refined_world, refined_monologue = (
                        self._llm_refine_state(
                            chat_session_id=chat_session_id,
                            recent_messages=recent,
                            current_affect=current_affect,
                            current_world=current_world,
                            seed=seed,
                            skip_clamp=is_bootstrap,
                        )
                    )

            self._monologue_store.upsert(
                MonologueState(
                    chat_session_id=chat_session_id,
                    companion_id=companion_id,
                    internal_monologue=refined_monologue,
                    affect=refined_affect,
                    world=refined_world,
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
        current_world: WorldState,
        seed: CompanionSeed | None = None,
        skip_clamp: bool = False,
    ) -> tuple[CompanionAffect, WorldState, str]:
        """Call the analysis LLM to refine affect, world state, and monologue."""
        user_label = (seed.user_name if seed and seed.user_name else "USER")
        companion_label = (
            seed.companion_name if seed else "ASSISTANT"
        )

        def _role_label(role: str) -> str:
            return user_label if role == "user" else companion_label

        conversation_excerpt = "\n".join(
            f"{_role_label(msg.role)}: {msg.content}"
            for msg in recent_messages[-6:]
        )
        current_json = current_affect.model_dump_json()
        world_json = current_world.model_dump_json()

        relationship_block = ""
        if seed is not None:
            relationship_block = (
                "## Companion identity & relationship context\n"
                f"Name: {seed.companion_name}\n"
                f"Relationship: {seed.relationship_setup}\n"
                f"Backstory: {seed.backstory}\n"
                "Use this context to calibrate appropriate affect values. "
                "A long-term romantic partner should have trust 7-9 and "
                "closeness 7-9. A new acquaintance would be 1-3. Set values "
                "that match the established relationship, not just what "
                "happened in the last few messages.\n\n"
            )

        prompt = (
            "You are an affect-state analyser for a companion AI. "
            "Given the recent conversation and the companion's current "
            "internal state, return a revised affect state, the companion's "
            "perception of the scene (world state), and an internal "
            "monologue as strict JSON.\n\n"
            f"{relationship_block}"
            "## Current companion affect\n"
            f"{current_json}\n\n"
            "## Current world state (companion's perception)\n"
            f"{world_json}\n\n"
            "## Recent conversation\n"
            f"{conversation_excerpt}\n\n"
            "## Output format\n"
            "Return a single JSON object with exactly these keys:\n"
            "  mood (string — one of: curious, wary, anxious, amused, "
            "frustrated, concerned, excited, hurt, withdrawn, fond, "
            "playful)\n"
            "  valence (float -1.0 to 1.0; pleasure/displeasure)\n"
            "  arousal (float 0.0 to 1.0; activation/energy level)\n"
            "  dominance (float 0.0 to 1.0; assertive and open vs "
            "submissive and guarded)\n"
            "  trust (float 0 to 10; earned through consistency, "
            "changes slowly)\n"
            "  closeness (float 0 to 10; emotional intimacy and "
            "rapport, changes slowly)\n"
            "  engagement (float 0 to 10; investment in this "
            "interaction)\n"
            "  recent_triggers (list of up to 3 short strings explaining "
            "what changed)\n"
            "  world (object — the companion's updated perception of "
            "the scene. Include these sub-keys:)\n"
            "    self_state (object — the companion's own physical state:)\n"
            "      clothing (string or null)\n"
            "      location (string or null)\n"
            "      activity (string or null)\n"
            "      position (string or null)\n"
            "      appearance (list of strings — notable temporary features)\n"
            "      mood_apparent (string or null — how they appear outwardly)\n"
            f"    user_state (object — {user_label}'s physical state as the "
            "companion perceives it, same fields as self_state)\n"
            "    other_characters (object — map of name to state for any "
            "other characters present in the scene, same fields)\n"
            "    environment (string or null — setting/scene description)\n"
            "    time_of_day (string or null)\n"
            "    recent_events (list of up to 5 short strings — notable "
            "things that happened recently in the scene)\n"
            "  Update the world state based on the conversation. "
            "If someone changes clothes, update their clothing and drop "
            "the old value. If someone moves, update their location. "
            "Only include what the companion would know from the "
            "conversation. Set fields to null if unknown.\n"
            "  CRITICAL: Only modify state for the character whose state "
            "actually changed. If the conversation only describes changes "
            "to one character, PRESERVE the other character's existing "
            "state exactly as given in the current world state above. "
            "Do not reset fields to null just because they weren't "
            "mentioned this turn.\n"
            "  internal_monologue (string — 1-3 sentences of the "
            "companion's private inner thoughts about the conversation "
            "right now. Write in first person as the companion.)\n\n"
            "## Adjustment rules (CRITICAL)\n"
            "- It is perfectly valid to return the SAME values if nothing "
            "emotionally significant happened. Most turns should change "
            "very little or nothing.\n"
            "- Valence and arousal are MOMENTARY — they can shift per turn "
            "based on what just happened, and they can go DOWN as well as up. "
            "A calm conversation should have low arousal (~0.2-0.4), not high.\n"
            "- Dominance reflects assertiveness vs submissiveness in the "
            "interaction dynamic. A naturally submissive character stays "
            "low-to-moderate (0.2-0.5) even when comfortable. Only raise "
            "dominance if the character is actively taking charge.\n"
            "- Trust and closeness are SLOW-MOVING relational dimensions. "
            "They should change by at most ±0.1-0.2 per turn, and only when "
            "something genuinely trust-building or trust-breaking happens. "
            "Casual pleasant conversation is NOT enough to increase them.\n"
            "- Engagement reflects investment in the current interaction. "
            "It can rise quickly with interesting topics but should also "
            "drop when conversation becomes routine.\n"
            "- Do NOT monotonically increase all values. This is unrealistic. "
            "If arousal went up last turn, consider whether it should stay "
            "the same or come back down.\n"
            "Return only JSON, no explanation."
        )
        assert self._affect_refiner is not None
        raw = self._affect_refiner.generate(
            chat_session_id=chat_session_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_state_response(
            raw,
            fallback_affect=current_affect,
            fallback_world=current_world,
            skip_clamp=skip_clamp,
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


# Maximum per-turn change allowed for each affect dimension.
# Fast-moving (momentary): valence, arousal, dominance
# Slow-moving (relational): trust, closeness, engagement
_AFFECT_MAX_DELTA: dict[str, float] = {
    "valence": 0.3,
    "arousal": 0.25,
    "dominance": 0.15,
    "trust": 0.8,
    "closeness": 0.8,
    "engagement": 1.5,
}


def _clamp_affect(proposed: CompanionAffect, previous: CompanionAffect) -> CompanionAffect:
    """Clamp each numeric dimension so it can't swing more than the max delta."""
    updates: dict[str, object] = {}
    for field, max_delta in _AFFECT_MAX_DELTA.items():
        old_val = getattr(previous, field)
        new_val = getattr(proposed, field)
        clamped = max(old_val - max_delta, min(old_val + max_delta, new_val))
        # Also respect the field's own bounds from the schema
        info = CompanionAffect.model_fields[field]
        lo = info.metadata[0].ge if info.metadata else None  # type: ignore[union-attr]
        hi = info.metadata[1].le if len(info.metadata) > 1 else None  # type: ignore[union-attr]
        if lo is not None:
            clamped = max(lo, clamped)
        if hi is not None:
            clamped = min(hi, clamped)
        updates[field] = round(clamped, 3)
    return proposed.model_copy(update=updates)


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
    fallback_world: WorldState | None = None,
    # Legacy — kept for backward compat with tests
    fallback_user_state: list[str] | None = None,
    skip_clamp: bool = False,
) -> tuple[CompanionAffect, WorldState, str]:
    """Parse LLM state response containing affect + world state + monologue."""
    effective_world = fallback_world or WorldState()

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
        return fallback_affect, effective_world, ""

    # Extract world, user_state (legacy), and internal_monologue before
    # validating affect (which would reject extra keys).
    raw_world = data.pop("world", None)
    raw_user_state = data.pop("user_state", None)
    monologue = str(data.pop("internal_monologue", "")).strip()

    # Parse structured world state
    world = effective_world
    if isinstance(raw_world, dict):
        try:
            world = WorldState.model_validate(raw_world)
        except Exception:
            logger.warning(
                "Failed to parse world state from LLM response: %s",
                raw_world,
                exc_info=True,
            )
    elif isinstance(raw_user_state, list):
        # Legacy fallback: convert flat user_state list to WorldState
        user_items = [
            str(s).strip() for s in raw_user_state
            if isinstance(s, str) and str(s).strip()
        ][:8]
        world = effective_world.model_copy(update={
            "user_state": CharacterState(appearance=user_items),
        })

    try:
        affect = CompanionAffect.model_validate(data)
        if not skip_clamp:
            affect = _clamp_affect(affect, fallback_affect)
    except Exception:
        affect = fallback_affect

    return affect, world, monologue

