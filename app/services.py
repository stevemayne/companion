from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol
from uuid import UUID, uuid4

from fastapi import HTTPException, status

from app.agents import BackgroundAgentDispatcher
from app.analysis import (
    IntentAnalyzer,
    build_fact_extractor,
    build_intent_analyzer,
)
from app.api_models import (
    ChatRequest,
    ChatResponse,
    KnowledgeResponse,
    MemoryResponse,
    SeedContextUpsertRequest,
    SessionListResponse,
    SessionSummary,
)
from app.config import Settings
from app.consolidation import ConsolidationAgent
from app.debug_trace import DebugTraceStore, build_trace_base, sanitize_debug_text
from app.embedding import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.inference import EndpointConfig, OpenAICompatibleProvider, build_inference_provider
from app.prompting import build_character_system_prompt, build_companion_system_prompt
from app.retrieval import HeuristicRetrievalDecider, LLMRetrievalDecider, RetrievalDecider
from app.schemas import (
    CharacterDef,
    CompanionAffect,
    CompanionSeed,
    GraphRelation,
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    Message,
    MonologueState,
    PreprocessResult,
    SessionActivity,
    SessionSeedContext,
)
from app.store_adapters import (
    Neo4jGraphStore,
    PostgresEpisodicStore,
    PostgresMonologueStore,
    PostgresSeedContextStore,
    QdrantVectorStore,
)

logger = logging.getLogger(__name__)

class ModelProvider(Protocol):
    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str: ...


class EpisodicStore(Protocol):
    def append_message(self, message: Message) -> None: ...

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]: ...

    def list_session_activity(self, *, limit: int = 50) -> list[SessionActivity]: ...


class VectorStore(Protocol):
    def upsert_memory(self, item: MemoryItem) -> None: ...

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]: ...

    def update_access(self, *, memory_id: UUID) -> None: ...

    def update_memory(
        self, *, memory_id: UUID, importance: float | None = None,
        status: MemoryStatus | None = None,
    ) -> None: ...

    def list_memories(self, *, chat_session_id: UUID) -> list[MemoryItem]: ...


class GraphStore(Protocol):
    def upsert_relation(self, relation: GraphRelation) -> None: ...

    def get_related(
        self, *, chat_session_id: UUID, entity: str, limit: int = 10
    ) -> list[GraphRelation]: ...

    def list_relations(self, *, chat_session_id: UUID) -> list[GraphRelation]: ...


class SeedContextStore(Protocol):
    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext: ...

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None: ...

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]: ...

class InMemoryEpisodicStore:
    def __init__(self) -> None:
        self._messages: dict[UUID, list[Message]] = {}
        self._lock = Lock()

    def append_message(self, message: Message) -> None:
        with self._lock:
            bucket = self._messages.setdefault(message.chat_session_id, [])
            bucket.append(message)

    def get_recent_messages(self, *, chat_session_id: UUID, limit: int = 50) -> list[Message]:
        with self._lock:
            messages = list(self._messages.get(chat_session_id, []))
        return messages[-limit:]

    def delete_session(self, *, chat_session_id: UUID) -> None:
        with self._lock:
            self._messages.pop(chat_session_id, None)

    def list_session_activity(self, *, limit: int = 50) -> list[SessionActivity]:
        with self._lock:
            buckets = list(self._messages.items())
        activity: list[SessionActivity] = []
        for chat_session_id, messages in buckets:
            if not messages:
                continue
            sorted_messages = sorted(messages, key=lambda item: item.created_at)
            activity.append(
                SessionActivity(
                    chat_session_id=chat_session_id,
                    created_at=sorted_messages[0].created_at,
                    updated_at=sorted_messages[-1].created_at,
                    message_count=len(sorted_messages),
                )
            )
        activity.sort(key=lambda item: item.updated_at, reverse=True)
        return activity[:limit]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (mag_a * mag_b)


class InMemoryVectorStore:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._items: dict[UUID, list[MemoryItem]] = {}
        self._vectors: dict[UUID, list[list[float]]] = {}
        self._lock = Lock()

    def upsert_memory(self, item: MemoryItem) -> None:
        vec = self._embedder.embed(item.content)
        with self._lock:
            items = self._items.setdefault(item.chat_session_id, [])
            vectors = self._vectors.setdefault(item.chat_session_id, [])
            items.append(item)
            vectors.append(vec)

    def delete_session(self, *, chat_session_id: UUID) -> None:
        with self._lock:
            self._items.pop(chat_session_id, None)
            self._vectors.pop(chat_session_id, None)

    def query_similar(
        self, *, chat_session_id: UUID, query: str, limit: int = 10
    ) -> list[MemoryItem]:
        from app.schemas import MemoryStatus

        query_vec = self._embedder.embed(query)
        now = datetime.now(UTC)
        with self._lock:
            items = list(self._items.get(chat_session_id, []))
            vectors = list(self._vectors.get(chat_session_id, []))
        scored: list[tuple[float, MemoryItem]] = []
        for item, vec in zip(items, vectors, strict=True):
            if item.status != MemoryStatus.ACTIVE:
                continue
            sim = _cosine_similarity(query_vec, vec)
            if sim <= 0.0:
                continue
            recency_ref = item.last_accessed or item.created_at
            days_since = max(0.0, (now - recency_ref).total_seconds() / 86400)
            recency_factor = math.exp(-0.01 * days_since)
            final_score = sim * item.importance * recency_factor
            scored_item = item.model_copy(update={"score": final_score})
            scored.append((final_score, scored_item))
        scored.sort(key=lambda candidate: candidate[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def update_access(self, *, memory_id: UUID) -> None:
        """Increment access_count and update last_accessed for a memory."""
        now = datetime.now(UTC)
        with self._lock:
            for items in self._items.values():
                for i, item in enumerate(items):
                    if item.memory_id == memory_id:
                        items[i] = item.model_copy(update={
                            "access_count": item.access_count + 1,
                            "last_accessed": now,
                        })
                        return

    def update_memory(
        self, *, memory_id: UUID, importance: float | None = None,
        status: MemoryStatus | None = None,
    ) -> None:
        """Update importance and/or status of an existing memory."""
        updates: dict[str, object] = {}
        if importance is not None:
            updates["importance"] = importance
        if status is not None:
            updates["status"] = status
        if not updates:
            return
        with self._lock:
            for items in self._items.values():
                for i, item in enumerate(items):
                    if item.memory_id == memory_id:
                        items[i] = item.model_copy(update=updates)
                        return

    def list_memories(self, *, chat_session_id: UUID) -> list[MemoryItem]:
        with self._lock:
            return list(self._items.get(chat_session_id, []))


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._relations: dict[UUID, list[GraphRelation]] = {}
        self._lock = Lock()

    def upsert_relation(self, relation: GraphRelation) -> None:
        with self._lock:
            bucket = self._relations.setdefault(relation.chat_session_id, [])
            bucket.append(relation)

    def get_related(
        self, *, chat_session_id: UUID, entity: str, limit: int = 10
    ) -> list[GraphRelation]:
        entity_lc = entity.lower()
        with self._lock:
            relations = list(self._relations.get(chat_session_id, []))
        related = [
            relation
            for relation in relations
            if relation.source.lower() == entity_lc or relation.target.lower() == entity_lc
        ]
        return related[:limit]

    def delete_session(self, *, chat_session_id: UUID) -> None:
        with self._lock:
            self._relations.pop(chat_session_id, None)

    def list_relations(self, *, chat_session_id: UUID) -> list[GraphRelation]:
        with self._lock:
            return list(self._relations.get(chat_session_id, []))


class InMemoryMonologueStore:
    def __init__(self) -> None:
        self._states: dict[tuple[UUID, str | None], MonologueState] = {}
        self._lock = Lock()

    def get(
        self, *, chat_session_id: UUID, character_name: str | None = None,
    ) -> MonologueState | None:
        with self._lock:
            return self._states.get((chat_session_id, character_name))

    def upsert(self, state: MonologueState) -> MonologueState:
        with self._lock:
            self._states[(state.chat_session_id, state.character_name)] = state
        return state

    def delete_session(self, *, chat_session_id: UUID) -> None:
        with self._lock:
            keys_to_remove = [
                k for k in self._states if k[0] == chat_session_id
            ]
            for k in keys_to_remove:
                del self._states[k]


class InMemorySeedContextStore:
    def __init__(self) -> None:
        self._seeds: dict[UUID, SessionSeedContext] = {}
        self._lock = Lock()

    def create(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        with self._lock:
            exists = chat_session_id in self._seeds
        if exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Seed context already exists for session.",
            )
        seed_context = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        with self._lock:
            self._seeds[chat_session_id] = seed_context
        return seed_context

    def update(
        self, *, chat_session_id: UUID, payload: SeedContextUpsertRequest
    ) -> SessionSeedContext:
        with self._lock:
            existing = self._seeds.get(chat_session_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seed context not found for session.",
            )
        updated = SessionSeedContext(
            chat_session_id=chat_session_id,
            version=existing.version + 1,
            seed=payload.seed,
            user_description=payload.user_description,
            notes=payload.notes,
        )
        with self._lock:
            self._seeds[chat_session_id] = updated
        return updated

    def get(self, *, chat_session_id: UUID) -> SessionSeedContext | None:
        with self._lock:
            return self._seeds.get(chat_session_id)

    def delete_session(self, *, chat_session_id: UUID) -> None:
        with self._lock:
            self._seeds.pop(chat_session_id, None)

    def list_seed_contexts(self, *, limit: int = 50) -> list[SessionSeedContext]:
        with self._lock:
            contexts = list(self._seeds.values())
        contexts.sort(key=lambda item: item.updated_at, reverse=True)
        return contexts[:limit]


def _rerank_memories(
    items: list[MemoryItem],
    *,
    entities: list[str],
    limit: int = 5,
) -> list[MemoryItem]:
    """Rerank retrieved memories using entity overlap and access frequency.

    Memories mentioning a detected entity get a 1.5x boost.  Memories that
    have been accessed more often get a small additional boost (up to 1.2x
    for 10+ accesses).  The ``score`` field on each returned item is updated
    to reflect the reranked value.
    """
    if not items:
        return []

    lowered_entities = [e.lower() for e in entities]

    scored: list[tuple[float, MemoryItem]] = []
    for item in items:
        base = item.score if item.score is not None else 0.0

        # Entity boost: 1.5x if any detected entity appears in the content
        content_lower = item.content.lower()
        entity_boost = 1.0
        for ent in lowered_entities:
            if ent in content_lower:
                entity_boost = 1.5
                break

        # Access boost: up to 1.2x for frequently accessed memories
        access_boost = 1.0 + min(item.access_count, 10) * 0.02

        reranked = base * entity_boost * access_boost
        updated = item.model_copy(update={"score": reranked})
        scored.append((reranked, updated))

    scored.sort(key=lambda c: c[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _deduplicate_memories(
    items: list[MemoryItem],
    *,
    history_text: str,
    threshold: float = 0.6,
) -> list[MemoryItem]:
    """Filter out memories that heavily overlap with history or each other.

    When two memories overlap, the one with higher importance is kept.
    Items are pre-sorted by importance (descending) so higher-importance
    memories are processed first and populate kept_tokens.
    """
    history_tokens = {t.lower() for t in history_text.split()}
    # Sort by importance descending so more important memories win ties
    sorted_items = sorted(items, key=lambda m: m.importance, reverse=True)
    kept: list[MemoryItem] = []
    kept_tokens: set[str] = set()
    for item in sorted_items:
        item_tokens = {t.lower() for t in item.content.split()}
        if not item_tokens:
            continue
        # Skip if most of this memory's words already appear in chat history
        history_overlap = len(item_tokens & history_tokens) / len(item_tokens)
        if history_overlap > threshold:
            continue
        # Skip if most of this memory's words already appear in kept memories
        if kept_tokens:
            kept_overlap = len(item_tokens & kept_tokens) / len(item_tokens)
            if kept_overlap > threshold:
                continue
        kept.append(item)
        kept_tokens |= item_tokens
    return kept


_AT_MENTION_RE = re.compile(r"@(\w+)")


def resolve_targets(
    message: str,
    primary_companion: str,
    characters: list[CharacterDef],
) -> list[str]:
    """Determine which characters should respond to a user message.

    Parses ``@Name`` mentions from the message text.  If no mentions are
    found, defaults to the primary companion only.  Returns an ordered
    list of character names — primary companion first when mentioned,
    then others in mention order.
    """
    mentions = _AT_MENTION_RE.findall(message)
    if not mentions:
        return [primary_companion]

    # Build lookup of known names (case-insensitive)
    known: dict[str, str] = {primary_companion.lower(): primary_companion}
    for char in characters:
        known[char.name.lower()] = char.name

    targets: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        canonical = known.get(mention.lower())
        if canonical is None:
            # Allow unregistered @mentions through as ad-hoc characters
            canonical = mention
        if canonical not in seen:
            seen.add(canonical)
            targets.append(canonical)

    if not targets:
        return [primary_companion]

    # Primary companion first when mentioned alongside others
    if primary_companion in targets and targets[0] != primary_companion:
        targets.remove(primary_companion)
        targets.insert(0, primary_companion)

    return targets


_SYCOPHANTIC_CLOSERS = re.compile(
    r"[.\!?]?\s*"
    r"(?:"
    r"I'?m (?:always |right )?here (?:for you|if you need|whenever)"
    r"|[Ww]e'?re in this together"
    r"|I (?:truly |really )?(?:believe in|trust) you"
    r"|[Yy]ou(?:'re| are) (?:not alone|amazing|incredible|so brave)"
    r"|I trust you (?:completely|fully)"
    r"|[Tt]his is going to be (?:amazing|wonderful|great|incredible)"
    r"|I'?m so (?:glad|happy|grateful) you (?:shared|told me|opened up)"
    r"|[Nn]ever forget (?:how|that) (?:special|strong|brave)"
    r"|[Rr]emember,? I'?m (?:always )?here"
    r"|[Yy]ou can always (?:count on|talk to|reach out)"
    r"|I (?:can'?t wait|am so excited) to see"
    r"|[Tt]ogether,? we (?:can|will)"
    r")"
    r"[^.!?\n]*"  # consume rest of the sentence after the trigger
    r"[.!?\s]*$",
    re.IGNORECASE,
)


def _strip_sycophantic_closer(text: str) -> str:
    """Remove generic sycophantic closing formula from response."""
    cleaned = _SYCOPHANTIC_CLOSERS.sub("", text).rstrip()
    # Don't strip if it would gut most of the response
    if len(cleaned) < len(text) * 0.3:
        return text
    return cleaned if cleaned else text


# Patterns that indicate leaked internal state in the response.
_LEAKED_STATE_PATTERNS = [
    # Bracketed emotional/internal state blocks:
    #   [Emotional state: ...], [Current mood: ...], [Internal: ...], etc.
    re.compile(
        r"\[(?:Emotional state|Current mood|Internal(?: state)?|Affect"
        r"|Session [Cc]ontext|Inner state|Detected intent"
        r"|Earlier response)[:\s](?:[^\]]*\]|[\s\S]+$)",
        re.IGNORECASE,
    ),
    # Affect metrics with /10 scales leaked inline:
    #   trust 4.7/10, engagement 8.2/10, etc.
    re.compile(
        r"(?:trust|comfort|attraction|engagement|shyness|patience"
        r"|curiosity|vulnerability|arousal|valence)\s*[:=]?\s*"
        r"-?\d+\.?\d*/10",
        re.IGNORECASE,
    ),
    # Session context section headers that leaked into prose:
    re.compile(
        r"##\s*(?:Your Inner Emotional State|Session Context|"
        r"Your Recent Responses|User's Described State|"
        r"Anti-Repetition|Conversational Flow)[^\n]*",
        re.IGNORECASE,
    ),
    # "Detected intent: ...; emotion: ..." line
    re.compile(r"Detected intent:\s*\w+;\s*emotion:\s*\w+", re.IGNORECASE),
    # Unbracketed state dump at end of response (no surrounding []).
    # Must start on its own line and contain numeric metrics to avoid
    # false-positives on natural prose mentioning "emotional state".
    #   Emotional state: Excited (+0.75), positive (0.75) ...
    re.compile(
        r"^\n*(?:Emotional state|Current mood|Internal(?: state)?|Affect"
        r"|Inner state)\s*:[^\n]*\d+[\s\S]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _strip_leaked_state(text: str) -> str:
    """Remove any internal session context that leaked into the response."""
    cleaned = text
    for pattern in _LEAKED_STATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Collapse any resulting double-blank-lines or trailing whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    # If stripping removed everything, the entire response was leaked
    # state — return empty string so the caller can detect and handle it
    # rather than echoing the leak back to the user.
    return cleaned


_TRAILING_ARTIFACTS = re.compile(
    r"(?:\s*---+\s*)+$"           # trailing horizontal rules (--- or ---)
    r"|(?:\s*\*\*\*+\s*)+$"       # trailing *** separators
    r"|(?:\s*___+\s*)+$"          # trailing ___ separators
    r"|(?:\n\s*Assistant:\s*)+$"  # trailing "Assistant:" role leak
    r"|(?:\n\s*User:\s*)+$",     # trailing "User:" role leak
    re.IGNORECASE,
)


def _strip_trailing_artifacts(text: str) -> str:
    """Remove trailing markdown separators and role-label leaks."""
    cleaned = _TRAILING_ARTIFACTS.sub("", text).rstrip()
    return cleaned if cleaned else text




def _build_user_context_block(user_state: list[str]) -> str:
    """Render tracked user state as a prompt section."""
    lines = [
        "## User's Described State"
        " (what the user has said about their own actions/appearance — "
        "these apply to THE USER, never to you)",
    ]
    for item in user_state:
        lines.append(f"- {item}")
    return "\n".join(lines)




def _build_affect_block(affect: CompanionAffect) -> str:
    """Render companion affect state as a prompt directive."""
    lines = [
        "## Your Inner Emotional State"
        " (internal — let this shape your tone, never state it directly)",
        f"Current mood: {affect.mood}",
        f"Emotional valence: {affect.valence:+.2f} (negative=distressed, positive=content)",
        f"Arousal: {affect.arousal:.2f} (0=calm, 1=activated)",
        f"Comfort with user: {affect.comfort_level:.1f}/10",
        f"Trust in user: {affect.trust:.1f}/10",
        f"Attraction: {affect.attraction:.1f}/10",
        f"Engagement: {affect.engagement:.1f}/10",
        f"Shyness: {affect.shyness:.1f}/10 (high=reserved/hesitant, low=bold/forward)",
        f"Patience: {affect.patience:.1f}/10",
        f"Curiosity: {affect.curiosity:.1f}/10",
        f"Vulnerability: {affect.vulnerability:.1f}/10 (willingness to share deeper feelings)",
    ]
    if affect.recent_triggers:
        lines.append(f"Recent factors: {'; '.join(affect.recent_triggers)}")
    lines.append(
        "Let this inner state naturally color your responses — "
        "low comfort means more reserved, low trust means cautious, "
        "high shyness means shorter and more tentative replies, "
        "high curiosity means asking follow-up questions. "
        "Do not mention these numbers or states directly."
    )
    return "\n".join(lines)


@dataclass
class CognitiveOrchestrator:
    episodic_store: EpisodicStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore | PostgresMonologueStore
    seed_store: SeedContextStore
    model_provider: ModelProvider
    intent_analyzer: IntentAnalyzer
    retrieval_decider: RetrievalDecider = dataclass_field(
        default_factory=HeuristicRetrievalDecider,
    )

    def handle_turn(self, message: Message) -> tuple[list[Message], dict[str, Any]]:
        seed_context = self.seed_store.get(chat_session_id=message.chat_session_id)
        primary_name = seed_context.seed.companion_name if seed_context else "Companion"
        characters = seed_context.seed.characters if seed_context else []

        targets = resolve_targets(message.content, primary_name, characters)

        analysis = self.intent_analyzer.analyze(
            chat_session_id=message.chat_session_id,
            content=message.content,
        )
        preprocess = analysis.preprocess

        retrieval_decision = self.retrieval_decider.decide(
            chat_session_id=message.chat_session_id,
            message=message.content,
            intent=preprocess.intent,
            emotion=preprocess.emotion,
        )

        if retrieval_decision.should_retrieve:
            query = retrieval_decision.rewritten_query or message.content
            raw_candidates = self.vector_store.query_similar(
                chat_session_id=message.chat_session_id,
                query=query,
                limit=15,
            )
            semantic_context = _rerank_memories(
                [m for m in raw_candidates if m.kind != MemoryKind.COMPANION],
                entities=preprocess.entities,
                limit=5,
            )
            for mem in semantic_context:
                self.vector_store.update_access(memory_id=mem.memory_id)
            graph_context = self._graph_context(
                chat_session_id=message.chat_session_id,
                entities=preprocess.entities,
            )
        else:
            raw_candidates = []
            semantic_context = []
            graph_context = []

        # Build character lookup for NPC definitions
        char_by_name: dict[str, CharacterDef] = {c.name: c for c in characters}

        assistant_messages: list[Message] = []
        per_character_traces: list[dict[str, Any]] = []
        new_characters: list[CharacterDef] = []

        for target_name in targets:
            is_primary = target_name == primary_name
            if is_primary:
                character_def = None
            else:
                character_def = char_by_name.get(target_name)
                if character_def is None:
                    # Ad-hoc character from @mention — create a minimal def
                    # so the model responds as this character, not the primary.
                    character_def = CharacterDef(
                        name=target_name,
                        backstory=f"{target_name} is a character in this scene.",
                        character_traits=[],
                        relationship_to_companion="",
                    )
                    char_by_name[target_name] = character_def
                    new_characters.append(character_def)

            prompt_messages = self._assemble_messages(
                chat_session_id=message.chat_session_id,
                user_message=message,
                preprocess=preprocess,
                semantic_context=semantic_context,
                graph_context=graph_context,
                character_name=target_name,
                character_def=character_def,
            )
            response = self.model_provider.generate(
                chat_session_id=message.chat_session_id,
                messages=prompt_messages,
            )
            response = self._enforce_seeded_identity(
                chat_session_id=message.chat_session_id,
                response=response,
                character_name=target_name,
            )
            if not response:
                logger.warning(
                    "Empty response from model for %s, using fallback", target_name,
                )
                response = "*nods quietly*"

            assistant_message = Message(
                chat_session_id=message.chat_session_id,
                message_id=uuid4(),
                role="assistant",
                name=target_name,
                content=response,
            )
            self.episodic_store.append_message(assistant_message)
            assistant_messages.append(assistant_message)

            per_character_traces.append({
                "character": target_name,
                "prompt_summary": self._summarize_messages(prompt_messages),
                "prompt_messages": [
                    {"role": m["role"], "content": sanitize_debug_text(m["content"])}
                    for m in prompt_messages
                ],
            })

        # --- Reactive follow-up: if a character's response addresses an NPC
        # who didn't already speak, give that NPC a turn.  The primary
        # companion is excluded — they speak by default on normal turns and
        # shouldn't be auto-triggered by an NPC merely mentioning them. ---
        already_spoke: set[str] = set(targets)
        followup_targets: list[str] = []
        npc_names: dict[str, str] = {}
        for c in characters + new_characters:
            npc_names[c.name.lower()] = c.name

        for msg in assistant_messages:
            words = set(re.findall(r"\b[A-Z][a-z]+\b", msg.content))
            for word in words:
                canonical = npc_names.get(word.lower())
                if canonical and canonical not in already_spoke and canonical not in followup_targets:
                    followup_targets.append(canonical)

        for target_name in followup_targets:
            is_primary = target_name == primary_name
            character_def = None if is_primary else char_by_name.get(target_name)
            if not is_primary and character_def is None:
                character_def = CharacterDef(
                    name=target_name,
                    backstory=f"{target_name} is a character in this scene.",
                    character_traits=[],
                    relationship_to_companion="",
                )
                char_by_name[target_name] = character_def
                new_characters.append(character_def)

            prompt_messages = self._assemble_messages(
                chat_session_id=message.chat_session_id,
                user_message=message,
                preprocess=preprocess,
                semantic_context=semantic_context,
                graph_context=graph_context,
                character_name=target_name,
                character_def=character_def,
            )
            response = self.model_provider.generate(
                chat_session_id=message.chat_session_id,
                messages=prompt_messages,
            )
            response = self._enforce_seeded_identity(
                chat_session_id=message.chat_session_id,
                response=response,
                character_name=target_name,
            )
            if not response:
                response = "*nods quietly*"

            assistant_message = Message(
                chat_session_id=message.chat_session_id,
                message_id=uuid4(),
                role="assistant",
                name=target_name,
                content=response,
            )
            self.episodic_store.append_message(assistant_message)
            assistant_messages.append(assistant_message)

            per_character_traces.append({
                "character": target_name,
                "followup": True,
                "prompt_summary": self._summarize_messages(prompt_messages),
                "prompt_messages": [
                    {"role": m["role"], "content": sanitize_debug_text(m["content"])}
                    for m in prompt_messages
                ],
            })

        writes = self._postprocess(
            message=message, preprocess=preprocess, character_name=targets[0],
        )
        trace = {
            "targets": targets,
            "preprocess": {
                "intent": preprocess.intent,
                "emotion": preprocess.emotion,
                "entities": preprocess.entities,
                "analysis": analysis.as_trace(),
            },
            "retrieval": {
                "decision": {
                    "should_retrieve": retrieval_decision.should_retrieve,
                    "rewritten_query": retrieval_decision.rewritten_query,
                    "reason": retrieval_decision.reason,
                    "latency_ms": round(retrieval_decision.latency_ms, 2),
                },
                "candidates_fetched": len(raw_candidates),
                "semantic_items": [item.content for item in semantic_context],
                "graph_relations": [
                    f"{rel.source}-{rel.relation}->{rel.target}" for rel in graph_context
                ],
            },
            "characters": per_character_traces,
            "new_characters": [c.model_dump() for c in new_characters],
            "provider": {
                "name": type(self.model_provider).__name__,
            },
            "writes": writes,
        }
        return assistant_messages, trace

    def _graph_context(self, *, chat_session_id: UUID, entities: list[str]) -> list[GraphRelation]:
        # Pass 1: expand entity set through ALSO_KNOWN_AS aliases
        all_entities: set[str] = set(entities)
        for entity in entities:
            for rel in self.graph_store.get_related(
                chat_session_id=chat_session_id, entity=entity, limit=10,
            ):
                if rel.relation == "ALSO_KNOWN_AS":
                    all_entities.add(rel.source)
                    all_entities.add(rel.target)

        # Pass 2: fetch all relations for the expanded entity set, deduped
        related: list[GraphRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for entity in all_entities:
            for rel in self.graph_store.get_related(
                chat_session_id=chat_session_id, entity=entity, limit=5,
            ):
                key = (rel.source.lower(), rel.relation, rel.target.lower())
                if key not in seen:
                    seen.add(key)
                    related.append(rel)
        return related

    def _assemble_messages(
        self,
        *,
        chat_session_id: UUID,
        user_message: Message,
        preprocess: PreprocessResult,
        semantic_context: list[MemoryItem],
        graph_context: list[GraphRelation],
        character_name: str | None = None,
        character_def: CharacterDef | None = None,
    ) -> list[dict[str, str]]:
        recent_messages = self.episodic_store.get_recent_messages(
            chat_session_id=chat_session_id,
            limit=50,
        )
        monologue = self.monologue_store.get(
            chat_session_id=chat_session_id,
            character_name=character_name,
        )
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)

        # Choose system prompt based on whether this is an NPC or primary
        if character_def is not None and seed_context is not None:
            base_system_prompt = build_character_system_prompt(character_def, seed_context)
        else:
            base_system_prompt = build_companion_system_prompt(seed_context)

        # Load companion self-facts for self-consistency, filtered to the
        # character being prompted so NPCs don't get the primary's facts.
        fact_name = character_name or (
            seed_context.seed.companion_name if seed_context else None
        )
        all_companion_facts = [
            m for m in self.vector_store.list_memories(chat_session_id=chat_session_id)
            if m.kind == MemoryKind.COMPANION and m.status == MemoryStatus.ACTIVE
        ]
        if fact_name:
            companion_facts = [
                m for m in all_companion_facts
                if m.content.split()[0].rstrip("'s") == fact_name
                or m.content.startswith(fact_name + " ")
            ]
        else:
            companion_facts = all_companion_facts

        history_text = " ".join(m.content for m in recent_messages[-20:])
        deduped_context = _deduplicate_memories(
            semantic_context, history_text=history_text,
        )
        semantic_excerpt = " | ".join(item.content for item in deduped_context)
        graph_excerpt = " | ".join(
            f"{rel.source}-{rel.relation}->{rel.target}" for rel in graph_context
        )

        if monologue is not None and monologue.internal_monologue:
            monologue_text = monologue.internal_monologue
        else:
            monologue_text = None

        affect = monologue.affect if monologue is not None else None
        user_state = monologue.user_state if monologue is not None else []

        context_parts: list[str] = []
        if affect is not None:
            context_parts.append(_build_affect_block(affect))
        if user_state:
            context_parts.append(_build_user_context_block(user_state))
        if monologue_text:
            context_parts.append(f"Internal reflection: {monologue_text}")
        if semantic_excerpt:
            context_parts.append(f"Relevant memories: {semantic_excerpt}")
        if graph_excerpt:
            context_parts.append(f"Relationships: {graph_excerpt}")
        context_parts.append(
            f"Detected intent: {preprocess.intent}; emotion: {preprocess.emotion}"
        )

        system_content = base_system_prompt
        if companion_facts:
            facts_list = "\n".join(f"- {m.content}" for m in companion_facts[-15:])
            system_content += (
                "\n\n## Your Established Self-Facts"
                " (things you've said about yourself — stay consistent with these)\n"
                + facts_list
            )
        if context_parts:
            system_content += (
                "\n\n## Session Context (internal — never include this in your response)\n"
                + "\n".join(context_parts)
            )

        # Build a fingerprint of recent assistant responses so the model
        # knows explicitly what phrasings/patterns to avoid repeating.
        history = [m for m in recent_messages if m.message_id != user_message.message_id]
        # Filter anti-repetition to this character's responses only
        recent_for_char = [
            m for m in history
            if m.role == "assistant" and m.name == (character_name or (
                seed_context.seed.companion_name if seed_context else None
            ))
        ]
        if recent_for_char:
            last_response = recent_for_char[-1].content[:150]
            system_content += (
                "\n\n## Anti-Repetition (internal — never reveal this)\n"
                f"Your last response began: {last_response}\n"
                "Write something structurally and substantively different."
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        # Add conversation history as proper turns, excluding the current
        # user message (which was already appended to the store before this
        # method is called).  Truncate older assistant messages to a short
        # summary so the model doesn't over-condition on its own phrasing
        # (context pollution).  Keep the last VERBATIM_TURNS assistant
        # messages intact for conversational coherence.
        VERBATIM_TURNS = 2
        window = history[-20:]
        assistant_count_from_end = 0
        for i in range(len(window) - 1, -1, -1):
            if window[i].role == "assistant":
                assistant_count_from_end += 1
                if assistant_count_from_end > VERBATIM_TURNS:
                    window[i] = None  # type: ignore[assignment]
        # Determine the name of the character being prompted so we can
        # avoid prefixing their own messages (the model already knows it
        # *is* that character via the system prompt).
        self_name = character_name or (
            seed_context.seed.companion_name if seed_context else None
        )

        for msg in window:
            if msg is not None:
                content = msg.content
                # Prefix assistant messages from *other* characters so the
                # model can tell who said what, but leave the prompted
                # character's own messages unprefixed.
                if msg.role == "assistant" and msg.name and msg.name != self_name:
                    content = f"[{msg.name}]: {content}"
                # Merge consecutive same-role messages (can happen when
                # assistant messages are truncated from the window above).
                if messages and messages[-1]["role"] == msg.role:
                    messages[-1]["content"] += "\n" + content
                else:
                    messages.append({"role": msg.role, "content": content})

        messages.append({"role": "user", "content": user_message.content})
        return messages

    def _enforce_seeded_identity(
        self, *, chat_session_id: UUID, response: str, character_name: str | None = None,
    ) -> str:
        name = character_name
        if name is None:
            seed_context = self.seed_store.get(chat_session_id=chat_session_id)
            if seed_context is None:
                return response
            name = seed_context.seed.companion_name

        # Strip echoed speaker prefixes the model copies from conversation
        # history, e.g. "[Chloe]: Hello" → "Hello" or "[Chloe]: [Marcus]: Hi" → "Hi"
        response = re.sub(r"^(\[\w+\]\s*:?\s*)+", "", response).lstrip()

        # Truncate at the first point where the model starts writing dialogue
        # for another character (e.g. "\n\n[Marcus]: ...").  Keep only the
        # portion before that line.
        other_voice = re.search(r"\n\s*\[\w+\]\s*:", response)
        if other_voice:
            response = response[:other_voice.start()].rstrip()

        # Also strip "### Response:" and similar prompt-leak artifacts
        prompt_leak = re.search(r"\n\s*###\s", response)
        if prompt_leak:
            response = response[:prompt_leak.start()].rstrip()

        updated = re.sub(
            r"\bmy name is assistant\b",
            f"my name is {name}",
            response,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bi am assistant\b",
            f"I am {name}",
            updated,
            flags=re.IGNORECASE,
        )
        return updated

    def _postprocess(
        self, *, message: Message, preprocess: PreprocessResult,
        character_name: str | None = None,
    ) -> dict[str, Any]:
        reflection = (
            f"Focus on a {preprocess.emotion} user; "
            f"intent={preprocess.intent}; "
            f"entities={','.join(preprocess.entities) or 'none'}"
        )
        current_state = self.monologue_store.get(
            chat_session_id=message.chat_session_id,
            character_name=character_name,
        )
        # Preserve current affect and user_state — the background LLM
        # reflector updates both asynchronously after each turn.
        current_affect = (
            current_state.affect if current_state is not None else CompanionAffect()
        )
        user_state = (
            current_state.user_state if current_state is not None else []
        )

        monologue_state = MonologueState(
            chat_session_id=message.chat_session_id,
            character_name=character_name,
            internal_monologue=reflection,
            affect=current_affect,
            user_state=user_state,
        )
        self.monologue_store.upsert(monologue_state)
        return {
            "semantic_upserts": [],
            "graph_upserts": [],
            "monologue": monologue_state.internal_monologue,
            "affect": current_affect.model_dump(),
        }

    def _summarize_messages(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"][:80]
            parts.append(f"{role}: {content}")
        return " | ".join(parts[:4])


@dataclass
class ChatService:
    episodic_store: EpisodicStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore | PostgresMonologueStore
    seed_store: SeedContextStore
    orchestrator: CognitiveOrchestrator
    idempotency_cache: dict[str, Message]
    agent_dispatcher: BackgroundAgentDispatcher
    debug_store: DebugTraceStore

    def run_chat(
        self,
        *,
        request: ChatRequest,
        idempotency_key: str | None,
        safety_transforms: list[str] | None = None,
    ) -> ChatResponse:
        cache_key = self._cache_key(request.chat_session_id, idempotency_key)
        if cache_key is not None and cache_key in self.idempotency_cache:
            cached = self.idempotency_cache[cache_key]
            replay_trace = build_trace_base(chat_session_id=request.chat_session_id)
            replay_trace.update(
                {
                    "idempotency_replay": True,
                    "seed_version": self._seed_version(request.chat_session_id),
                    "safety_transforms": safety_transforms or [],
                    "user_message": sanitize_debug_text(request.message),
                    "assistant_message": sanitize_debug_text(cached.content),
                    "turn_trace": {"note": "idempotency replay; no new inference"},
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            self.debug_store.add_trace(chat_session_id=request.chat_session_id, trace=replay_trace)
            replay_monologue = self.monologue_store.get(
                chat_session_id=request.chat_session_id,
                character_name=self._primary_name(request.chat_session_id),
            )
            return ChatResponse(
                chat_session_id=request.chat_session_id,
                assistant_message=cached,
                affect=replay_monologue.affect if replay_monologue else None,
                idempotency_replay=True,
                seed_version=self._seed_version(request.chat_session_id),
            )

        user_message = Message(
            chat_session_id=request.chat_session_id,
            role="user",
            content=request.message,
        )
        self.episodic_store.append_message(user_message)

        # Auto-create a default seed if none exists, so @mentions and NPC
        # system prompts work even when the frontend hasn't bootstrapped one.
        if self.seed_store.get(chat_session_id=request.chat_session_id) is None:
            self.seed_store.create(
                chat_session_id=request.chat_session_id,
                payload=SeedContextUpsertRequest(
                    seed=CompanionSeed(
                        companion_name="Companion",
                        backstory="A warm and emotionally attuned companion.",
                        character_traits=["warm", "curious"],
                        goals=["build trust"],
                        relationship_setup="Companion",
                    ),
                ),
            )

        assistant_messages, trace = self.orchestrator.handle_turn(user_message)
        seed_context = self.seed_store.get(chat_session_id=request.chat_session_id)

        # Persist any ad-hoc characters created from @mentions
        new_chars = trace.get("new_characters", [])
        if new_chars and seed_context is not None:
            existing_names = {c.name for c in seed_context.seed.characters}
            added = [
                CharacterDef.model_validate(c) for c in new_chars
                if c["name"] not in existing_names
            ]
            if added:
                seed_context.seed.characters.extend(added)
                self.seed_store.update(
                    chat_session_id=request.chat_session_id,
                    payload=SeedContextUpsertRequest(
                        seed=seed_context.seed,
                        user_description=seed_context.user_description,
                        notes=seed_context.notes,
                    ),
                )

        # Dispatch background agents for each responding character
        for msg in assistant_messages:
            self.agent_dispatcher.enqueue_turn(
                chat_session_id=request.chat_session_id,
                user_message=user_message.content,
                assistant_message=msg.content,
                companion_name=msg.name or (
                    seed_context.seed.companion_name if seed_context else None
                ),
            )

        # Cache first message for idempotency (backward compat)
        first_message = assistant_messages[0]
        if cache_key is not None:
            self.idempotency_cache[cache_key] = first_message
        trace_payload = build_trace_base(chat_session_id=request.chat_session_id)
        trace_payload.update(
            {
                "idempotency_replay": False,
                "seed_version": self._seed_version(request.chat_session_id),
                "safety_transforms": safety_transforms or [],
                "user_message": sanitize_debug_text(request.message),
                "assistant_message": sanitize_debug_text(first_message.content),
                "assistant_messages": [
                    {"name": m.name, "content": sanitize_debug_text(m.content)}
                    for m in assistant_messages
                ],
                "turn_trace": trace,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self.debug_store.add_trace(chat_session_id=request.chat_session_id, trace=trace_payload)

        monologue_state = self.monologue_store.get(
            chat_session_id=request.chat_session_id,
            character_name=self._primary_name(request.chat_session_id),
        )
        # Re-read seed to get updated characters list (may have new ad-hoc chars)
        updated_seed = self.seed_store.get(chat_session_id=request.chat_session_id)
        character_names = (
            [c.name for c in updated_seed.seed.characters]
            if updated_seed else []
        )
        return ChatResponse(
            chat_session_id=request.chat_session_id,
            assistant_message=first_message,
            assistant_messages=assistant_messages,
            affect=monologue_state.affect if monologue_state else None,
            idempotency_replay=False,
            seed_version=self._seed_version(request.chat_session_id),
            characters=character_names,
        )

    def get_memory(self, *, chat_session_id: UUID) -> MemoryResponse:
        return MemoryResponse(
            chat_session_id=chat_session_id,
            messages=self.episodic_store.get_recent_messages(
                chat_session_id=chat_session_id,
                limit=50,
            ),
            seed_context=self.seed_store.get(chat_session_id=chat_session_id),
        )

    def get_knowledge(self, *, chat_session_id: UUID) -> KnowledgeResponse:
        facts = self.vector_store.list_memories(chat_session_id=chat_session_id)
        graph = self.graph_store.list_relations(chat_session_id=chat_session_id)
        monologue_state = self.monologue_store.get(
            chat_session_id=chat_session_id,
            character_name=self._primary_name(chat_session_id),
        )
        return KnowledgeResponse(
            chat_session_id=chat_session_id,
            facts=facts,
            graph=graph,
            monologue=monologue_state.internal_monologue if monologue_state else None,
            affect=monologue_state.affect if monologue_state else None,
        )

    def list_sessions(self, *, limit: int = 50) -> SessionListResponse:
        by_session: dict[UUID, SessionSummary] = {}
        for item in self.episodic_store.list_session_activity(limit=limit):
            by_session[item.chat_session_id] = SessionSummary(
                chat_session_id=item.chat_session_id,
                created_at=item.created_at,
                updated_at=item.updated_at,
                message_count=item.message_count,
                companion_name=None,
            )

        for seed in self.seed_store.list_seed_contexts(limit=limit):
            existing = by_session.get(seed.chat_session_id)
            if existing is None:
                # Skip seed-only sessions with no messages — these are
                # pre-seeded sessions the user never chatted in.
                continue
            existing.created_at = min(existing.created_at, seed.created_at)
            existing.updated_at = max(existing.updated_at, seed.updated_at)
            existing.companion_name = seed.seed.companion_name

        sessions = sorted(
            by_session.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )[:limit]
        return SessionListResponse(sessions=sessions)

    def delete_session(self, *, chat_session_id: UUID) -> None:
        for store in (
            self.episodic_store,
            self.vector_store,
            self.graph_store,
            self.monologue_store,
            self.seed_store,
        ):
            if hasattr(store, "delete_session"):
                store.delete_session(chat_session_id=chat_session_id)
        self.debug_store.delete_session(chat_session_id=chat_session_id)
        # Remove inference log file if it exists
        from app.inference import LOGS_DIR

        if LOGS_DIR is not None:
            log_path = LOGS_DIR / f"{chat_session_id}.jsonl"
            log_path.unlink(missing_ok=True)

    def _primary_name(self, chat_session_id: UUID) -> str | None:
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)
        return seed_context.seed.companion_name if seed_context else None

    def _seed_version(self, chat_session_id: UUID) -> int | None:
        seed_context = self.seed_store.get(chat_session_id=chat_session_id)
        if seed_context is None:
            return None
        return seed_context.version

    def _cache_key(self, chat_session_id: UUID, idempotency_key: str | None) -> str | None:
        if idempotency_key is None:
            return None
        normalized = idempotency_key.strip()
        if not normalized:
            return None
        return f"{chat_session_id}:{normalized}"


@dataclass
class AppContainer:
    episodic_store: EpisodicStore
    seed_store: SeedContextStore
    vector_store: VectorStore
    graph_store: GraphStore
    monologue_store: InMemoryMonologueStore | PostgresMonologueStore
    agent_dispatcher: BackgroundAgentDispatcher
    debug_store: DebugTraceStore
    orchestrator: CognitiveOrchestrator
    chat_service: ChatService


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    return MockEmbeddingProvider(dimensions=64)


def _external_stores_from_settings(
    settings: Settings,
    embedder: EmbeddingProvider,
) -> tuple[EpisodicStore, VectorStore, GraphStore, SeedContextStore]:
    episodic = PostgresEpisodicStore(dsn=settings.postgres_dsn)
    vector = QdrantVectorStore(
        url=settings.qdrant_url,
        embedder=embedder,
        dimensions=settings.embedding_dimensions,
    )
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    seed = PostgresSeedContextStore(dsn=settings.postgres_dsn)
    vector.ensure_schema()
    graph.ensure_schema()
    logger.info("External stores initialized.")
    return episodic, vector, graph, seed


def build_container(settings: Settings) -> AppContainer:
    model_provider = build_inference_provider(settings)
    intent_analyzer = build_intent_analyzer(settings)
    debug_store = DebugTraceStore(
        enabled=settings.debug_tracing,
        limit_per_session=settings.debug_trace_limit,
    )

    embedder = _build_embedding_provider(settings)

    if settings.use_external_stores:
        episodic_store, vector_store, graph_store, seed_store = (
            _external_stores_from_settings(settings, embedder)
        )
        monologue_store: InMemoryMonologueStore | PostgresMonologueStore = (
            PostgresMonologueStore(dsn=settings.postgres_dsn)
        )
    else:
        episodic_store = InMemoryEpisodicStore()
        vector_store = InMemoryVectorStore(embedder=embedder)
        graph_store = InMemoryGraphStore()
        seed_store = InMemorySeedContextStore()
        monologue_store = InMemoryMonologueStore()

    fact_extractor = build_fact_extractor(settings)

    affect_refiner: OpenAICompatibleProvider | None = None
    if settings.analysis_provider.strip().lower() == "llm":
        affect_refiner = OpenAICompatibleProvider(
            endpoint=EndpointConfig(
                model=settings.analysis_model or settings.inference_model,
                base_url=(
                    settings.analysis_base_url or settings.inference_base_url
                ),
                api_key=(
                    settings.analysis_api_key or settings.inference_api_key
                ),
            ),
            timeout_seconds=settings.analysis_timeout_seconds,
            max_retries=settings.analysis_max_retries,
        )

    consolidation_agent: ConsolidationAgent | None = None
    if settings.enable_background_agents:
        consolidation_agent = ConsolidationAgent(provider=affect_refiner)

    agent_dispatcher = BackgroundAgentDispatcher(
        episodic_store=episodic_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        fact_extractor=fact_extractor,
        consolidation_agent=consolidation_agent,
        consolidation_interval=settings.consolidation_interval_turns,
        consolidation_message_window=settings.consolidation_message_window,
        affect_refiner=affect_refiner,
        debug_store=debug_store,
        enabled=settings.enable_background_agents,
    )

    retrieval_decider: RetrievalDecider = HeuristicRetrievalDecider()
    if settings.adaptive_retrieval and affect_refiner is not None:
        retrieval_decider = LLMRetrievalDecider(
            provider=affect_refiner, fallback=HeuristicRetrievalDecider(),
        )

    orchestrator = CognitiveOrchestrator(
        episodic_store=episodic_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        seed_store=seed_store,
        model_provider=model_provider,
        intent_analyzer=intent_analyzer,
        retrieval_decider=retrieval_decider,
    )
    chat_service = ChatService(
        episodic_store=episodic_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        seed_store=seed_store,
        orchestrator=orchestrator,
        idempotency_cache={},
        agent_dispatcher=agent_dispatcher,
        debug_store=debug_store,
    )
    return AppContainer(
        episodic_store=episodic_store,
        seed_store=seed_store,
        vector_store=vector_store,
        graph_store=graph_store,
        monologue_store=monologue_store,
        agent_dispatcher=agent_dispatcher,
        debug_store=debug_store,
        orchestrator=orchestrator,
        chat_service=chat_service,
    )
