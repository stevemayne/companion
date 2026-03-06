# Companion Abstraction Design

## Vision

A **Companion** is a self-contained autonomous agent with its own identity,
memories, knowledge graph, emotional state, and temporal arc. A **Session** is
a shared space where the user and one or more Companions interact. Companions
can respond to the user, to each other, or initiate interaction on their own.

```
Session (shared space)
├── Transcript        (shared message log — who said what)
├── Companion "Chloe" (own memories, graph, affect, monologue, seed)
├── Companion "Marcus" (own memories, graph, affect, monologue, seed)
└── World State       (time, location, events — future game engine link)
```

## Core Abstraction

### CompanionContext — the black box

Every companion is represented at runtime by a `CompanionContext` that bundles
all of its private state behind a uniform interface. The orchestrator never
reaches into session-level stores directly — it receives a `CompanionContext`
and operates on it.

```python
@dataclass
class CompanionContext:
    """Everything the orchestrator needs to run one companion's turn."""
    companion_id: UUID
    seed: CompanionSeed              # identity, backstory, traits, goals
    memories: VectorStore            # scoped to this companion
    graph: GraphStore                # scoped to this companion
    monologue: MonologueStore        # scoped to this companion
    affect: CompanionAffect          # current emotional state
    user_description: str | None     # shared across session (read-only)
```

The stores inside a `CompanionContext` are **views** — thin wrappers that
automatically scope all reads/writes to `(session_id, companion_id)`. The
underlying storage is shared infrastructure, but each companion only sees
its own partition.

```python
class ScopedVectorStore:
    """Wraps a VectorStore, pinning all operations to one companion."""
    def __init__(self, inner: VectorStore, session_id: UUID, companion_id: UUID):
        self._inner = inner
        self._session_id = session_id
        self._companion_id = companion_id

    def query_similar(self, *, query: str, limit: int = 10) -> list[MemoryItem]:
        return self._inner.query_similar(
            session_id=self._session_id,
            companion_id=self._companion_id,
            query=query, limit=limit,
        )
    # ... upsert_memory, list_memories, etc. all scoped the same way
```

### Session — the container

```python
@dataclass
class Session:
    session_id: UUID
    transcript: Transcript           # shared message log
    companions: dict[UUID, CompanionContext]
    world: WorldState | None         # future: game engine state
```

### Transcript — shared history with attribution

The message log is shared (everyone can see what was said) but each message
is attributed to a speaker:

```python
class Message(BaseModel):
    session_id: UUID
    message_id: UUID
    speaker_id: UUID | None          # None = user, else companion_id
    speaker_name: str                # display name at time of message
    role: Role                       # "user" | "assistant" | "system"
    content: str
    created_at: datetime
```

When assembling a prompt for Companion A, the orchestrator reads the shared
transcript and labels messages from other speakers (Companion B, the user)
but leaves Companion A's own messages as plain `assistant` turns.

## Schema Changes

### New: CompanionSeed (extended)

```python
class CompanionSeed(BaseModel):
    companion_name: str
    backstory: str
    character_traits: list[str]
    goals: list[str]
    relationship_setup: str          # relationship to the user
    # --- new fields ---
    routine: str | None = None       # daily routine description
    location: str | None = None      # current location in world
    relationships: dict[str, str] = {}  # companion_name -> relationship desc
```

### Extended: MemoryItem, GraphRelation, MonologueState

Add `companion_id: UUID` to each. This is the partition key that scoped
stores use to filter. The field is set automatically by the scoped wrapper.

```python
class MemoryItem(BaseModel):
    session_id: UUID
    companion_id: UUID               # NEW
    memory_id: UUID
    kind: MemoryKind
    content: str
    # ... rest unchanged

class GraphRelation(BaseModel):
    session_id: UUID
    companion_id: UUID               # NEW
    source: str
    relation: str
    target: str
    confidence: float

class MonologueState(BaseModel):
    session_id: UUID
    companion_id: UUID               # NEW
    internal_monologue: str
    affect: CompanionAffect
    user_state: list[str]
    updated_at: datetime
```

## Orchestrator Refactor

### Current (session-centric)

```
handle_turn(message) →
    seed = seed_store.get(session_id)         # one companion
    monologue = monologue_store.get(session_id)  # one state
    facts = vector_store.list(session_id)     # all facts
    ...generate response...
    monologue_store.upsert(session_id, ...)   # one state
```

### Proposed (companion-centric)

```
handle_turn(message, companion: CompanionContext) →
    # Everything comes from the companion context — no session lookups
    monologue = companion.monologue.get()
    facts = companion.memories.list()
    prompt = build_prompt(companion.seed, monologue, facts, ...)
    response = model.generate(prompt)
    companion.monologue.upsert(new_state)
```

The orchestrator becomes a **pure function of (transcript + companion context
+ user message) → response**. It doesn't know or care whether there are other
companions in the session.

### Turn Router

A new component sits above the orchestrator and decides **who speaks**:

```python
class TurnRouter:
    """Decides which companions respond to a given event."""

    def route_user_message(self, message: str, session: Session) -> list[CompanionContext]:
        """Parse @mentions, or default to primary companion."""

    def route_companion_message(self, speaker: CompanionContext, response: str,
                                 session: Session) -> list[CompanionContext]:
        """Check if the response addresses other companions (reactive follow-up)."""

    def route_world_event(self, event: WorldEvent, session: Session) -> list[CompanionContext]:
        """Future: game engine events that prompt companions to act."""
```

Note: `main` currently has no multi-character routing — the orchestrator
always runs a single companion per session. `TurnRouter` is entirely new,
not a replacement for existing code. The `feat/multi` branch had a
`resolve_targets` function and reactive follow-up scan; those were early
experiments. This design supersedes them with a cleaner separation.

The turn flow becomes:

```
User message arrives
  → TurnRouter.route_user_message() → [Chloe]
  → Orchestrator.run(transcript, Chloe, message) → Chloe's response
  → TurnRouter.route_companion_message(Chloe, response) → [Marcus]
  → Orchestrator.run(transcript, Marcus, message) → Marcus's response
  → TurnRouter.route_companion_message(Marcus, response) → []
  → Done
```

## Autonomous Behaviour (Companion-Initiated Turns)

### The Tick Loop

Companions don't just react — they have their own temporal arcs. A background
**tick loop** runs periodically and gives each companion a chance to act:

```python
class CompanionTicker:
    """Periodic heartbeat that lets companions initiate interaction."""

    async def tick(self, session: Session, now: datetime):
        for companion in session.companions.values():
            decision = await self._should_act(companion, session, now)
            if decision.should_act:
                event = CompanionInitiatedEvent(
                    companion_id=companion.companion_id,
                    trigger=decision.trigger,  # "routine", "thought", "reaction"
                )
                # Feed through the same orchestrator pipeline
                response = orchestrator.run(
                    transcript=session.transcript,
                    companion=companion,
                    event=event,
                )
                session.transcript.append(response)
```

### Decision Factors

Whether a companion acts on a tick depends on:

- **Time since last interaction** — companions get restless or miss the user
- **Routine** — "Chloe goes to the lab at 9am" → she might message about it
- **Affect state** — high curiosity + low engagement → might reach out
- **World events** — location change, time-of-day triggers, NPC encounters
- **Inter-companion dynamics** — Marcus might respond to something Chloe said
  earlier, even without user prompting

```python
@dataclass
class ActDecision:
    should_act: bool
    trigger: str          # what motivated this
    urgency: float        # 0-1, used to prioritise when multiple want to act
    message_hint: str     # optional context for the prompt
```

### Prompt Adaptation for Self-Initiated Turns

When a companion initiates, the prompt is slightly different — there's no
user message to respond to. Instead:

```python
def build_self_initiated_prompt(companion: CompanionContext, trigger: str) -> str:
    """Build a prompt for companion-initiated interaction."""
    return (
        f"You are {companion.seed.companion_name}. "
        f"You haven't heard from the user in a while. "
        f"Trigger: {trigger}. "
        f"Write a natural message to the user — or to another character "
        f"if that feels more appropriate right now."
    )
```

## Game Engine Integration Points

The `WorldState` is a future extension point for linking to an external game
engine. The companion system doesn't need to implement the game engine — it
just needs to consume world state as context:

```python
@dataclass
class WorldState:
    current_time: datetime            # in-game time
    location_graph: dict[str, list[str]]  # who is where
    recent_events: list[WorldEvent]   # things that happened
    weather: str | None = None        # ambient context

@dataclass
class WorldEvent:
    timestamp: datetime
    event_type: str                   # "arrival", "departure", "incident", ...
    description: str
    involved: list[UUID]              # companion_ids involved
```

World state flows into the prompt assembly as additional context, alongside
memories and affect. The game engine publishes events; the tick loop consumes
them.

## Migration Path

### Phase 1: Companion as first-class entity (this work)
Starting from `main` which has a single-companion, session-scoped architecture:
- Add `companion_id` to schemas (Message, MemoryItem, GraphRelation, MonologueState)
- Implement `ScopedVectorStore`, `ScopedGraphStore`, `ScopedMonologueStore`
- Build `CompanionContext` dataclass and refactor orchestrator to accept it
- Build `TurnRouter` (new — no multi-character routing exists on `main`)
- DB migration: add `companion_id` columns with default for existing data
- API: `/v1/sessions/{sid}/companions/{cid}/seed`, etc.
- Backward compat: single-companion sessions work exactly as before

### Phase 2: Multi-companion sessions
- Session can hold N companions
- Shared transcript with speaker attribution
- Reactive follow-ups via `TurnRouter.route_companion_message`
- Frontend: companion selector, multi-speaker message display

### Phase 3: Autonomous behaviour
- `CompanionTicker` background loop
- Self-initiated message generation
- SSE push for companion-initiated messages (not just request-response)
- Routine/schedule support in `CompanionSeed`

### Phase 4: World integration
- `WorldState` schema and store
- Game engine event bridge
- Location-aware prompting
- Companion-to-companion interaction driven by world proximity

## Key Design Principles

1. **Companion is the unit of composition** — all cognitive state lives inside
   `CompanionContext`. The orchestrator is a stateless function.

2. **Shared transcript, private mind** — everyone sees what was said, but
   memories, affect, and monologue are per-companion.

3. **Uniform turn pipeline** — whether triggered by user message, companion
   follow-up, tick loop, or world event, the same orchestrator path runs.

4. **Scoped stores, not separate stores** — a single Qdrant collection, a
   single Postgres table, partitioned by `companion_id`. No infrastructure
   multiplication.

5. **Game engine is external** — the companion system consumes world state,
   it doesn't simulate it. Keep the boundary clean.
