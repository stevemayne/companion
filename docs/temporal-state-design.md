# Temporal State: Companion World Model

## Core Idea

Each companion is a game entity that maintains its own subjective world model. The companion perceives the scene through conversation — what's been said and narrated is what they know. Their world state is first-person: "I see the user wearing X, we're in Y, I'm doing Z."

This is the companion's simulation state, stored inside their object. It's not a shared global truth — it's what *this character* believes to be true right now.

## The Problem Today

Facts are extracted into a flat store with no temporal awareness. "User is wearing a blue dress" and "User changed into pajamas" both exist as active facts. The system has no mechanism to know the first one is stale.

Mixing transient scene state (clothing, position, activity) with durable facts (relationships, preferences, history) in the same store is the root cause. These are fundamentally different kinds of knowledge with different lifecycles.

## Design: Companion World State

### Schema

```python
class CharacterState(BaseModel):
    """Physical/locational state of a character as perceived by the observer."""
    clothing: str | None = None         # "red sundress", "pajamas"
    location: str | None = None         # "living room", "kitchen"
    activity: str | None = None         # "cooking dinner", "reading"
    position: str | None = None         # "sitting on the couch", "standing by the window"
    appearance: list[str] = []          # notable temporary features
    mood_apparent: str | None = None    # how they APPEAR to the observer (not internal)

class WorldState(BaseModel):
    """A companion's subjective perception of the current scene."""
    self_state: CharacterState = CharacterState()
    user_state: CharacterState = CharacterState()
    other_characters: dict[str, CharacterState] = {}  # name -> state
    environment: str | None = None      # "cozy apartment, evening light"
    time_of_day: str | None = None      # "evening", "late night"
    recent_events: list[str] = []       # last few notable things that happened
```

### Where It Lives

`WorldState` lives inside `CompanionContext`, persisted alongside the monologue/affect state. It's the companion's private simulation — scoped to `(session_id, companion_id)`.

```python
class MonologueState(BaseModel):
    chat_session_id: UUID
    companion_id: UUID | None = None
    internal_monologue: str = ""
    affect: CompanionAffect = CompanionAffect()
    world: WorldState = WorldState()           # NEW — replaces user_state
    # user_state: list[str]                    # DEPRECATED
    updated_at: datetime
```

### How It Updates

The **reflector** already runs every turn and processes recent conversation. Expand its output to include structured world state:

```
Current reflector output:
  affect, user_state (list[str]), internal_monologue

New reflector output:
  affect, world (WorldState), internal_monologue
```

The reflector prompt becomes:

> Given the recent conversation, update the companion's world model.
> For each character (self, user, others), update what the companion
> CURRENTLY perceives about their clothing, location, activity, position,
> and apparent mood. Drop anything that's no longer true. Only include
> what the companion would reasonably know from the conversation.

Key principle: **latest turn wins for mutable state**. If the user says "I changed into pajamas", the reflector sets `user_state.clothing = "pajamas"` and the old value is simply gone. No deprecation logic needed — it's a state machine, not a log.

### How It's Consumed

The system prompt builder reads `companion.world` and renders it:

```
## Current Scene
You are in the living room. It's evening.
You are wearing a blue cardigan and jeans, sitting on the couch.
The user is wearing pajamas, lying on the other end of the couch.
Emma is visiting — she's in the kitchen with Rex.
```

This replaces the current `_build_user_context_block(user_state)`.

### What Changes for Extraction

The extraction prompt gets an explicit exclusion:

> Do NOT extract as facts: current clothing, physical position, current
> activity, current location, or temporary physical appearance. These are
> tracked as scene state, not durable facts.
>
> DO extract: relationships, preferences, plans, events, traits, identity,
> skills, opinions, projects.

This prevents transient state from polluting the fact store.

## Knowledge Architecture After This Change

```
CompanionContext
├── seed (identity, backstory, traits)
├── world: WorldState              ← NEW: mutable scene state (latest wins)
│   ├── self_state: CharacterState
│   ├── user_state: CharacterState
│   ├── other_characters: dict
│   ├── environment, time_of_day
│   └── recent_events
├── monologue (internal thoughts)
├── affect (emotional state)
├── memories: ScopedVectorStore    ← durable facts only (no transient state)
└── graph: ScopedGraphStore        ← entity relationships
```

**Mutable state** (clothing, location, activity) → `WorldState`, updated every turn, latest wins.
**Durable knowledge** (relationships, preferences, history) → fact store + graph, superseded only by consolidation.

## Toward a Game Engine

This design points toward each companion being a proper game entity:

1. **Perception** — the companion only knows what's been narrated/said in their presence
2. **State** — each companion maintains their own world model independently
3. **Multiple companions** — in a scene with two companions, each has their own `WorldState` and might perceive things differently (one was in the room, the other wasn't)
4. **Persistence** — `WorldState` persists across reconnections within a session; durable facts persist across sessions
5. **Scene transitions** — "Let's go to the bedroom" updates `environment` and `location` for both characters; previous room state is simply replaced

Future extensions:
- **Inventory**: items the character is holding or has nearby
- **Knowledge asymmetry**: what one character knows vs another (secrets, surprises)
- **Time tracking**: elapsed time, scheduled events ("the demo is next Friday")
- **Spatial awareness**: who can see/hear whom, room layouts
- **NPC state**: other characters in the scene with their own `CharacterState`

## Implementation Order

1. Add `CharacterState` and `WorldState` to `app/schemas.py`
2. Add `world: WorldState` to `MonologueState` (keep `user_state` for backward compat initially)
3. Update the reflector prompt to output structured world state
4. Update `_parse_state_response` to parse `WorldState` from LLM output
5. Update system prompt builder to render scene from `WorldState`
6. Update extraction prompt to exclude transient physical state
7. Deprecate `user_state` field
8. Add `WorldState` to the knowledge API response
9. Update frontend to display scene state
