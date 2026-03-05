# Multi-Character Support (Option B)

Independent inference, graph, and knowledge per character. Users direct
conversation with `@Name` notation. Characters arrive organically through
prompts and responses, each with a unique name registered to the session.

---

## Phase 1: Schema & Data Model

All changes are additive with defaults, so existing single-character sessions
continue to work unchanged.

### 1a. Add `name` to `Message`

- [ ] `app/schemas.py` — `Message`: add `name: str | None = None`
- [ ] `InMemoryEpisodicStore` — no change needed (Message carries name)
- [ ] `PostgresEpisodicStore` — add `name` column to `episodic_messages`,
  include in INSERT/SELECT
- [ ] Frontend `ChatMessage` type — add optional `name` field

### 1b. Add `characters` to seed context

- [ ] `app/schemas.py` — new model:
  ```python
  class CharacterDef(BaseModel):
      name: str = Field(min_length=1)
      backstory: str = Field(min_length=1)
      character_traits: list[str] = Field(default_factory=list)
      relationship_to_companion: str = Field(default="")
  ```
- [ ] `CompanionSeed` — add `characters: list[CharacterDef] = []`
- [ ] Existing seeds with no characters field get `[]` via Pydantic default

### 1c. Key monologue/affect by `(session_id, character_name)`

- [ ] `MonologueState` — add `character_name: str | None = None`
- [ ] `InMemoryMonologueStore` — change key from `UUID` to
  `tuple[UUID, str | None]`
- [ ] `PostgresMonologueStore` — add `character_name` column, composite
  unique on `(chat_session_id, character_name)`
- [ ] All `get`/`upsert` callers pass `character_name`

### 1d. Key vector/graph stores per character

- [ ] `MemoryItem` — add `character_name: str | None = None`
- [ ] `GraphRelation` — add `character_name: str | None = None`
- [ ] In-memory stores: filter on `character_name` in query methods
- [ ] `QdrantVectorStore` — add `character_name` to payload and filters
- [ ] `Neo4jGraphStore` — add `character_name` property, update MATCH clauses

---

## Phase 2: Orchestration — Per-Character Inference

### 2a. Resolve target characters from user message

- [ ] New function in `app/services.py`:
  ```python
  def resolve_targets(
      message: str,
      primary_companion: str,
      characters: list[CharacterDef],
  ) -> list[str]:
  ```
- [ ] Parse `@Name` mentions from message text (case-insensitive match
  against registered character names)
- [ ] No `@` mention → default to primary companion only
- [ ] `@Name` → return that character (or multiple if multiple `@` mentions)
- [ ] Order: primary companion first if mentioned, then others in mention order

### 2b. Per-character `handle_turn` loop

- [ ] `CognitiveOrchestrator.handle_turn` returns
  `tuple[list[Message], dict]` instead of `tuple[Message, dict]`
- [ ] For each target character name:
  1. Build character-specific system prompt
  2. Load character-specific monologue/affect from store keyed by
     `(session_id, character_name)`
  3. Load character-specific memories and graph (filtered by
     `character_name`)
  4. Call `model_provider.generate()` with character system prompt
  5. Create `Message(role="assistant", name=character_name, ...)`
  6. Append to episodic store immediately so next character sees it
- [ ] Intent analysis runs once per turn (shared across characters)
- [ ] Retrieval runs per-character (each has own memory/graph)

### 2c. Update `_assemble_messages` for character context

- [ ] Accept `character_name` parameter
- [ ] System prompt: use `build_companion_system_prompt` for primary,
  `build_character_system_prompt` for NPCs
- [ ] Load monologue/affect for the specific character
- [ ] Load companion self-facts filtered by character
- [ ] History rendering: include `name` so model knows who said what —
  prefix content with `[CharName]: ` for multi-character messages
- [ ] Anti-repetition: filter to this character's recent responses only

### 2d. Update prompting

- [ ] `app/prompting.py` — new function:
  ```python
  def build_character_system_prompt(
      character: CharacterDef,
      seed_context: SessionSeedContext,
  ) -> str:
  ```
- [ ] Instructs model to respond as the NPC character
- [ ] Includes character's backstory, traits, relationship to companion
- [ ] Includes awareness of other active characters in the scene
- [ ] Shares the same response rules as primary companion

---

## Phase 3: Chat Service & API Layer

### 3a. Update `ChatService.run_chat`

- [ ] Handle `list[Message]` from orchestrator
- [ ] `ChatResponse` — add `assistant_messages: list[Message]`
  (keep `assistant_message` pointing to first for backward compat)
- [ ] Dispatch background agents once per responding character
- [ ] Idempotency cache: key includes sorted character names

### 3b. Update SSE streaming

- [ ] `event_generator` in `main.py` loops through each character response
- [ ] Per character: emit `character_start` event with `{"name": "..."}`,
  then `delta` chunks, then `character_done`
- [ ] Wrap with existing `start`/`done` events for the overall turn
- [ ] Event sequence:
  ```
  event: start        {session_id, request_id, ...}
  event: char_start   {name: "Luna"}
  event: delta        {chunk: "...", name: "Luna"}
  event: char_done    {name: "Luna"}
  event: char_start   {name: "Marcus"}
  event: delta        {chunk: "...", name: "Marcus"}
  event: char_done    {name: "Marcus"}
  event: done          {session_id, message_ids: [...]}
  ```

### 3c. Character management API

- [ ] `POST /v1/sessions/{id}/characters` — add character to seed's
  `characters` list, body = `CharacterDef`
- [ ] `DELETE /v1/sessions/{id}/characters/{name}` — remove character
- [ ] `GET /v1/sessions/{id}/characters` — list active characters
- [ ] These update the `SessionSeedContext` (bump version)

---

## Phase 4: Background Agents

### 4a. Per-character fact extraction & affect

- [ ] `BackgroundAgentDispatcher.enqueue_turn` — call once per character
  response with the character's name
- [ ] Fact extractor stores results with `character_name` on `MemoryItem`
- [ ] Affect refiner updates monologue keyed by `(session_id, character_name)`

### 4b. Per-character consolidation

- [ ] Track turn counts per `(session_id, character_name)`
- [ ] Consolidation runs independently per character

---

## Phase 5: Frontend

### 5a. Render named messages

- [ ] `ChatMessage` type: add `name?: string`
- [ ] Message display: show `msg.name` as speaker label instead of
  hardcoded companion label
- [ ] Color-code or style differently per character name

### 5b. `@` mention in composer

- [ ] On `@` keystroke in textarea, show autocomplete dropdown of active
  character names (from seed context)
- [ ] Pass raw `@Name` text to backend; backend parses it

### 5c. Character management UI

- [ ] New section in seed panel: "Characters"
- [ ] Add character form: name, backstory, traits, relationship
- [ ] List existing characters with remove button
- [ ] Calls character management API endpoints

### 5d. Per-character affect display

- [ ] Knowledge panel: show affect per character (tabs or dropdown)
- [ ] Each character's facts/graph shown separately

---

## Implementation Order

Sequence designed so the app is shippable after each step:

| Step | What | Shippable state |
|------|------|-----------------|
| 1 | Phase 1a+1b: Schema changes | Backward compatible, no behavior change |
| 2 | Phase 2a: `@` parsing | Returns primary only (no-op), testable |
| 3 | Phase 2b+2c+2d: Multi-character orchestration | Core feature works via API |
| 4 | Phase 3a+3b: API/SSE changes | Multiple responses stream to frontend |
| 5 | Phase 5a: Render named messages | Users see multi-character chat |
| 6 | Phase 1c+1d: Per-character store keying | Memory/affect isolation |
| 7 | Phase 4: Background agents per-character | Full knowledge pipeline |
| 8 | Phase 3c+5b+5c+5d: Management UI | Character CRUD and display |

---

## Key Design Decisions

- **Characters are session-scoped** — defined in the seed context, not global
- **Primary companion is always a character** — `companion_name` from seed
  is the first/default character
- **`@` routing is optional** — no `@` mention defaults to primary companion
- **Sequential generation** — characters respond in order, each seeing
  prior responses. No parallel generation (characters need to react to
  each other)
- **Shared episodic store, per-character everything else** — all messages
  go in one timeline, but memories/graph/affect are isolated per character
- **Same model for all characters** — different system prompts, same
  inference provider. Could be extended later to per-character model config
