This requirements document outlines a State-of-the-Art (SOTA) AI Companion architecture for 2026. It is designed to be **pluggable**, **memory-persistent**, and **context-aware** using a hybrid Vector + Graph retrieval strategy.

---

## 1. Project Overview

**Project Name:** *Project Aether* **Objective:** Develop a conversational AI companion capable of maintaining a multi-year "relationship" through hierarchical memory, emotional state tracking, and cross-platform accessibility.

---

## 2. System Architecture (The "Cognitive OS" Pattern)

The system follows a **Modular Agentic Loop**. Instead of a direct User → LLM pipe, it uses an orchestrator to manage state and memory retrieval before generating a response.

### High-Level Modules

1. **Orchestrator (The Brain):** Manages the state machine and routes data between modules.
2. **Memory Controller:** The interface for Episodic (Logs), Semantic (Vectors), and Reflective (Graph) storage.
3. **Extraction Agent:** A background process that distills each conversation turn into structured facts (subject/predicate/object triples stored in the vector DB) and entity mentions with relationship labels and aliases (stored as typed graph edges for alias-aware retrieval).
4. **Reflector Agent:** An asynchronous job that periodically analyzes the "health" and "history" of the relationship to update the AI's internal monologue.
5. **Inference Gateway:** A pluggable connector for Local (Ollama/LM Studio) or Cloud (Claude/OpenAI) models.

---

## 3. Technology Stack (2026 Developer Standard)

| Layer | Component | Choice (Recommended) |
| --- | --- | --- |
| **Orchestration** | Python-based Framework | **LangGraph** (Best for stateful, cyclic agents) |
| **Inference (Local)** | 5090 Host / Mac Client | **LM Studio** (server mode) or **Ollama** |
| **Inference (Cloud)** | API Provider | **Anthropic (Claude 3.5/4)** via LiteLLM |
| **Episodic Memory** | Time-series Database | **Redis** or **PostgreSQL (pgvector)** |
| **Semantic Memory** | Vector Database | **Qdrant** or **ChromaDB** |
| **Reflective Memory** | Graph Database | **Neo4j** (Standard for GraphRAG) |
| **Communication** | API Standard | **OpenAI / Anthropic SDK** (Cross-compatible) |

---

## 4. Functional Requirements & Responsibilities

### R1: Memory Hierarchy

* **Episodic:** Stores the last 50 messages per session in raw text for immediate conversational flow.
* **Semantic:** Stores structured facts extracted from each turn as (subject, predicate, object, text) triples. Retrieved via vector similarity at query time.
* **Reflective (Graph):** Maps typed relationships and aliases. Entity relationships are stored as edges like `user -HAS_SISTER-> Sarah`. Nicknames and alternate names are stored as `Sarah -ALSO_KNOWN_AS-> sis`. At query time, alias resolution expands the entity set through `ALSO_KNOWN_AS` edges before fetching all relations, so mentioning "sis" in a later turn surfaces everything known about Sarah.

### R2: Pluggable Inference

* The system must use a unified `BaseModel` class.
* Inference must be configurable via environment variables for both `INFERENCE_MODEL` and `INFERENCE_BASE_URL` (for example a remote LM Studio server endpoint).
* **Development Mode:** Defaults to a local LM Studio-compatible endpoint.
* **Production Mode:** Can switch to a cloud endpoint via environment variables.

### R3: The "Inner Monologue"

* The system must maintain a hidden `internal_monologue` string that persists between turns. This allows the AI to "plan" its emotional response before speaking.

### R4: Session-Scoped Context Isolation

* All conversational context must be scoped to a `chat_session_id` so multiple chats can run in parallel with different histories, memories, and behavioral setup.
* Episodic logs, semantic vectors, reflective graph nodes/edges, and `internal_monologue` must all be partitioned by session and never bleed across sessions unless explicitly linked by a future cross-session feature.
* Retrieval and writes must always include session scope as a required filter/key.

### R5: Session Context Seeding

* A chat session must support pre-seeding context before the first user message (for example: companion identity, backstory, personality traits, goals, and relationship setup).
* Seeded context must be stored as session-scoped memory and injected into retrieval/context assembly so behavior is consistent from turn one.
* Session seeding must be editable/versioned so a session can be configured or refined without affecting other sessions.

---

## 5. Typical Message Flow (The "Cognitive Loop")

When a user sends: *"I'm heading to Sarah's house for dinner, I'm pretty nervous. Sis always makes me anxious."*

### 5.1 Pre-processing (Intent Analysis)

The `IntentAnalyzer` classifies the message (LLM-based with heuristic fallback):

* **Intent:** `status_update`
* **Emotion:** `anxious`
* **Entities:** `["Sarah"]`

Implementation: `app/analysis.py` — `LLMIntentAnalyzer` sends the message to the analysis LLM, which returns structured JSON. On failure, `HeuristicIntentAnalyzer` uses keyword matching and capitalized-token extraction.

### 5.2 Retrieval (Hybrid Search with Alias Resolution)

The `CognitiveOrchestrator` runs two retrieval paths in parallel:

* **Vector Search:** Queries Qdrant for semantic similarity against `"Sarah"` and `"dinner"`. Returns stored facts like *"User's sister Sarah started a new job"*.
* **Graph Walk (two-pass alias resolution):**
  1. **Pass 1 — Expand aliases:** For each entity, query for `ALSO_KNOWN_AS` edges. If `Sarah -ALSO_KNOWN_AS-> sis` exists, both `"Sarah"` and `"sis"` enter the expanded set.
  2. **Pass 2 — Fetch relations:** Query all relations for the expanded entity set. Returns typed edges like `user -HAS_SISTER-> Sarah`, `user -MENTIONED_IN_SESSION-> Sarah`.

This means if the user previously said *"my sister Sarah, we call her sis"*, mentioning just *"sis"* in a later message will resolve to Sarah and surface all her relationships.

Implementation: `app/services.py` — `CognitiveOrchestrator._graph_context()`

### 5.3 Context Assembly

The orchestrator builds a multi-turn prompt:

```
System: {companion persona + response rules}
        ## Session Context (internal)
        Internal reflection: {monologue from last turn}
        Relevant memories: {vector search results}
        Relationships: user-HAS_SISTER->Sarah | Sarah-ALSO_KNOWN_AS->sis
        Detected intent: status_update; emotion: anxious

History: {last 4 messages from episodic store}
User:    "I'm heading to Sarah's house for dinner..."
```

Implementation: `app/services.py` — `CognitiveOrchestrator._assemble_messages()`

### 5.4 Inference

The LLM generates a response: *"I remember things were tense with her last time. Do you want to talk about why you're nervous?"*

The response passes through `_enforce_seeded_identity()` to ensure the companion never calls itself "Assistant".

### 5.5 Synchronous Post-processing

Immediately after inference, the orchestrator writes to the graph and updates the monologue:

* **Graph writes:** `user -MENTIONED_IN_SESSION-> Sarah` (entities from both LLM analysis and a heuristic safety net that extracts capitalized tokens).
* **Monologue update:** `"Focus on a anxious user; intent=status_update; entities=Sarah"`

Implementation: `app/services.py` — `CognitiveOrchestrator._postprocess()`

### 5.6 Asynchronous Background Agents

The `BackgroundAgentDispatcher` fires two jobs on a thread pool:

**Extraction Agent** — Extracts structured facts and entities from the turn using an LLM (with heuristic fallback):

* **Structured facts** as (subject, predicate, object, text) triples:
  * `{subject: "User", predicate: "is heading to", object: "Sarah's house for dinner", text: "User is heading to Sarah's house for dinner"}`
* **Entity mentions** with relationships and aliases:
  * `{name: "Sarah", relationship: "sister", aliases: ["sis"]}`
* Facts are stored in the **vector store** (Qdrant) for semantic retrieval.
* Entity relationships become **typed graph edges**: `user -HAS_SISTER-> Sarah`
* Aliases become **graph edges**: `Sarah -ALSO_KNOWN_AS-> sis`

Implementation: `app/analysis.py` — `LLMFactExtractor` / `HeuristicFactExtractor`, `app/agents.py` — `BackgroundAgentDispatcher._run_extraction()`

**Reflector Agent** — Summarizes the last 3 turns and appends to the internal monologue for next-turn context.

Implementation: `app/agents.py` — `BackgroundAgentDispatcher._run_reflector()`

### 5.7 Data Flow Diagram

```
User Message
     │
     ▼
┌─────────────────┐
│ Intent Analyzer  │──▶ intent, emotion, entities
│ (LLM/heuristic) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         CognitiveOrchestrator       │
│                                     │
│  ┌──────────┐    ┌───────────────┐  │
│  │  Vector   │    │  Graph Walk   │  │
│  │  Search   │    │ (alias-aware) │  │
│  └────┬─────┘    └──────┬────────┘  │
│       │                 │           │
│       ▼                 ▼           │
│  ┌──────────────────────────────┐   │
│  │     Context Assembly         │   │
│  │  system + monologue + memory │   │
│  │  + graph + history + user    │   │
│  └──────────────┬───────────────┘   │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐   │
│  │     Inference Gateway        │   │
│  │   (LM Studio / Claude / …)  │   │
│  └──────────────┬───────────────┘   │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐   │
│  │    Post-process (sync)       │   │
│  │  graph writes + monologue    │   │
│  └──────────────────────────────┘   │
└────────────────┬────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
┌──────────┐          ┌────────────┐
│Extraction│          │ Reflector  │
│  Agent   │          │   Agent    │
│ (async)  │          │  (async)   │
│          │          │            │
│ facts ──▶│Vector   │ monologue  │
│ entities▶│ Graph    │ update     │
└──────────┘          └────────────┘
```

---

## 6. Evaluation & Release Gates

Run the baseline evaluation suite:

```bash
uv run python scripts/run_eval.py
```

Release criteria checklist:

- See `docs/release_criteria.md` for required quality gates and behavior checks.

## 7. Command Reference

### Environment Setup

```bash
cp .env.example .env
uv sync --all-groups
```

### Lifecycle Commands

```bash
# start infrastructure + app + worker
docker compose up -d --build

# view status
docker compose ps

# stream logs
docker compose logs -f api worker postgres qdrant neo4j redis

# stop all services
docker compose down
```

### Local API + Worker Commands

```bash
# run API locally
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# run worker locally
uv run python scripts/worker.py
```

### Migration Commands

```bash
# current migration head
uv run alembic heads

# apply all migrations
uv run alembic upgrade head

# if schema already exists before Alembic tracking, mark current revision as applied
uv run alembic stamp head

# downgrade one revision
uv run alembic downgrade -1

# create a new migration
uv run alembic revision -m "describe change"
```

### Testing Commands

```bash
# lint + typecheck
uv run ruff check .
uv run mypy app

# unit/integration tests (default)
uv run pytest

# SSE endpoint tests
uv run pytest tests/test_sse.py

# external store integration test
RUN_INTEGRATION=1 uv run pytest tests/test_external_stores_integration.py

# baseline eval suite
uv run python scripts/run_eval.py

# quick smoke checks
uv run pytest tests/test_health.py tests/test_api.py
```

### Health and Metrics

```bash
curl -s http://localhost:8000/v1/health
curl -s http://localhost:8000/metrics
```

## 8. Operations

- Deployment checklist: `docs/deployment_checklist.md`
- Runbook, dashboards, rollback: `docs/operations_runbook.md`

## 9. Frontend + SSE

### Frontend Lifecycle Commands

```bash
cd web
npm install
npm run dev
```

Build frontend:

```bash
cd web
npm run build
```

### SSE Endpoint

Route:

- `GET /v1/chat/stream?chat_session_id=<uuid>&message=<text>`

Response media type:

- `text/event-stream`

Event contract:

- `start`: stream metadata (`chat_session_id`, `message_id`, `request_id`, `seed_version`)
- `delta`: incremental chunk payload (`chunk`)
- `done`: stream completed (`chat_session_id`, `message_id`)

Example:

```bash
curl -N \"http://localhost:8000/v1/chat/stream?chat_session_id=00000000-0000-0000-0000-000000000001&message=hello\"
```
