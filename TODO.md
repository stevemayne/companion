# Project Aether TODO

## Next Steps Checklist

- [x] 1. Project Bootstrap
  - [x] Create repo structure: `app/`, `tests/`, `alembic/`, `scripts/`, `docker/`
  - [x] Set up FastAPI app with health endpoint, config, logging, and DI baseline
  - [x] Add toolchain: dependency manager, lint, type-check, tests, pre-commit, `.env.example`
  - [x] Deliverable: runnable API skeleton + CI lint/test pipeline
- [x] 2. Core Domain Contracts
  - [x] Define Pydantic schemas for messages, memory items, entities, graph relations, monologue state
  - [x] Add required `chat_session_id` across all context-bearing schemas and contracts
  - [x] Define session seed schemas (companion identity, backstory, traits, goals, relationship setup)
  - [x] Define ports/interfaces for providers and stores
  - [x] Deliverable: stable swappable contracts
- [x] 3. MVP API Surface
  - [x] Implement `POST /v1/chat`, `GET /v1/health`, `GET /v1/memory/{chat_session_id}`
  - [x] Add session lifecycle endpoints for creating/updating session seed context
  - [x] Require `chat_session_id` in request path/payload for context reads and writes
  - [x] Add validation, idempotency support, and error model
  - [x] Deliverable: end-to-end chat path with mocked dependencies
- [x] 4. Orchestrator + Cognitive Loop
  - [x] Implement preprocess, retrieval, context assembly, inference, and response path
  - [x] Persist `internal_monologue`
  - [x] Deliverable: working loop with deterministic tests
- [x] 5. Memory Layer Integration
  - [x] Episodic store (PostgreSQL)
  - [x] Semantic store (pgvector or Qdrant)
  - [x] Reflective store (Neo4j)
  - [x] Enforce session partitioning in all memory writes and retrieval queries
  - [x] Persist and version session seed context per `chat_session_id`
  - [x] Deliverable: hybrid retrieval with fallbacks
- [x] 6. Inference Gateway
  - [x] Add local and cloud provider adapters
  - [x] Add required config for `INFERENCE_MODEL` and `INFERENCE_BASE_URL` (remote LM Studio compatible)
  - [x] Add env-based routing + retry/timeout/failover policy
  - [x] Deliverable: pluggable inference
- [x] 7. Async Agents
  - [x] Add Extraction and Reflector background jobs
  - [x] Deliverable: non-blocking post-processing
- [x] 8. Observability + Safety
  - [x] Add metrics/tracing/logging/correlation IDs
  - [x] Add guardrails, redaction, auth, and rate limiting
  - [x] Deliverable: operable and safer baseline
- [x] 9. Testing + Evaluation
  - [x] Add unit + integration tests and eval set
  - [x] Add tests proving strict cross-session isolation (no context leakage)
  - [x] Add tests proving seeded context is applied from first turn and is version-editable per session
  - [x] Deliverable: regression and release criteria
- [x] 10. Deployment
  - [x] Containerize API + workers
  - [x] Add migrations, runbooks, dashboards, rollback plan
  - [x] Deliverable: staging to production release

## In Progress

- [x] Step 10 completed

## Next Phase: Frontend + Streaming

- [ ] 11. React Frontend + SSE
  - [x] Scaffold React + TypeScript app in `web/`
  - [x] Build chat UI with session management and message streaming UX
  - [x] Implement SSE backend endpoint for streamed chat responses
  - [x] Wire frontend SSE client to render incremental assistant output
  - [ ] Add reconnect/error handling for stream interruptions
  - [ ] Add backend and frontend tests for SSE behavior
  - [ ] Deliverable: browser chat app receiving streamed responses from FastAPI via SSE

## Next Phase: Companion UX + Prompting

- [ ] 12. Stronger Companion Prompt Layer
  - [ ] Implement a dedicated companion persona/system prompt builder from session seed fields
  - [ ] Ensure prompt always uses seeded companion name (no generic "Assistant" fallback)
  - [ ] Include relational context (relationship setup, goals, tone constraints) on every turn
  - [ ] Add tests validating seeded identity/tone continuity across turns
  - [ ] Deliverable: consistent companion-style responses tied to session seed

- [ ] 13. UI Seeding + Configuration Defaults
  - [ ] Add frontend seed form (name, backstory, traits, goals, relationship setup)
  - [ ] Preload sensible defaults for quick session start
  - [ ] Call `POST /v1/sessions/{chat_session_id}/seed` on new session bootstrap
  - [ ] Add UI flow to update seed via `PUT /v1/sessions/{chat_session_id}/seed`
  - [ ] Persist selected defaults in UI state for reuse
  - [ ] Add frontend tests for seed create/update request flows
  - [ ] Deliverable: users can configure companion profile directly from the UI
