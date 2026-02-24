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
- [ ] 8. Observability + Safety
  - [ ] Add metrics/tracing/logging/correlation IDs
  - [ ] Add guardrails, redaction, auth, and rate limiting
  - [ ] Deliverable: operable and safer baseline
- [ ] 9. Testing + Evaluation
  - [ ] Add unit + integration tests and eval set
  - [ ] Add tests proving strict cross-session isolation (no context leakage)
  - [ ] Add tests proving seeded context is applied from first turn and is version-editable per session
  - [ ] Deliverable: regression and release criteria
- [ ] 10. Deployment
  - [ ] Containerize API + workers
  - [ ] Add migrations, runbooks, dashboards, rollback plan
  - [ ] Deliverable: staging to production release

## In Progress

- [ ] Step 8 currently in progress
