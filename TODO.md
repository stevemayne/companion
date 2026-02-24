# Project Aether TODO

## Next Steps Checklist

- [x] 1. Project Bootstrap
  - [x] Create repo structure: `app/`, `tests/`, `alembic/`, `scripts/`, `docker/`
  - [x] Set up FastAPI app with health endpoint, config, logging, and DI baseline
  - [x] Add toolchain: dependency manager, lint, type-check, tests, pre-commit, `.env.example`
  - [x] Deliverable: runnable API skeleton + CI lint/test pipeline
- [ ] 2. Core Domain Contracts
  - [ ] Define Pydantic schemas for messages, memory items, entities, graph relations, monologue state
  - [ ] Add required `chat_session_id` across all context-bearing schemas and contracts
  - [ ] Define session seed schemas (companion identity, backstory, traits, goals, relationship setup)
  - [ ] Define ports/interfaces for providers and stores
  - [ ] Deliverable: stable swappable contracts
- [ ] 3. MVP API Surface
  - [ ] Implement `POST /v1/chat`, `GET /v1/health`, `GET /v1/memory/{user_id}`
  - [ ] Add session lifecycle endpoints for creating/updating session seed context
  - [ ] Require `chat_session_id` in request path/payload for context reads and writes
  - [ ] Add validation, idempotency support, and error model
  - [ ] Deliverable: end-to-end chat path with mocked dependencies
- [ ] 4. Orchestrator + Cognitive Loop
  - [ ] Implement preprocess, retrieval, context assembly, inference, and response path
  - [ ] Persist `internal_monologue`
  - [ ] Deliverable: working loop with deterministic tests
- [ ] 5. Memory Layer Integration
  - [ ] Episodic store (PostgreSQL)
  - [ ] Semantic store (pgvector or Qdrant)
  - [ ] Reflective store (Neo4j)
  - [ ] Enforce session partitioning in all memory writes and retrieval queries
  - [ ] Persist and version session seed context per `chat_session_id`
  - [ ] Deliverable: hybrid retrieval with fallbacks
- [ ] 6. Inference Gateway
  - [ ] Add local and cloud provider adapters
  - [ ] Add env-based routing + retry/timeout/failover policy
  - [ ] Deliverable: pluggable inference
- [ ] 7. Async Agents
  - [ ] Add Extraction and Reflector background jobs
  - [ ] Deliverable: non-blocking post-processing
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

- [ ] Step 2 currently in progress
