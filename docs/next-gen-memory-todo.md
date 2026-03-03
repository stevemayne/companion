# Next-Generation Memory Architecture — Remaining Phases

Phases 0 (Real Embeddings), 1 (Importance Scoring), 2 (Consolidation Loop), and 4 (Adaptive Retrieval) are complete.

---

## Phase 2: Consolidation Loop — DONE

**Goal:** Periodically review recent conversation and consolidate extracted facts — reinforcing repeated mentions, superseding contradictions, and promoting durable knowledge.

### New file: `app/consolidation.py`

- [x] `ConsolidationAgent` with heuristic and LLM consolidation paths
- [x] `_heuristic_consolidate()` — token-overlap reinforcement
- [x] LLM prompt that returns reinforce/supersede/new_facts JSON
- [x] `_parse_consolidation_response()` with fenced-JSON support

### Modify: `app/agents.py`

- [x] Add `_run_consolidation()` to `BackgroundAgentDispatcher`
- [x] Trigger every N turns within a session (configurable via `consolidation_interval_turns`)
- [x] Turn counting per session in `enqueue_turn()`

### Modify: `app/services.py` + `app/store_adapters.py`

- [x] `update_memory(memory_id, importance, status)` on VectorStore protocol + both implementations
- [x] Wire `ConsolidationAgent` creation into `build_container()`

### Config: `app/config.py`

- [x] `consolidation_interval_turns: int = 10`
- [x] `consolidation_message_window: int = 20`

### Tests: `tests/test_consolidation.py` (21 tests)

- [x] Test that repeated mentions of a fact increase its importance
- [x] Test that contradictions supersede old facts (via LLM parse)
- [x] Test that consolidation is idempotent
- [x] Token overlap, heuristic consolidation, LLM response parsing, update_memory

---

## Phase 3: Temporal Awareness

**Goal:** Give the companion a sense of time — how long since the last message, what time of day it is, and what activity the user is engaged in.

### Modify: `app/schemas.py`

- [ ] Add `current_activity: str | None = None` to `MonologueState`
- [ ] Add `activity_started_at: datetime | None = None` to `MonologueState`

### Modify: `app/services.py`

- [ ] In `_assemble_messages()`: calculate time delta since last message, inject into session context
  - `"Time since last message: 3 minutes"` / `"4 hours"` / `"2 days"`
  - `"Current time: evening (8:47 PM)"`
- [ ] Activity extraction: extend `_extract_user_state()` to detect shared activities ("Let's cook dinner", "I'm watching a movie")
- [ ] Activity decay: if time gap > 2 hours, clear `current_activity`

### Modify: `app/prompting.py`

- [ ] Add temporal context section to the session context block

**Note:** This phase is independent of Phases 2 and 4 and can be implemented any time.

---

## Phase 4: Adaptive Retrieval — DONE

**Goal:** Instead of always retrieving memories, use a lightweight decision layer to skip retrieval for trivial messages.

### New file: `app/retrieval.py`

- [x] `RetrievalDecision` dataclass with `should_retrieve`, `rewritten_query`, `reason`, `latency_ms`
- [x] `HeuristicRetrievalDecider` — skips greetings/filler/short messages, no LLM call
- [x] `LLMRetrievalDecider` — lightweight LLM prompt with query rewriting, falls back to heuristic
- [x] `_parse_retrieval_response()` — JSON parsing with fenced-code support

### Modify: `app/services.py`

- [x] `CognitiveOrchestrator` gains `retrieval_decider` field (default: heuristic)
- [x] `handle_turn()` checks decision before vector/graph retrieval
- [x] Retrieval decision logged in debug traces (`retrieval.decision`)
- [x] `build_container()` wires LLM decider when `adaptive_retrieval=True` + LLM analysis provider

### Config: `app/config.py`

- [x] `adaptive_retrieval: bool = False` (off by default)

### Tests: `tests/test_retrieval.py` (48 tests)

- [x] Heuristic skips greetings, filler, short messages (parametrized)
- [x] Heuristic retrieves for substantive messages, questions, proper nouns
- [x] LLM response parsing (retrieve/skip, fenced JSON, defaults)
- [x] LLM decider with mock provider, fallback on error and bad JSON

---

## Phase 5: Graph Refactoring + Forgetting

**Goal:** Reduce graph noise, add memory decay, and allow user-initiated forgetting.

### Graph narrowing

- [ ] Remove `MENTIONED_IN_SESSION` writes from `_postprocess()` (low-value noise)
- [ ] Entity relationships from extraction go to vector store as semantic memories with structured metadata
- [ ] Graph reserved for multi-hop relational queries only (optional, loaded by consolidation)

### Forgetting

- [ ] New `_run_decay()` background job at session start
- [ ] Archives memories where `importance * recency_factor < 0.05`
- [ ] New API: `DELETE /v1/memory/{session_id}/{memory_id}` for user-initiated forgetting
- [ ] Config: `memory_decay_lambda: float = 0.01`, `memory_archive_threshold: float = 0.05`

---

## Remaining

```
Phase 3: Temporal Awareness       ← independent, can do any time
    ↓
Phase 5: Graph + Forgetting       ← next up
```

## Key Files

| File | Phase | Status |
|------|-------|--------|
| `app/embedding.py` | 0 | Done |
| `app/config.py` | 0, 1, 2, 4 | Done |
| `app/store_adapters.py` | 0, 1, 2 | Done |
| `app/services.py` | 0, 1, 2, 4 | Done |
| `app/schemas.py` | 1, 3 | Phase 1 done, Phase 3 pending |
| `app/agents.py` | 1, 2 | Done |
| `app/analysis.py` | 1 | Done |
| `app/consolidation.py` | 2 | Done |
| `app/retrieval.py` | 4 | Done |
| `app/prompting.py` | 3 | Pending |
