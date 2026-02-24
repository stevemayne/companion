# Release Criteria

A build is release-candidate ready when all criteria below are true:

1. Static quality gates pass:
- `uv run ruff check .`
- `uv run mypy app`

2. Test suite passes:
- `uv run pytest`
- `RUN_INTEGRATION=1 uv run pytest tests/test_external_stores_integration.py`

3. Evaluation suite passes:
- `uv run python scripts/run_eval.py`

4. Critical behavior checks are green:
- strict cross-session isolation (seed, memory, monologue)
- seeded context applies from first turn
- seed context supports versioned updates per session
- guardrails remain active (prompt-injection block, PII redaction)
