# Operations Runbook

## Deployment Lifecycle

1. Build and start stack:
- `docker compose up -d --build`

2. Check service health:
- `docker compose ps`
- `curl -s http://localhost:8000/v1/health`

3. Apply DB migrations:
- `uv run alembic upgrade head`

4. Tail logs:
- `docker compose logs -f api worker`

5. Stop stack:
- `docker compose down`

## Dashboard Signals

Use `GET /metrics` and track:

- `aether_http_requests_total` by `status`
- `aether_http_request_duration_seconds` p95 and p99
- `http_429` rate limit responses
- `http_401` auth failures
- chat endpoint error rates (`/v1/chat`, status >= 400)

## Rollback Plan

1. Roll back application containers:
- deploy previous image tag for `api` and `worker`
- restart services and verify `/v1/health`

2. Roll back DB schema one revision if needed:
- `uv run alembic downgrade -1`

3. Verify system after rollback:
- run smoke tests (`uv run pytest tests/test_health.py tests/test_api.py`)
- verify metrics and error rates return to baseline
