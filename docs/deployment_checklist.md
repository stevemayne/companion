# Deployment Checklist

- [ ] `docker compose up -d --build`
- [ ] `uv run alembic upgrade head`
- [ ] `curl http://localhost:8000/v1/health`
- [ ] `uv run pytest tests/test_health.py tests/test_api.py`
- [ ] Verify `/metrics` and logs
- [ ] Confirm inference endpoint configuration (`INFERENCE_MODEL`, `INFERENCE_BASE_URL`)
