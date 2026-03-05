from __future__ import annotations

import pytest

import app.inference as _inference_mod
from app.config import get_settings


@pytest.fixture(autouse=True)
def isolate_from_dotenv(monkeypatch: pytest.MonkeyPatch):
    """Prevent .env from leaking into tests.

    By pointing the env file at a nonexistent path, pydantic-settings
    falls back to class defaults (which are already safe: mock inference,
    in-memory stores, heuristic analysis, mock embeddings).  Tests that
    pass explicit kwargs to Settings() get exactly what they ask for.
    """
    monkeypatch.setenv("ENV_FILE", "/dev/null")
    monkeypatch.setattr("app.config.Settings.model_config", {"env_file": ""})
    # Disable per-session inference log files during tests
    monkeypatch.setattr(_inference_mod, "LOGS_DIR", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
