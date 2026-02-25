from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def force_mock_inference(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
