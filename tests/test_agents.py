from __future__ import annotations

import time
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.main import create_app


def test_background_agents_process_jobs_non_blocking() -> None:
    app = create_app()
    client = TestClient(app)
    session_id = str(uuid4())

    start = time.perf_counter()
    response = client.post(
        "/v1/chat",
        json={"chat_session_id": session_id, "message": "I met Sarah at dinner."},
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 1.0

    session_uuid = UUID(session_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        metrics = app.state.container.agent_dispatcher.get_metrics(chat_session_id=session_uuid)
        if metrics.extraction_jobs >= 1 and metrics.reflector_jobs >= 1:
            break
        time.sleep(0.05)

    metrics = app.state.container.agent_dispatcher.get_metrics(chat_session_id=session_uuid)
    assert metrics.extraction_jobs >= 1
    assert metrics.reflector_jobs >= 1
    assert metrics.failures == 0
