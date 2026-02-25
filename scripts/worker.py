from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

running = True


def _stop(_: int, __: object) -> None:
    global running
    running = False


def main() -> int:
    from app.config import get_settings
    from app.services import build_container

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    settings = get_settings()
    container = build_container(settings)

    print("worker.started")
    try:
        while running:
            # Placeholder heartbeat worker; extraction/reflector currently enqueue in-process.
            time.sleep(5)
            print("worker.heartbeat")
    finally:
        container.agent_dispatcher.shutdown()
        print("worker.stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
