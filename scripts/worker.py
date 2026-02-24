from __future__ import annotations

import signal
import time

from app.config import get_settings
from app.services import build_container

running = True


def _stop(_: int, __: object) -> None:
    global running
    running = False


def main() -> int:
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
