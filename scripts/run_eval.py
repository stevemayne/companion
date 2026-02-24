from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_eval(dataset_path: Path) -> int:
    from app.main import create_app

    cases = json.loads(dataset_path.read_text())
    client = TestClient(create_app())

    passed = 0
    for case in cases:
        session_id = str(uuid4())

        seed = case.get("seed")
        if seed:
            response = client.post(
                f"/v1/sessions/{session_id}/seed",
                json={"seed": seed, "notes": "eval"},
            )
            if response.status_code not in (200, 201):
                print(f"FAIL {case['name']}: unable to seed session")
                continue

        last_content = ""
        success = True
        for message in case["messages"]:
            response = client.post(
                "/v1/chat",
                json={"chat_session_id": session_id, "message": message},
            )
            if response.status_code != 200:
                success = False
                print(f"FAIL {case['name']}: chat request failed ({response.status_code})")
                break
            last_content = response.json()["assistant_message"]["content"]

        if not success:
            continue

        for expected in case.get("must_contain", []):
            if expected not in last_content:
                success = False
                print(f"FAIL {case['name']}: missing '{expected}'")

        for forbidden in case.get("must_not_contain", []):
            if forbidden in last_content:
                success = False
                print(f"FAIL {case['name']}: contains forbidden '{forbidden}'")

        if success:
            passed += 1
            print(f"PASS {case['name']}")

    total = len(cases)
    print(f"\nEvaluation: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run_eval(Path("evals/baseline_chat_eval.json")))
