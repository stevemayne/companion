from __future__ import annotations

from pathlib import Path

from scripts.run_eval import run_eval


def test_baseline_eval_suite_passes() -> None:
    dataset = Path("evals/baseline_chat_eval.json")
    assert run_eval(dataset) == 0
