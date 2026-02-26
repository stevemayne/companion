from __future__ import annotations

from uuid import uuid4

from app.analysis import HeuristicIntentAnalyzer, LLMIntentAnalyzer


class _JsonProvider:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, *, chat_session_id: object, prompt: str) -> str:
        del chat_session_id, prompt
        return self.payload


def test_heuristic_analyzer_ignores_question_words_for_entities() -> None:
    analyzer = HeuristicIntentAnalyzer()

    outcome = analyzer.analyze(chat_session_id=uuid4(), content="What are you wearing?")

    assert outcome.preprocess.intent == "question"
    assert outcome.preprocess.entities == []
    assert outcome.used_provider == "heuristic"


def test_llm_analyzer_uses_structured_json_when_valid() -> None:
    analyzer = LLMIntentAnalyzer(
        provider=_JsonProvider(
            '{"intent":"question","emotion":"neutral","entities":["Sarah","What"]}'
        ),
        fallback=HeuristicIntentAnalyzer(),
    )

    outcome = analyzer.analyze(chat_session_id=uuid4(), content="What did Sarah say?")

    assert outcome.used_provider == "llm"
    assert outcome.fallback_reason is None
    assert outcome.preprocess.intent == "question"
    assert outcome.preprocess.entities == ["Sarah"]


def test_llm_analyzer_falls_back_on_invalid_output() -> None:
    analyzer = LLMIntentAnalyzer(
        provider=_JsonProvider("not-json-at-all"),
        fallback=HeuristicIntentAnalyzer(),
    )

    outcome = analyzer.analyze(chat_session_id=uuid4(), content="I feel anxious about dinner.")

    assert outcome.requested_provider == "llm"
    assert outcome.used_provider == "heuristic"
    assert outcome.fallback_reason is not None
    assert outcome.preprocess.intent == "status_update"
