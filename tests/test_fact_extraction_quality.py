"""Fixture-based quality tests for fact extraction.

Each case in the fixture defines an input conversation turn and the expected
structured facts. The test validates that:
  1. Expected facts survive the validate_facts() filter.
  2. No forbidden subjects appear after validation.
  3. The heuristic extractor produces facts with correct subject attribution.
  4. Expected entities have valid structure for graph storage.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.analysis import (
    EntityMention,
    ExtractedFact,
    HeuristicFactExtractor,
    validate_facts,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fact_extraction_cases.json"


def _load_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Validation filter tests — verify expected facts survive filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_expected_facts_survive_validation(case: dict) -> None:
    expected = case["expected_facts"]
    companion_name = case.get("companion_name")
    must_not_subjects = set(case.get("must_not_have_subject", []))

    structured = []
    for ef in expected:
        structured.append(
            ExtractedFact(
                subject=ef.get("subject", "User"),
                predicate=ef.get("predicate", ""),
                object=ef.get("object", ""),
                text=ef.get("text", ef.get("text_contains", "")),
            )
        )

    validated = validate_facts(structured, companion_name=companion_name)

    assert len(validated) == len(expected)
    for fact in validated:
        assert fact.subject not in must_not_subjects, (
            f"Fact has forbidden subject '{fact.subject}': {fact.text}"
        )


# ---------------------------------------------------------------------------
# Heuristic extractor tests — verify subject attribution on prepared inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_heuristic_subject_attribution(case: dict) -> None:
    """All facts produced by the heuristic must have subject='User'."""
    extractor = HeuristicFactExtractor()
    outcome = extractor.extract(
        chat_session_id=uuid4(),
        user_message=case["user_message"],
        assistant_message=case["assistant_message"],
        companion_name=case.get("companion_name"),
    )
    must_not_subjects = set(case.get("must_not_have_subject", []))
    for fact in outcome.facts:
        assert fact.subject == "User", f"Heuristic produced non-User subject: {fact}"
        assert fact.subject not in must_not_subjects


# ---------------------------------------------------------------------------
# Entity structure tests — verify expected entities are well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [c for c in _load_cases() if c.get("expected_entities")],
    ids=lambda c: c["name"],
)
def test_expected_entities_are_valid(case: dict) -> None:
    """Expected entities must have non-empty names and valid structure."""
    for ee in case["expected_entities"]:
        entity = EntityMention(
            name=ee["name"],
            relationship=ee.get("relationship", ""),
            aliases=ee.get("aliases", []),
        )
        assert entity.name, "Entity must have a non-empty name"
        assert isinstance(entity.aliases, list)
        # Relationship to user should be a simple label
        if entity.relationship:
            assert " " not in entity.relationship or entity.relationship.count(" ") <= 1, (
                f"Relationship should be a short label, got: {entity.relationship}"
            )
