#!/usr/bin/env python3
"""Test harness for LLM fact/entity extraction quality.

Fires conversation turns at the real extraction pipeline and checks
results against expected facts, entities, and graph relations.

Usage:
    # Run all cases
    python scripts/extraction_harness.py

    # Run specific cases by name (substring match)
    python scripts/extraction_harness.py sister emma japan

    # Show the raw LLM prompt for a case (no LLM call)
    python scripts/extraction_harness.py --show-prompt sister

    # Verbose mode — print full extraction output for every case
    python scripts/extraction_harness.py -v
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path so we can import app modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.analysis import (  # noqa: E402
    EntityMention,
    ExtractionOutcome,
    LLMFactExtractor,
    _parse_extraction_payload,
    _partition_facts,
    validate_facts,
)
from app.config import Settings  # noqa: E402
from app.inference import EndpointConfig, OpenAICompatibleProvider  # noqa: E402

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "extraction_harness_cases.json"

DUMMY_SESSION = UUID("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# Test case schema
# ---------------------------------------------------------------------------


@dataclass
class ExpectedFact:
    """A fact we expect to see (or not see) in extraction output."""

    text_contains: str
    subject: str = "User"
    kind: str = "user"  # "user" or "companion"


@dataclass
class ExpectedEntity:
    """An entity we expect in the graph output."""

    name: str
    relationship: str  # expected relation label (case-insensitive)
    owner: str = "User"


@dataclass
class ForbiddenPattern:
    """Something that should NOT appear."""

    field: str  # "fact_text", "entity_name", "relation"
    contains: str


@dataclass
class TestCase:
    name: str
    user_message: str
    assistant_message: str
    companion_name: str | None = None
    expected_facts: list[ExpectedFact] | None = None
    expected_entities: list[ExpectedEntity] | None = None
    forbidden: list[ForbiddenPattern] | None = None
    notes: str = ""


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text())
    cases: list[TestCase] = []
    for item in raw:
        cases.append(
            TestCase(
                name=item["name"],
                user_message=item["user_message"],
                assistant_message=item["assistant_message"],
                companion_name=item.get("companion_name"),
                expected_facts=[
                    ExpectedFact(**ef) for ef in item.get("expected_facts", [])
                ],
                expected_entities=[
                    ExpectedEntity(**ee) for ee in item.get("expected_entities", [])
                ],
                forbidden=[
                    ForbiddenPattern(**fp) for fp in item.get("forbidden", [])
                ],
                notes=item.get("notes", ""),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    name: str
    passed: bool
    outcome: ExtractionOutcome | None
    latency_ms: float
    errors: list[str]
    warnings: list[str]


def evaluate_case(
    extractor: LLMFactExtractor,
    case: TestCase,
) -> CaseResult:
    errors: list[str] = []
    warnings: list[str] = []

    start = time.perf_counter()
    try:
        outcome = extractor.extract(
            chat_session_id=DUMMY_SESSION,
            user_message=case.user_message,
            assistant_message=case.assistant_message,
            companion_name=case.companion_name,
        )
    except Exception as exc:
        return CaseResult(
            name=case.name,
            passed=False,
            outcome=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            errors=[f"Extraction raised: {exc}"],
            warnings=[],
        )
    latency = (time.perf_counter() - start) * 1000

    if outcome.fallback_reason:
        errors.append(f"Extraction failed with fallback: {outcome.fallback_reason}")

    all_fact_texts = [f.text for f in outcome.facts + outcome.companion_facts]

    # Check expected facts
    for ef in case.expected_facts or []:
        needle = ef.text_contains.lower()
        if ef.kind == "companion":
            pool = [f.text for f in outcome.companion_facts]
        else:
            pool = [f.text for f in outcome.facts]

        if not any(needle in t.lower() for t in pool):
            errors.append(
                f"Missing {ef.kind} fact containing '{ef.text_contains}' "
                f"(subject={ef.subject})"
            )

    # Check expected entities
    for ee in case.expected_entities or []:
        match = None
        for entity in outcome.entities:
            if entity.name.lower() == ee.name.lower():
                match = entity
                break
        if not match:
            # Also check by partial match
            for entity in outcome.entities:
                if ee.name.lower() in entity.name.lower():
                    match = entity
                    break
        if not match:
            errors.append(f"Missing entity '{ee.name}'")
        else:
            actual_rel = match.relationship.lower().replace("_", " ")
            expected_rel = ee.relationship.lower().replace("_", " ")
            if expected_rel not in actual_rel and actual_rel not in expected_rel:
                errors.append(
                    f"Entity '{ee.name}' has relationship '{match.relationship}', "
                    f"expected '{ee.relationship}'"
                )
            if match.owner.lower() != ee.owner.lower():
                warnings.append(
                    f"Entity '{ee.name}' owner is '{match.owner}', "
                    f"expected '{ee.owner}'"
                )

    # Check forbidden patterns
    for fp in case.forbidden or []:
        needle = fp.contains.lower()
        if fp.field == "fact_text":
            for t in all_fact_texts:
                if needle in t.lower():
                    errors.append(f"Forbidden fact text containing '{fp.contains}': {t}")
        elif fp.field == "entity_name":
            for e in outcome.entities:
                if needle in e.name.lower():
                    errors.append(f"Forbidden entity name containing '{fp.contains}': {e.name}")
        elif fp.field == "relation":
            for e in outcome.entities:
                if needle in e.relationship.lower():
                    errors.append(
                        f"Forbidden relation containing '{fp.contains}': "
                        f"{e.name} ({e.relationship})"
                    )

    return CaseResult(
        name=case.name,
        passed=len(errors) == 0,
        outcome=outcome,
        latency_ms=latency,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_outcome(outcome: ExtractionOutcome, indent: str = "  ") -> None:
    print(f"{indent}User facts ({len(outcome.facts)}):")
    for f in outcome.facts:
        print(f"{indent}  [{f.importance:.1f}] {f.text}")
    print(f"{indent}Companion facts ({len(outcome.companion_facts)}):")
    for f in outcome.companion_facts:
        print(f"{indent}  [{f.importance:.1f}] {f.text}")
    print(f"{indent}Entities ({len(outcome.entities)}):")
    for e in outcome.entities:
        aliases = f" (aka {', '.join(e.aliases)})" if e.aliases else ""
        print(
            f"{indent}  {e.owner} --{e.relationship}--> {e.name} "
            f"[{e.entity_type}]{aliases}"
        )
    print(f"{indent}Latency: {outcome.latency_ms:.0f}ms")


def show_prompt(extractor: LLMFactExtractor, case: TestCase) -> None:
    prompt = extractor._extraction_prompt(
        case.user_message, case.assistant_message, case.companion_name,
    )
    print(f"\n{'=' * 70}")
    print(f"PROMPT for: {case.name}")
    print(f"{'=' * 70}")
    print(prompt)
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_extractor() -> LLMFactExtractor:
    settings = Settings()
    endpoint = EndpointConfig(
        model=settings.analysis_model or settings.inference_model,
        base_url=settings.analysis_base_url or settings.inference_base_url,
        api_key=settings.analysis_api_key or settings.inference_api_key or "",
    )
    provider = OpenAICompatibleProvider(
        endpoint=endpoint,
        timeout_seconds=settings.analysis_timeout_seconds,
        max_retries=settings.analysis_max_retries,
        temperature=0.3,
    )
    return LLMFactExtractor(provider=provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction test harness")
    parser.add_argument(
        "filters",
        nargs="*",
        help="Run only cases whose name contains one of these substrings",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print full extraction output for every case",
    )
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the LLM prompt and exit (no LLM call)",
    )
    parser.add_argument(
        "--fixture", type=Path, default=FIXTURE_PATH,
        help="Path to fixture JSON file",
    )
    args = parser.parse_args()

    if not args.fixture.exists():
        print(f"Fixture file not found: {args.fixture}")
        print("Creating template fixture file...")
        _create_template_fixture(args.fixture)
        print(f"Created {args.fixture} — edit it and re-run.")
        return

    cases = load_cases(args.fixture)
    if args.filters:
        cases = [
            c for c in cases
            if any(f.lower() in c.name.lower() for f in args.filters)
        ]
        if not cases:
            print(f"No cases matching: {args.filters}")
            return

    extractor = build_extractor()

    if args.show_prompt:
        for case in cases:
            show_prompt(extractor, case)
        return

    print(f"Running {len(cases)} extraction case(s)...\n")

    results: list[CaseResult] = []
    for case in cases:
        result = evaluate_case(extractor, case)
        results.append(result)

        status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
        print(f"  {status}  {result.name} ({result.latency_ms:.0f}ms)")

        if result.errors:
            for err in result.errors:
                print(f"         \033[31m✗ {err}\033[0m")
        if result.warnings:
            for warn in result.warnings:
                print(f"         \033[33m⚠ {warn}\033[0m")

        if args.verbose and result.outcome:
            print_outcome(result.outcome, indent="         ")
        print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    print(f"{'=' * 60}")
    print(
        f"Results: {passed}/{len(results)} passed, "
        f"{failed} failed, avg latency {avg_latency:.0f}ms"
    )
    if failed:
        print(f"\nFailed cases:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
    print()

    sys.exit(1 if failed else 0)


def _create_template_fixture(path: Path) -> None:
    template = [
        {
            "name": "invention_basic",
            "user_message": "I've been working on a new invention - a portable water purifier that uses UV light.",
            "assistant_message": "That sounds amazing! What inspired you to work on this project?",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "water purifier", "subject": "User", "kind": "user"},
                {"text_contains": "UV light", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "portable water purifier", "relationship": "invention", "owner": "User"},
            ],
            "forbidden": [],
            "notes": "Basic invention extraction",
        },
        {
            "name": "sister_emma",
            "user_message": "I bumped into my sister Emma today. She's visiting from Portland next weekend.",
            "assistant_message": "That's great news! We should plan something fun.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "sister", "subject": "User", "kind": "user"},
                {"text_contains": "Portland", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Emma", "relationship": "sister", "owner": "User"},
            ],
            "forbidden": [
                {"field": "entity_name", "contains": "Sarah"},
                {"field": "entity_name", "contains": "Chloe"},
            ],
            "notes": "Entity name accuracy — must extract 'Emma' not any other name",
        },
        {
            "name": "research_not_disease",
            "user_message": "How's your research going? Are you still working on that protein folding project?",
            "assistant_message": "Yes! The protein stuff is fascinating. Understanding how proteins misfold can help with diseases like Alzheimer's.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "protein folding", "subject": "Chloe", "kind": "companion"},
            ],
            "expected_entities": [
                {"name": "protein folding", "relationship": "research", "owner": "Chloe"},
            ],
            "forbidden": [
                {"field": "relation", "contains": "disease"},
                {"field": "fact_text", "contains": "User has Alzheimer"},
                {"field": "fact_text", "contains": "User suffers"},
            ],
            "notes": "Research topic should not be classified as a disease the user has",
        },
        {
            "name": "business_partner_tom",
            "user_message": "My business partner Tom thinks we should pivot to a different design. I'm not sure I agree.",
            "assistant_message": "That's tough. What are his reasons for wanting to change?",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "Tom", "subject": "User", "kind": "user"},
                {"text_contains": "pivot", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Tom", "relationship": "business partner", "owner": "User"},
            ],
            "forbidden": [],
            "notes": "Relationship type should be business_partner, not just friend or person",
        },
        {
            "name": "mothers_name_and_city",
            "user_message": "My mum called today - she's been asking about you. Her name's Margaret. She made that amazing roast last time we visited in Edinburgh.",
            "assistant_message": "I'd love to visit her again! That roast was incredible.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "Margaret", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Margaret", "relationship": "mother", "owner": "User"},
            ],
            "forbidden": [
                {"field": "relation", "contains": "home"},
                {"field": "fact_text", "contains": "User lives in Edinburgh"},
            ],
            "notes": "Edinburgh is where mum lives, not the user. Margaret is the mum's name.",
        },
        {
            "name": "pet_and_allergy",
            "user_message": "Emma's bringing her dog Rex. Hope you don't mind - I know you're allergic to cats but dogs are fine, right?",
            "assistant_message": "Of course! Dogs are great and Rex is such a sweetie.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "allergic to cats", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Rex", "relationship": "pet", "owner": "User"},
            ],
            "forbidden": [
                {"field": "fact_text", "contains": "User is allergic to dogs"},
            ],
            "notes": "Chloe is allergic to cats (not dogs). Rex is Emma's dog.",
        },
        {
            "name": "travel_plan",
            "user_message": "I've been thinking about taking a trip to Japan in the fall. I'd love to see Kyoto and Tokyo.",
            "assistant_message": "Japan sounds amazing! I've always wanted to visit.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "Japan", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Japan", "relationship": "travel destination", "owner": "User"},
            ],
            "forbidden": [],
            "notes": "Travel plan extraction with place entities",
        },
        {
            "name": "companion_self_fact",
            "user_message": "What do you like to cook?",
            "assistant_message": "I absolutely love making pasta from scratch! There's something so satisfying about kneading the dough. I also enjoy baking — my chocolate soufflé is getting pretty good.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "pasta", "subject": "Chloe", "kind": "companion"},
            ],
            "expected_entities": [],
            "forbidden": [
                {"field": "fact_text", "contains": "User loves pasta"},
                {"field": "fact_text", "contains": "User enjoys baking"},
            ],
            "notes": "Companion self-facts should be attributed to Chloe, not User",
        },
        {
            "name": "no_extraction_from_greeting",
            "user_message": "Hey, how's it going?",
            "assistant_message": "I'm doing great, thanks for asking! How was your day?",
            "companion_name": "Chloe",
            "expected_facts": [],
            "expected_entities": [],
            "forbidden": [],
            "notes": "Greetings should produce no facts or entities",
        },
        {
            "name": "stressed_about_deadline",
            "user_message": "I'm feeling a bit stressed about a deadline. The prototype needs to be ready for a demo next Friday.",
            "assistant_message": "I'm sorry to hear that. Is there anything I can do to help?",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "stressed", "subject": "User", "kind": "user"},
                {"text_contains": "prototype", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [],
            "forbidden": [
                {"field": "fact_text", "contains": "Chloe is stressed"},
            ],
            "notes": "Emotional state and deadline should be attributed to User",
        },
        {
            "name": "camping_memory",
            "user_message": "I actually got the idea while we were camping at Lake Tahoe last summer. Remember the dodgy water?",
            "assistant_message": "Oh right, I do remember that! We had to boil all our drinking water.",
            "companion_name": "Chloe",
            "expected_facts": [
                {"text_contains": "Lake Tahoe", "subject": "User", "kind": "user"},
            ],
            "expected_entities": [
                {"name": "Lake Tahoe", "relationship": "camping location", "owner": "User"},
            ],
            "forbidden": [],
            "notes": "Shared memory with place extraction",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2) + "\n")


if __name__ == "__main__":
    main()
