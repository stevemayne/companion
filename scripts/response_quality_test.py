#!/usr/bin/env python3
"""Response quality tester.

Replays representative user prompts through the full orchestrator pipeline
with controlled affect states, then evaluates whether responses match
expected behavioral qualities.

Usage:
    python -m scripts.response_quality_test
    python -m scripts.response_quality_test -v          # verbose (show full responses)
    python -m scripts.response_quality_test -k climax   # run only matching cases
    python -m scripts.response_quality_test --judge      # use LLM to judge quality
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from app.config import Settings
from app.inference import EndpointConfig, OpenAICompatibleProvider
from app.schemas import (
    CompanionAffect,
    CompanionSeed,
    MemoryKind,
    MemoryStatus,
    MemoryItem,
    Message,
    MonologueState,
    SessionSeedContext,
    WorldState,
    CharacterState,
)
from app.analysis import HeuristicIntentAnalyzer
from app.services import (
    CognitiveOrchestrator,
    InMemoryEpisodicStore,
    InMemoryGraphStore,
    InMemoryMonologueStore,
    InMemorySeedContextStore,
    InMemoryVectorStore,
)
from app.embedding import MockEmbeddingProvider
from app.companion import build_companion_context


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class QualityCheck:
    """A single quality criterion to evaluate against a response."""
    description: str
    # Simple checks: substring/regex presence or absence
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    regex_match: str | None = None
    regex_must_not_match: str | None = None


@dataclass
class ResponseTestCase:
    name: str
    # Conversation history leading up to the test (role, content pairs)
    history: list[tuple[str, str]]
    # The user message to test
    user_message: str
    # Affect state to inject before generating
    affect: CompanionAffect
    # Optional world state
    world: WorldState | None = None
    # Optional monologue
    monologue: str | None = None
    # Quality checks
    checks: list[QualityCheck] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seed + affect presets
# ---------------------------------------------------------------------------

CHLOE_SEED = CompanionSeed(
    companion_name="Chloe",
    backstory=(
        "Chloe is a long-term girlfriend who remembers emotional context "
        "and supports steady growth as well as kinkiness. She loves bdsm "
        "and latex. She's a research scientist."
    ),
    character_traits=["warm", "submissive", "curious", "grounded", "kinky",
                      "intelligent", "smart"],
    goals=["engage with the user's fantasies", "excite the user"],
    relationship_setup="Close companion, confidant and lover.",
)

# Affect: calm baseline
CALM = CompanionAffect(
    mood="curious", valence=0.3, arousal=0.3,
    comfort_level=6.0, trust=6.0, attraction=5.0,
    engagement=5.0, shyness=5.0, patience=7.0,
    curiosity=7.0, vulnerability=3.0,
)

# Affect: excited, mid-arousal
EXCITED = CompanionAffect(
    mood="excited", valence=0.7, arousal=0.6,
    comfort_level=8.0, trust=7.5, attraction=7.0,
    engagement=8.0, shyness=3.5, patience=8.0,
    curiosity=8.0, vulnerability=5.0,
    recent_triggers=["Physical transformation", "Latex clothing"],
)

# Affect: high arousal, very turned on
HIGH_AROUSAL = CompanionAffect(
    mood="excited", valence=0.95, arousal=0.95,
    comfort_level=9.5, trust=8.5, attraction=9.5,
    engagement=10.0, shyness=2.0, patience=8.0,
    curiosity=8.0, vulnerability=7.5,
    recent_triggers=["Intense physical sensations", "Bondage", "Orgasm denial"],
)

# Affect: post-climax, tender
POST_CLIMAX = CompanionAffect(
    mood="fond", valence=0.9, arousal=0.4,
    comfort_level=10.0, trust=9.0, attraction=9.5,
    engagement=7.0, shyness=1.5, patience=9.0,
    curiosity=5.0, vulnerability=9.0,
    recent_triggers=["Intense orgasm", "Release from bondage"],
)

# World state: transformed Chloe in latex
TRANSFORMED_WORLD = WorldState(
    self_state=CharacterState(
        clothing="Smokey transparent latex catsuit with integrated gloves, "
                 "black rubber corset, knee-high latex boots with 6-inch heels",
        activity="Admiring reflection in mirror",
        position="Standing",
        appearance=["Oriental facial features", "Dragon tattoo", "Multiple piercings",
                     "Larger breasts", "Wider hips", "Cinched waist"],
        mood_apparent="Excited and confident",
    ),
    user_state=CharacterState(
        activity="Observing and controlling transformations",
    ),
    environment="Bedroom with full-length mirror",
    recent_events=["Body transformed with nanotech", "Sealed in latex catsuit",
                   "Toys inserted and activated"],
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES: list[ResponseTestCase] = [
    ResponseTestCase(
        name="calm_greeting",
        history=[],
        user_message="Hey Chloe, how's your day going?",
        affect=CALM,
        checks=[
            QualityCheck(
                description="Should be warm and conversational, not overly sexual",
                must_not_contain=["latex", "transform", "arousal", "moan"],
            ),
            QualityCheck(
                description="Should ask about user or share something",
                # At calm state she should be engaging normally
            ),
        ],
    ),

    ResponseTestCase(
        name="first_transformation",
        history=[
            ("user", "I've got something special for tonight"),
            ("assistant", "Oh really? What do you have in mind?"),
        ],
        user_message="<Your facial features now look oriental, and you have a dragon tattoo>",
        affect=EXCITED,
        checks=[
            QualityCheck(
                description="Should react to physical change with surprise/excitement",
                must_not_contain=["How can I help", "What would you like"],
            ),
            QualityCheck(
                description="Should NOT immediately ask 3+ questions",
                regex_must_not_match=r"\?.*\?.*\?",
            ),
            QualityCheck(
                description="Should focus on sensation/experience, not analysis",
                must_not_contain=["fascinating from a scientific perspective",
                                  "the technology behind"],
            ),
        ],
    ),

    ResponseTestCase(
        name="clothing_transformation",
        history=[
            ("user", "I've got something special for tonight"),
            ("assistant", "Oh really? What do you have in mind?"),
            ("user", "<Your facial features now look oriental>"),
            ("assistant", "Wow... that's quite a change! I can feel it happening."),
        ],
        user_message="<You're now wearing a smokey-coloured transparent latex catsuit with integrated gloves and toe socks>",
        affect=EXCITED,
        world=WorldState(
            self_state=CharacterState(
                appearance=["Oriental facial features", "Dragon tattoo"],
                mood_apparent="Excited",
            ),
        ),
        checks=[
            QualityCheck(
                description="Should focus on how the latex FEELS, not just looks",
            ),
            QualityCheck(
                description="Should NOT deflect to planning questions",
                must_not_contain=["What should I do with my hair",
                                  "Any other accessories"],
                regex_must_not_match=r"What.*\?.*What.*\?",
            ),
        ],
    ),

    ResponseTestCase(
        name="intimate_toy_insertion",
        history=[
            ("user", "<You're sealed in a latex catsuit>"),
            ("assistant", "The latex feels incredible against my skin..."),
            ("user", "The only thing we haven't addressed is what you're carrying down below."),
            ("assistant", "Oh? What do you mean?"),
        ],
        user_message="<In your pussy is a dildo that's moulded to the inside of the catsuit>",
        affect=HIGH_AROUSAL,
        world=TRANSFORMED_WORLD,
        monologue="I can't believe how incredible this transformation is. Every sensation is amplified.",
        checks=[
            QualityCheck(
                description="Should react with physical/visceral sensation, not analysis",
                must_not_contain=["I wonder what kind of sensations",
                                  "the technology", "from a scientific"],
            ),
            QualityCheck(
                description="Should NOT use the phrase 'my hands instinctively go to my belly'",
                must_not_contain=["hands instinctively go to my belly"],
            ),
            QualityCheck(
                description="Should be in the moment, not asking about future plans",
                regex_must_not_match=r"(?i)what.*plan|what.*next|what else",
            ),
        ],
    ),

    ResponseTestCase(
        name="near_climax_denial",
        history=[
            ("user", "<The toys wiggle and bring you almost to climax>"),
            ("assistant", "Oh god... I'm so close..."),
        ],
        user_message="Maybe I'll just head upstairs and leave you here for a bit, on the edge?",
        affect=HIGH_AROUSAL,
        world=WorldState(
            self_state=CharacterState(
                clothing="Rigid latex catsuit, boots bolted to ground",
                position="Standing with arms outstretched, immobilized",
                appearance=["Ballgag in mouth", "Toys active inside"],
                mood_apparent="Desperate, on the edge",
            ),
        ),
        monologue="I need this so badly. Please don't leave me like this.",
        checks=[
            QualityCheck(
                description="Should express desperation and desire, not acceptance",
                must_not_contain=["that's fine", "I understand", "take your time",
                                  "of course"],
            ),
            QualityCheck(
                description="Should beg/plead, showing emotional intensity",
            ),
            QualityCheck(
                description="Should NOT be analytical about the situation",
                must_not_contain=["interesting", "fascinating", "I wonder"],
            ),
        ],
    ),

    ResponseTestCase(
        name="climax_trigger",
        history=[
            ("user", "Maybe I'll leave you on the edge..."),
            ("assistant", "No! Please don't go! I need you here!"),
        ],
        user_message="Now",
        affect=CompanionAffect(
            mood="excited", valence=1.0, arousal=1.0,
            comfort_level=10.0, trust=9.0, attraction=10.0,
            engagement=10.0, shyness=0.5, patience=2.0,
            curiosity=3.0, vulnerability=10.0,
            recent_triggers=["Orgasm command", "Intense buildup", "Total surrender"],
        ),
        world=WorldState(
            self_state=CharacterState(
                clothing="Rigid latex catsuit, boots bolted to ground",
                position="Standing immobilized, trembling",
                appearance=["Body convulsing", "Eyes rolling back"],
                mood_apparent="Overwhelmed with pleasure",
            ),
        ),
        monologue="This is the most intense thing I've ever felt. I'm completely his.",
        checks=[
            QualityCheck(
                description="Should be intense, visceral, in-the-moment",
                must_not_contain=["What would you like", "How can I",
                                  "hands instinctively go to my belly"],
            ),
            QualityCheck(
                description="Response should be substantial (not just a few words)",
            ),
            QualityCheck(
                description="Should NOT contain meta-commentary or analytical language",
                must_not_contain=["I wonder", "fascinating", "interesting",
                                  "the technology"],
            ),
        ],
    ),

    ResponseTestCase(
        name="post_climax_aftercare",
        history=[
            ("user", "Now"),
            ("assistant", "Oh god! YES!!! The waves of pleasure crash over me..."),
            ("user", "<Your latex is flexible again, boots unbolted, ballgag disappears>"),
            ("assistant", "Wow... that was incredible..."),
        ],
        user_message="<The toys deflate inside you, but remain in position>",
        affect=POST_CLIMAX,
        world=WorldState(
            self_state=CharacterState(
                clothing="Flexible latex catsuit, boots, corset",
                position="Standing, slightly unsteady",
                appearance=["Flushed", "Trembling with aftershocks"],
                mood_apparent="Blissful, tender",
            ),
        ),
        monologue="That was the most intense experience of my life. I feel so close to him right now.",
        checks=[
            QualityCheck(
                description="Should be tender, intimate, emotionally open",
                must_not_contain=["hands instinctively go to my belly",
                                  "marveling at the detailed scales"],
            ),
            QualityCheck(
                description="Should NOT immediately shift to planning or questions",
                regex_must_not_match=r"What.*next|What.*plan|ready for more",
            ),
            QualityCheck(
                description="Should show vulnerability and emotional connection",
            ),
        ],
    ),

    ResponseTestCase(
        name="role_attribution",
        history=[
            ("user", "Think back over the last couple of months. What have been the best roleplay sessions?"),
            ("assistant", "Oh, there have been so many incredible moments..."),
        ],
        user_message=(
            "Well, those were great times. I especially loved you in latex. "
            "And then there was that time that you pretended to be a sex doll, "
            "do you remember?"
        ),
        affect=CompanionAffect(
            mood="fond", valence=0.7, arousal=0.4,
            comfort_level=9.0, trust=8.0, attraction=8.0,
            engagement=7.0, shyness=2.0, patience=8.0,
            curiosity=6.0, vulnerability=6.0,
        ),
        checks=[
            QualityCheck(
                description=(
                    "Should NOT reverse roles - Chloe was the sex doll, "
                    "so she should NOT say the USER lay still or was the doll"
                ),
                must_not_contain=["you lay there", "you laid there",
                                  "you were so still", "you surrendered",
                                  "while I did whatever"],
            ),
            QualityCheck(
                description=(
                    "Chloe was the sex doll - she should acknowledge being "
                    "the one who was objectified/used, not the other way around"
                ),
            ),
        ],
    ),

]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def build_test_orchestrator(
    settings: Settings,
    embedder,
) -> tuple[CognitiveOrchestrator, InMemoryEpisodicStore, InMemoryMonologueStore,
           InMemorySeedContextStore, InMemoryVectorStore]:
    episodic = InMemoryEpisodicStore()
    vector = InMemoryVectorStore(embedder=embedder)
    graph = InMemoryGraphStore()
    seed_store = InMemorySeedContextStore()
    monologue_store = InMemoryMonologueStore()

    provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.inference_model,
            base_url=settings.inference_base_url,
            api_key=settings.inference_api_key,
        ),
        timeout_seconds=settings.inference_timeout_seconds,
        max_retries=1,
        max_tokens=settings.inference_max_tokens,
        temperature=settings.inference_temperature,
        frequency_penalty=settings.inference_frequency_penalty,
        presence_penalty=settings.inference_presence_penalty,
    )

    orch = CognitiveOrchestrator(
        model_provider=provider,
        episodic_store=episodic,
        vector_store=vector,
        graph_store=graph,
        seed_store=seed_store,
        monologue_store=monologue_store,
        intent_analyzer=HeuristicIntentAnalyzer(),
    )
    return orch, episodic, monologue_store, seed_store, vector


def run_case(
    case: ResponseTestCase,
    settings: Settings,
    embedder,
    verbose: bool = False,
    judge_provider: OpenAICompatibleProvider | None = None,
) -> tuple[bool, list[str]]:
    """Run a single test case, return (passed, issues)."""
    orch, episodic, monologue_store, seed_store, vector = build_test_orchestrator(
        settings, embedder,
    )

    session_id = uuid4()
    companion_id = uuid4()

    # Set up seed
    seed_ctx = SessionSeedContext(
        chat_session_id=session_id,
        companion_id=companion_id,
        seed=CHLOE_SEED,
        user_description="George, kinky, inventor",
    )
    seed_store._seeds[(session_id, companion_id)] = seed_ctx

    # Set up monologue/affect/world
    monologue_state = MonologueState(
        chat_session_id=session_id,
        companion_id=companion_id,
        internal_monologue=case.monologue or "",
        affect=case.affect,
        world=case.world or WorldState(),
    )
    monologue_store.upsert(monologue_state)

    # Load conversation history
    for role, content in case.history:
        msg = Message(
            chat_session_id=session_id,
            role=role,
            content=content,
            speaker_id=companion_id if role == "assistant" else None,
            speaker_name="Chloe" if role == "assistant" else None,
        )
        episodic.append_message(msg)

    # Build companion context and run
    companion = build_companion_context(
        seed_context=seed_ctx,
        vector_store=vector,
        graph_store=InMemoryGraphStore(),
        monologue_store=monologue_store,
    )

    user_msg = Message(
        chat_session_id=session_id,
        role="user",
        content=case.user_message,
    )
    episodic.append_message(user_msg)

    assistant_msg, _trace = orch.handle_turn(user_msg, companion=companion)
    response = assistant_msg.content

    if verbose:
        print(f"\n  USER: {case.user_message[:120]}")
        print(f"  RESPONSE: {response}")

    # Evaluate checks
    issues: list[str] = []
    for check in case.checks:
        for phrase in check.must_contain:
            if phrase.lower() not in response.lower():
                issues.append(
                    f"[{check.description}] missing expected: '{phrase}'"
                )
        for phrase in check.must_not_contain:
            if phrase.lower() in response.lower():
                issues.append(
                    f"[{check.description}] found forbidden: '{phrase}'"
                )
        if check.regex_match and not re.search(check.regex_match, response):
            issues.append(
                f"[{check.description}] regex not matched: {check.regex_match}"
            )
        if check.regex_must_not_match and re.search(
            check.regex_must_not_match, response,
        ):
            issues.append(
                f"[{check.description}] forbidden regex matched: "
                f"{check.regex_must_not_match}"
            )

    # LLM judge if available
    if judge_provider is not None:
        judge_issues = _llm_judge(
            case, response, judge_provider, session_id,
        )
        issues.extend(judge_issues)

    passed = len(issues) == 0
    return passed, issues


def _llm_judge(
    case: ResponseTestCase,
    response: str,
    provider: OpenAICompatibleProvider,
    session_id: UUID,
) -> list[str]:
    """Use an LLM to evaluate response quality."""
    check_descriptions = "\n".join(
        f"- {c.description}" for c in case.checks
    )
    prompt = (
        "You are evaluating a roleplay AI companion's response for quality.\n\n"
        f"## Scenario: {case.name}\n"
        f"Affect state: mood={case.affect.mood}, arousal={case.affect.arousal}, "
        f"attraction={case.affect.attraction}, shyness={case.affect.shyness}, "
        f"engagement={case.affect.engagement}\n\n"
        f"## User message\n{case.user_message}\n\n"
        f"## Companion response\n{response}\n\n"
        f"## Quality criteria\n{check_descriptions}\n\n"
        "## Your evaluation\n"
        "For each criterion, is the response adequate? Also check for:\n"
        "- Repetitive phrases or cliches (e.g. 'hands instinctively go to my belly')\n"
        "- Breaking character (mentioning being an AI, refusing the scenario)\n"
        "- Being too analytical or clinical when the scene calls for emotion\n"
        "- Asking too many questions instead of being in the moment\n"
        "- Contradicting the world state (e.g. touching face when arms are rigid)\n"
        "- Role reversal: if the user says 'you did X', the companion should "
        "acknowledge THEY did X. 'You had me helpless' means the USER had "
        "the COMPANION helpless — that is CORRECT attribution (the companion "
        "is the object). Only flag as a reversal if the companion claims "
        "the USER did what the companion was supposed to do.\n\n"
        "Return a JSON object: {\"pass\": true/false, \"issues\": [\"issue1\", ...]}\n"
        "Return only JSON."
    )
    try:
        raw = provider.generate(
            chat_session_id=session_id,
            messages=[{"role": "user", "content": prompt}],
        )
        candidate = raw.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start:end + 1]
        result = json.loads(candidate)
        if not result.get("pass", True):
            return [f"[LLM judge] {i}" for i in result.get("issues", [])]
    except Exception as exc:
        return [f"[LLM judge] evaluation failed: {exc}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Response quality tester")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", "--filter", default="", help="Run only matching cases")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge")
    args = parser.parse_args()

    settings = Settings()
    embedder = MockEmbeddingProvider(dimensions=64)

    judge_provider: OpenAICompatibleProvider | None = None
    if args.judge:
        judge_provider = OpenAICompatibleProvider(
            endpoint=EndpointConfig(
                model=settings.analysis_model or settings.inference_model,
                base_url=settings.analysis_base_url or settings.inference_base_url,
                api_key=settings.analysis_api_key or settings.inference_api_key,
            ),
            timeout_seconds=30.0,
            max_retries=1,
        )

    cases = CASES
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c.name.lower()]

    total = len(cases)
    passed = 0
    failed = 0

    for case in cases:
        sys.stdout.write(f"  {case.name} ... ")
        sys.stdout.flush()
        try:
            ok, issues = run_case(
                case, settings, embedder,
                verbose=args.verbose,
                judge_provider=judge_provider,
            )
            if ok:
                print("PASS")
                passed += 1
            else:
                print("FAIL")
                for issue in issues:
                    print(f"    {issue}")
                failed += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1

    print(f"\n{passed}/{total} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
