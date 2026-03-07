from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.inference import EndpointConfig, OpenAICompatibleProvider
from app.schemas import PreprocessResult

logger = logging.getLogger(__name__)

ENTITY_STOPWORDS = {
    # Question words
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whom",
    "whose",
    # Pronouns
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    # Determiners / articles
    "the",
    "a",
    "an",
    "some",
    "any",
    "all",
    "each",
    "every",
    "no",
    "other",
    # Common verbs / auxiliaries
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "did",
    "does",
    "done",
    "doing",
    "has",
    "have",
    "had",
    "having",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "go",
    "went",
    "gone",
    "get",
    "got",
    "let",
    "make",
    "made",
    "take",
    "took",
    "come",
    "came",
    "see",
    "saw",
    "know",
    "knew",
    "think",
    "thought",
    "tell",
    "told",
    "said",
    "say",
    "give",
    "gave",
    "want",
    "need",
    "like",
    "just",
    "also",
    "still",
    "keep",
    "kept",
    # Common adjectives / adverbs
    "good",
    "great",
    "nice",
    "bad",
    "well",
    "very",
    "really",
    "sure",
    "much",
    "more",
    "most",
    "many",
    "few",
    "new",
    "old",
    "big",
    "little",
    "long",
    "short",
    "first",
    "last",
    "next",
    "same",
    "different",
    "right",
    "wrong",
    "only",
    "even",
    "already",
    "never",
    "always",
    # Conjunctions / prepositions
    "and",
    "but",
    "or",
    "so",
    "yet",
    "for",
    "nor",
    "about",
    "after",
    "before",
    "from",
    "into",
    "with",
    "without",
    "over",
    "under",
    "between",
    "through",
    "during",
    "since",
    "until",
    # Greetings / interjections
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank",
    "yes",
    "oh",
    "okay",
    "sorry",
    "please",
    "wow",
    "yeah",
    "nah",
    # Other common sentence starters
    "here",
    "there",
    "now",
    "then",
    "today",
    "tomorrow",
    "yesterday",
    "maybe",
    "perhaps",
    "actually",
    "basically",
    "honestly",
    "not",
    "don",
    "doesn",
    "didn",
    "won",
    "wouldn",
    "couldn",
    "shouldn",
    "isn",
    "aren",
    "wasn",
    "weren",
    "hasn",
    "haven",
    "hadn",
    # Contractions (with straight and curly apostrophes)
    "i'm",
    "i've",
    "i'd",
    "i'll",
    "it's",
    "he's",
    "she's",
    "we're",
    "we've",
    "we'd",
    "we'll",
    "they're",
    "they've",
    "they'd",
    "they'll",
    "you're",
    "you've",
    "you'd",
    "you'll",
    "that's",
    "there's",
    "here's",
    "what's",
    "who's",
    "let's",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "i\u2019m",
    "i\u2019ve",
    "i\u2019d",
    "i\u2019ll",
    "it\u2019s",
    "he\u2019s",
    "she\u2019s",
    "we\u2019re",
    "we\u2019ve",
    "they\u2019re",
    "they\u2019ve",
    "you\u2019re",
    "you\u2019ve",
    "that\u2019s",
    "there\u2019s",
    "here\u2019s",
    "don\u2019t",
    "doesn\u2019t",
    "didn\u2019t",
    "won\u2019t",
    "wouldn\u2019t",
    "can\u2019t",
    "couldn\u2019t",
    "shouldn\u2019t",
}

ALLOWED_INTENTS = {"question", "status_update", "statement"}
ALLOWED_EMOTIONS = {"anxious", "positive", "neutral"}


class ModelProvider(Protocol):
    def generate(self, *, chat_session_id: UUID, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class AnalysisOutcome:
    preprocess: PreprocessResult
    requested_provider: str
    used_provider: str
    fallback_reason: str | None
    latency_ms: float

    def as_trace(self) -> dict[str, str | float | None]:
        return {
            "requested_provider": self.requested_provider,
            "used_provider": self.used_provider,
            "fallback_reason": self.fallback_reason,
            "latency_ms": round(self.latency_ms, 2),
        }


class IntentAnalyzer(Protocol):
    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome: ...


class _LLMAnalysisPayload(BaseModel):
    intent: Literal["question", "status_update", "statement", "other"] = "other"
    emotion: str = "neutral"
    entities: list[str] = Field(default_factory=list)

    @field_validator("emotion")
    @classmethod
    def normalize_emotion(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in ("happy", "excited", "great", "good"):
            return "positive"
        if normalized in ("anxious", "worried", "nervous", "scared"):
            return "anxious"
        if normalized in ("sad", "angry", "negative"):
            return "neutral"
        return normalized if normalized in ALLOWED_EMOTIONS else "neutral"

    @field_validator("entities")
    @classmethod
    def normalize_entities(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for token in value:
            entity = re.sub(
                r"^[,.!?;:()\[\]{}\"\u2018\u2019\u201c\u201d']+"
                r"|[,.!?;:()\[\]{}\"\u2018\u2019\u201c\u201d']+$",
                "",
                token,
            )
            if not entity:
                continue
            if entity.lower() in ENTITY_STOPWORDS:
                continue
            if entity not in cleaned:
                cleaned.append(entity)
        return cleaned

    def to_preprocess_result(self) -> PreprocessResult:
        intent = self.intent if self.intent in ALLOWED_INTENTS else "statement"
        emotion = self.emotion if self.emotion in ALLOWED_EMOTIONS else "neutral"
        return PreprocessResult(intent=intent, emotion=emotion, entities=self.entities)


class HeuristicIntentAnalyzer:
    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome:
        del chat_session_id
        start = perf_counter()
        lowered = content.lower()
        if any(term in lowered for term in ("nervous", "anxious", "worried", "scared")):
            emotion = "anxious"
        elif any(term in lowered for term in ("happy", "excited", "great", "good")):
            emotion = "positive"
        else:
            emotion = "neutral"

        if "?" in content:
            intent = "question"
        elif any(term in lowered for term in ("i am", "i'm", "i feel", "today")):
            intent = "status_update"
        else:
            intent = "statement"

        entities: list[str] = []
        for token in content.split():
            cleaned = re.sub(
                r"^[,.!?;:()\[\]{}\"\u2018\u2019\u201c\u201d']+"
                r"|[,.!?;:()\[\]{}\"\u2018\u2019\u201c\u201d']+$",
                "",
                token,
            )
            if not cleaned or len(cleaned) <= 1 or not cleaned[:1].isupper():
                continue
            if cleaned.lower() in ENTITY_STOPWORDS:
                continue
            if cleaned not in entities:
                entities.append(cleaned)

        return AnalysisOutcome(
            preprocess=PreprocessResult(intent=intent, emotion=emotion, entities=entities),
            requested_provider="heuristic",
            used_provider="heuristic",
            fallback_reason=None,
            latency_ms=(perf_counter() - start) * 1000,
        )


class LLMIntentAnalyzer:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        fallback: IntentAnalyzer,
    ) -> None:
        self._provider = provider
        self._fallback = fallback

    def analyze(self, *, chat_session_id: UUID, content: str) -> AnalysisOutcome:
        start = perf_counter()
        try:
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[{"role": "user", "content": self._analysis_prompt(content)}],
            )
            payload = _parse_llm_payload(raw)
            return AnalysisOutcome(
                preprocess=payload.to_preprocess_result(),
                requested_provider="llm",
                used_provider="llm",
                fallback_reason=None,
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback.analyze(chat_session_id=chat_session_id, content=content)
            return AnalysisOutcome(
                preprocess=fallback.preprocess,
                requested_provider="llm",
                used_provider=fallback.used_provider,
                fallback_reason=type(exc).__name__,
                latency_ms=(perf_counter() - start) * 1000,
            )

    def _analysis_prompt(self, content: str) -> str:
        return (
            "You are a preprocessing classifier. "
            "Return strict JSON with keys: intent, emotion, entities.\n"
            "intent must be one of: question, status_update, statement, other.\n"
            "emotion should be a short label (prefer: anxious, positive, neutral).\n"
            "entities should contain named people/places/things and must not include "
            "question words.\n"
            f"message: {content}"
        )


def build_intent_analyzer(settings: Settings) -> IntentAnalyzer:
    provider = settings.analysis_provider.strip().lower()
    heuristic = HeuristicIntentAnalyzer()
    if provider != "llm":
        return heuristic

    llm_provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.analysis_model or settings.inference_model,
            base_url=settings.analysis_base_url or settings.inference_base_url,
            api_key=settings.analysis_api_key or settings.inference_api_key,
        ),
        timeout_seconds=settings.analysis_timeout_seconds,
        max_retries=settings.analysis_max_retries,
    )
    return LLMIntentAnalyzer(provider=llm_provider, fallback=heuristic)


def _parse_llm_payload(raw: str) -> _LLMAnalysisPayload:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON returned by analysis model.") from exc
    return _LLMAnalysisPayload.model_validate(data)


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    text: str
    importance: float = 0.5


@dataclass(frozen=True)
class EntityMention:
    name: str
    relationship: str
    owner: str = "User"
    entity_type: str = "person"
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionOutcome:
    facts: list[ExtractedFact]
    requested_provider: str
    used_provider: str
    fallback_reason: str | None
    latency_ms: float
    entities: list[EntityMention] = field(default_factory=list)
    companion_facts: list[ExtractedFact] = field(default_factory=list)


_ASSISTANT_SUBJECTS = {
    "assistant",
    "the assistant",
    "ai",
    "the ai",
    "companion",
    "the companion",
}


def _partition_facts(
    facts: list[ExtractedFact],
    companion_name: str | None,
) -> tuple[list[ExtractedFact], list[ExtractedFact]]:
    """Split facts into (user_facts, companion_facts) based on subject."""
    companion_lower = companion_name.strip().lower() if companion_name else None
    user_facts: list[ExtractedFact] = []
    companion_facts: list[ExtractedFact] = []
    seen_texts: set[str] = set()
    for fact in facts:
        if not fact.text.strip():
            continue
        if fact.text in seen_texts:
            continue
        seen_texts.add(fact.text)
        subject_lower = fact.subject.strip().lower()
        if companion_lower and subject_lower == companion_lower:
            companion_facts.append(fact)
        elif subject_lower in _ASSISTANT_SUBJECTS:
            continue  # reject generic "assistant"/"ai" subjects
        else:
            user_facts.append(fact)
    return user_facts, companion_facts


def validate_facts(
    facts: list[ExtractedFact],
    companion_name: str | None = None,
    assistant_message: str | None = None,
) -> list[ExtractedFact]:
    valid: list[ExtractedFact] = []
    companion_lower = companion_name.strip().lower() if companion_name else None
    seen_texts: set[str] = set()

    # Build a token set from the assistant message to detect facts that
    # were likely extracted from the assistant's words, not the user's.
    assistant_tokens: set[str] = set()
    if assistant_message:
        assistant_tokens = {t.lower() for t in assistant_message.split() if len(t) > 2}

    for fact in facts:
        if not fact.subject.strip() or not fact.text.strip():
            continue
        subject_lower = fact.subject.strip().lower()
        # Never store facts about the assistant/companion
        if subject_lower in _ASSISTANT_SUBJECTS:
            continue
        if companion_lower and subject_lower == companion_lower:
            continue
        # Reject facts whose text is primarily about the assistant
        text_lower = fact.text.strip().lower()
        if text_lower.startswith(("the assistant ", "assistant ")):
            continue
        if companion_lower and text_lower.startswith(
            (f"the {companion_lower} ", f"{companion_lower} ")
        ):
            continue
        # Reject facts whose content words overlap heavily with the
        # assistant message — likely extracted from assistant speech.
        if assistant_tokens:
            fact_tokens = {t.lower() for t in fact.text.split() if len(t) > 2}
            if fact_tokens:
                overlap = len(fact_tokens & assistant_tokens) / len(fact_tokens)
                if overlap > 0.7:
                    continue
        if fact.text in seen_texts:
            continue
        seen_texts.add(fact.text)
        valid.append(fact)
    return valid


class FactExtractor(Protocol):
    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
        user_name: str | None = None,
    ) -> ExtractionOutcome: ...


class LLMFactExtractor:
    def __init__(self, *, provider: ModelProvider) -> None:
        self._provider = provider

    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
        user_name: str | None = None,
    ) -> ExtractionOutcome:
        start = perf_counter()
        try:
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[
                    {
                        "role": "user",
                        "content": self._extraction_prompt(
                            user_message,
                            assistant_message,
                            companion_name,
                            user_name,
                        ),
                    },
                ],
            )
            all_facts, entities = _parse_extraction_payload(raw)
            effective_name = companion_name or "Companion"
            user_facts, companion_facts = _partition_facts(
                all_facts, effective_name,
            )
            user_facts = validate_facts(
                user_facts,
                companion_name,
                assistant_message=assistant_message,
            )
            return ExtractionOutcome(
                facts=user_facts,
                companion_facts=companion_facts,
                requested_provider="llm",
                used_provider="llm",
                fallback_reason=None,
                latency_ms=(perf_counter() - start) * 1000,
                entities=entities,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM fact extraction failed: %s", exc,
            )
            return ExtractionOutcome(
                facts=[],
                requested_provider="llm",
                used_provider="llm",
                fallback_reason=type(exc).__name__,
                latency_ms=(perf_counter() - start) * 1000,
            )

    def _extraction_prompt(
        self,
        user_message: str,
        assistant_message: str,
        companion_name: str | None,
        user_name: str | None = None,
    ) -> str:
        name = companion_name or "Companion"
        user_label = user_name or "User"
        return (
            "You are a fact and entity extraction system for a conversation "
            f"between {user_label} and a companion named '{name}'.\n\n"
            "Given a conversation turn, extract facts from BOTH speakers "
            "and any named entities mentioned.\n\n"
            "## Fact rules\n"
            "- Each fact must have: subject, predicate, object, text, "
            "importance.\n"
            f"- subject: '{user_label}' for user facts, '{name}' for companion self-facts.\n"
            "- predicate: the verb phrase.\n"
            "- object: the target of the action.\n"
            "- text: a short third-person declarative sentence.\n"
            "- importance: float 0.0-1.0. Relationships and core identity "
            "(0.7-0.9), preferences and opinions (0.5-0.7), transient "
            "observations (0.2-0.4).\n"
            "- The subject must be WHO THE FACT IS ABOUT, not who caused "
            f"or mentioned it. If {user_label} transforms the companion's body, "
            f"the resulting physical state is a fact about {name}, not "
            f"{user_label}. If the companion describes {user_label}'s trait, "
            f"the subject is '{user_label}'.\n"
            "- Pay careful attention to WHO says WHAT. Preserve the direction "
            "of the relationship exactly as stated.\n"
            "- CRITICAL: When the companion describes past shared experiences "
            f"or memories, extract ONLY what {user_label} explicitly stated or "
            "confirmed. The companion may fabricate or embellish memories. "
            f"If {user_label} says 'you pretended to be X', extract that the "
            "companion pretended to be X — do NOT reverse the roles based on "
            "what the companion claims happened.\n"
            f"- When {user_label} and the companion disagree about who did what, "
            f"trust {user_label}'s version. {user_label} is the source of truth.\n\n"
            f"## {user_label}'s facts\n"
            "Extract personal details, preferences, relationships, events, "
            f"emotions, plans, and opinions ABOUT {user_label}.\n\n"
            f"## {name}'s self-facts\n"
            f"Extract identity, preferences, personal history, skills, "
            f"abilities, and actions that reveal {name}'s traits or state. "
            f"This includes roleplay actions in asterisks "
            f"(e.g. '*I test my strength*' → '{name} tested their strength'). "
            f"Render ALL companion facts in third person: "
            f"'I love cooking' → '{name} loves cooking'.\n"
            f"Do NOT extract from {name}:\n"
            "- Conversational responses ('I understand', 'That sounds nice')\n"
            "- Questions or suggestions directed at the user\n"
            "- Meta-statements ('I'm here for you', 'I want to help')\n"
            "- Empathy expressions ('I can see why you feel that way')\n\n"
            "## Entity rules\n"
            "- Only extract entities that represent LASTING relationships, "
            "identities, or ongoing concerns — people, pets, workplaces, "
            "homes, long-term projects, recurring hobbies. "
            "Do NOT extract transient mentions (a meal, a material used "
            "once, a one-off action, a generic object).\n"
            "- Each entity MUST have a non-empty relationship. If you "
            "cannot name the relationship in 1-2 words, do not extract "
            "the entity.\n"
            "- name: short canonical form, 1-3 words. Use the EXACT name "
            "mentioned in the conversation — do NOT substitute names from "
            "other turns or confuse different people. If the user says "
            "'my sister Emma', the entity name is 'Emma', NOT any other "
            "person's name. Pay very careful attention to which name is "
            "used in THIS turn.\n"
            "- relationship: how the entity relates to the owner — this "
            "is a ROLE label describing the entity's relationship TO the "
            "owner. Think: 'What is this entity to the owner?' "
            "Good: sister, friend, workplace, pet, hobby, research_project, "
            "home_city, business_partner, invention, travel_destination. "
            "Bad: disease (unless the owner HAS the disease — a research "
            "topic is 'research_subject'), material_transformed, "
            "storage_location, meal, thing_mentioned.\n"
            "- IMPORTANT: If someone STUDIES or RESEARCHES a topic, the "
            "relationship is 'research_subject' or 'research_project', "
            "NOT the topic itself. E.g. a scientist studying protein folding "
            "→ relationship='research_subject', NOT 'disease'.\n"
            "- IMPORTANT: 'home_city' means where someone LIVES, not where "
            "they visited. 'We visited Edinburgh' does NOT make Edinburgh "
            "a home_city. A visit destination is 'visit_location'. Only "
            "use 'home_city' when someone explicitly says they live there "
            "or it's clearly their residence.\n"
            f"- owner: '{user_label}' or '{name}' — who this entity belongs to or "
            f"relates to. {user_label}'s family members belong to '{user_label}'. "
            f"{name}'s own traits or possessions belong to '{name}'.\n"
            "- entity_type: person, pet, place, organization, project, "
            "concept.\n"
            "- aliases: genuine nicknames or shortened names used by the "
            "speakers. 'sis' for Sarah = good. Do NOT include trivial "
            "variations like adding/removing articles ('the lab' for 'Lab') "
            "or restating the name. Empty array if no real nicknames.\n"
            "- Do not create a second, vaguer entity for something already "
            "named more specifically.\n\n"
            "## Output format\n"
            "Return strict JSON with keys 'facts' and 'entities'.\n\n"
            "Example 1:\n"
            f'  {user_label}: "My sister Sarah started a new job at Google."\n'
            f'  {name}: "That\'s wonderful! I used to work in HR myself."\n'
            "  {{\n"
            '    "facts": [\n'
            f'      {{"subject": "{user_label}", "predicate": "has sister who started",'
            ' "object": "a new job",'
            f' "text": "{user_label}\'s sister Sarah started a new job",'
            ' "importance": 0.7}},\n'
            f'      {{"subject": "{name}", "predicate": "used to work in",'
            f' "object": "HR",'
            f' "text": "{name} used to work in HR",'
            ' "importance": 0.5}}\n'
            "    ],\n"
            '    "entities": [\n'
            '      {{"name": "Sarah", "relationship": "sister",'
            f' "owner": "{user_label}", "entity_type": "person",'
            ' "aliases": []}},\n'
            '      {{"name": "Google", "relationship": "workplace",'
            f' "owner": "{user_label}", "entity_type": "organization",'
            ' "aliases": []}}\n'
            "    ]\n"
            "  }}\n"
            "  Note: 'new job' is a transient event, not an entity.\n\n"
            "Example 2 (roleplay actions):\n"
            f'  {user_label}: "Can you try lifting that boulder?"\n'
            f'  {name}: "*I test out my newfound strength by easily '
            'lifting the heavy boulder* I had no idea I was this strong!"\n'
            "  {{\n"
            '    "facts": [\n'
            f'      {{"subject": "{name}", "predicate": "has",'
            f' "object": "newfound strength",'
            f' "text": "{name} has newfound strength and can easily lift '
            'heavy objects",'
            ' "importance": 0.6}}\n'
            "    ],\n"
            '    "entities": []\n'
            "  }}\n\n"
            f"{user_label}'s message: {user_message}\n"
            f"{name}'s response: {assistant_message}"
        )


class _NoOpFactExtractor:
    """Returns empty results when LLM extraction is not configured."""

    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
        user_name: str | None = None,
    ) -> ExtractionOutcome:
        del chat_session_id, user_message, assistant_message, companion_name, user_name
        return ExtractionOutcome(
            facts=[],
            requested_provider="none",
            used_provider="none",
            fallback_reason=None,
            latency_ms=0.0,
        )


def build_fact_extractor(settings: Settings) -> FactExtractor:
    provider = settings.analysis_provider.strip().lower()
    if provider != "llm":
        return _NoOpFactExtractor()

    llm_provider = OpenAICompatibleProvider(
        endpoint=EndpointConfig(
            model=settings.analysis_model or settings.inference_model,
            base_url=settings.analysis_base_url or settings.inference_base_url,
            api_key=settings.analysis_api_key or settings.inference_api_key,
        ),
        timeout_seconds=settings.analysis_timeout_seconds,
        max_retries=settings.analysis_max_retries,
    )
    return LLMFactExtractor(provider=llm_provider)


def _parse_extraction_payload(raw: str) -> tuple[list[ExtractedFact], list[EntityMention]]:
    candidate = raw.strip()
    # Try fenced JSON (array or object)
    fenced = re.search(r"```(?:json)?\s*([\[{].*?[}\]])\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Find outermost JSON structure
        obj_start = candidate.find("{")
        arr_start = candidate.find("[")
        if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
            end = candidate.rfind("}")
            if end > obj_start:
                candidate = candidate[obj_start : end + 1]
        elif arr_start >= 0:
            end = candidate.rfind("]")
            if end > arr_start:
                candidate = candidate[arr_start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON returned by fact extraction model.") from exc

    if isinstance(data, dict):
        facts_raw = data.get("facts", [])
        entities_raw = data.get("entities", [])
    elif isinstance(data, list):
        facts_raw = data
        entities_raw = []
    else:
        raise ValueError("Fact extraction model did not return a JSON object or array.")

    facts = _parse_facts_list(facts_raw)
    entities = _parse_entities_list(entities_raw)
    return facts, entities


def _parse_facts_list(items: list) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    seen_texts: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                text = f"{subject} {predicate} {obj}".strip()
            raw_importance = item.get("importance")
            try:
                importance = max(0.0, min(1.0, float(raw_importance)))
            except (TypeError, ValueError):
                importance = 0.5
            if text and subject and text not in seen_texts:
                seen_texts.add(text)
                facts.append(
                    ExtractedFact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        text=text,
                        importance=importance,
                    )
                )
        elif isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in seen_texts:
                seen_texts.add(cleaned)
                facts.append(ExtractedFact(subject="User", predicate="", object="", text=cleaned))
    return facts


def _is_trivial_alias(name: str, alias: str) -> bool:
    """Reject aliases that are just article/case variants of the entity name."""
    n = name.lower().strip()
    a = alias.lower().strip()
    for article in ("the ", "a ", "an "):
        if n.startswith(article):
            n = n[len(article):]
        if a.startswith(article):
            a = a[len(article):]
    return n == a


def _parse_entities_list(items: list) -> list[EntityMention]:
    entities: list[EntityMention] = []
    seen_names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        relationship = str(item.get("relationship", "")).strip()
        owner = str(item.get("owner", "User")).strip() or "User"
        entity_type = str(item.get("entity_type", "person")).strip() or "person"
        raw_aliases = item.get("aliases", [])
        aliases = [
            str(a).strip() for a in raw_aliases
            if isinstance(a, str) and str(a).strip()
            and not _is_trivial_alias(name, str(a).strip())
        ]
        entities.append(EntityMention(
            name=name, relationship=relationship,
            owner=owner, entity_type=entity_type, aliases=aliases,
        ))
    return entities
