from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.inference import EndpointConfig, OpenAICompatibleProvider
from app.schemas import PreprocessResult

ENTITY_STOPWORDS = {
    # Question words
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    # Pronouns
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    # Determiners / articles
    "the", "a", "an", "some", "any", "all", "each", "every", "no", "other",
    # Common verbs / auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "did", "does", "done", "doing",
    "has", "have", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "go", "went", "gone", "get", "got", "let", "make", "made", "take", "took",
    "come", "came", "see", "saw", "know", "knew", "think", "thought",
    "tell", "told", "said", "say", "give", "gave", "want", "need", "like",
    "just", "also", "still", "keep", "kept",
    # Common adjectives / adverbs
    "good", "great", "nice", "bad", "well", "very", "really", "sure",
    "much", "more", "most", "many", "few", "new", "old", "big", "little",
    "long", "short", "first", "last", "next", "same", "different",
    "right", "wrong", "only", "even", "already", "never", "always",
    # Conjunctions / prepositions
    "and", "but", "or", "so", "yet", "for", "nor",
    "about", "after", "before", "from", "into", "with", "without",
    "over", "under", "between", "through", "during", "since", "until",
    # Greetings / interjections
    "hello", "hi", "hey", "thanks", "thank", "yes", "no", "oh", "okay",
    "sorry", "please", "wow", "yeah", "nah",
    # Other common sentence starters
    "here", "there", "now", "then", "today", "tomorrow", "yesterday",
    "maybe", "perhaps", "actually", "basically", "honestly",
    "not", "don", "doesn", "didn", "won", "wouldn", "can", "couldn",
    "shouldn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
    # Contractions (with straight and curly apostrophes)
    "i'm", "i've", "i'd", "i'll", "it's", "he's", "she's", "we're", "we've",
    "we'd", "we'll", "they're", "they've", "they'd", "they'll", "you're",
    "you've", "you'd", "you'll", "that's", "there's", "here's", "what's",
    "who's", "let's", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "i\u2019m", "i\u2019ve", "i\u2019d", "i\u2019ll", "it\u2019s", "he\u2019s", "she\u2019s",
    "we\u2019re", "we\u2019ve", "they\u2019re", "they\u2019ve", "you\u2019re", "you\u2019ve",
    "that\u2019s", "there\u2019s", "here\u2019s", "don\u2019t", "doesn\u2019t", "didn\u2019t",
    "won\u2019t", "wouldn\u2019t", "can\u2019t", "couldn\u2019t", "shouldn\u2019t",
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
            entity = token.strip(",.!?;:()[]{}\"''\u2018\u2019\u201c\u201d")
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
            cleaned = token.strip(",.!?;:()[]{}\"''\u2018\u2019\u201c\u201d")
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

_FIRST_PERSON = re.compile(
    r"\b(i\s+am|i'm|i\s+was|i\s+have|i\s+had|i\s+feel|i\s+like|i\s+love|"
    r"i\s+want|i\s+need|i\s+think|i\s+know|i\s+went|i\s+met|i\s+saw|"
    r"i\s+argued|i\s+told|my\s+\w+|we\s+\w+)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    text: str


@dataclass(frozen=True)
class EntityMention:
    name: str
    relationship: str
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionOutcome:
    facts: list[ExtractedFact]
    requested_provider: str
    used_provider: str
    fallback_reason: str | None
    latency_ms: float
    entities: list[EntityMention] = field(default_factory=list)


_ASSISTANT_SUBJECTS = {
    "assistant", "the assistant", "ai", "the ai",
    "companion", "the companion",
}


def validate_facts(
    facts: list[ExtractedFact],
    companion_name: str | None = None,
) -> list[ExtractedFact]:
    valid: list[ExtractedFact] = []
    companion_lower = companion_name.strip().lower() if companion_name else None
    seen_texts: set[str] = set()
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
    ) -> ExtractionOutcome: ...


class HeuristicFactExtractor:
    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
    ) -> ExtractionOutcome:
        del chat_session_id, assistant_message
        start = perf_counter()
        facts: list[ExtractedFact] = []
        seen_texts: set[str] = set()
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(user_message) if s.strip()]
        for sentence in sentences:
            if "?" in sentence:
                continue
            if len(sentence.split()) < 3:
                continue
            if _FIRST_PERSON.search(sentence):
                text = _to_declarative(sentence)
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    facts.append(ExtractedFact(subject="User", predicate="", object="", text=text))
        facts = validate_facts(facts, companion_name)
        return ExtractionOutcome(
            facts=facts,
            requested_provider="heuristic",
            used_provider="heuristic",
            fallback_reason=None,
            latency_ms=(perf_counter() - start) * 1000,
        )


class LLMFactExtractor:
    def __init__(self, *, provider: ModelProvider, fallback: FactExtractor) -> None:
        self._provider = provider
        self._fallback = fallback

    def extract(
        self,
        *,
        chat_session_id: UUID,
        user_message: str,
        assistant_message: str,
        companion_name: str | None = None,
    ) -> ExtractionOutcome:
        start = perf_counter()
        try:
            raw = self._provider.generate(
                chat_session_id=chat_session_id,
                messages=[
                    {"role": "user", "content": self._extraction_prompt(user_message, assistant_message, companion_name)},
                ],
            )
            facts, entities = _parse_extraction_payload(raw)
            facts = validate_facts(facts, companion_name)
            return ExtractionOutcome(
                facts=facts,
                requested_provider="llm",
                used_provider="llm",
                fallback_reason=None,
                latency_ms=(perf_counter() - start) * 1000,
                entities=entities,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback.extract(
                chat_session_id=chat_session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                companion_name=companion_name,
            )
            return ExtractionOutcome(
                facts=fallback.facts,
                requested_provider="llm",
                used_provider=fallback.used_provider,
                fallback_reason=type(exc).__name__,
                latency_ms=(perf_counter() - start) * 1000,
            )

    def _extraction_prompt(self, user_message: str, assistant_message: str, companion_name: str | None) -> str:
        companion_rule = ""
        if companion_name:
            companion_rule = (
                f"- The assistant in this conversation is named '{companion_name}'. "
                f"'{companion_name}' must NEVER appear as a fact subject. "
                f"If '{companion_name}' is involved, place them in the object field.\n"
            )
        return (
            "You are a fact and entity extraction system. Given a user message and an "
            "assistant response, extract two things:\n\n"
            "1. **facts** — structured triples about the user.\n"
            "2. **entities** — named people, pets, or organizations mentioned.\n\n"
            "## Fact rules\n"
            "- Each fact must have: subject, predicate, object, text.\n"
            "- subject: who performs the action (usually 'User').\n"
            "- predicate: the verb phrase ('argued with', 'is interested in', 'has').\n"
            "- object: the target of the action ('Sarah', 'magic', 'a cat named Luna').\n"
            "- text: a short declarative sentence rendering of the triple.\n"
            "- Focus on: personal details, preferences, relationships, events, "
            "emotions, plans, and opinions.\n"
            "- Pay careful attention to WHO does WHAT to WHOM. Preserve the direction "
            "of the relationship exactly as stated.\n"
            f"{companion_rule}"
            "- NEVER extract facts about the assistant, what the assistant said, "
            "felt, imagined, or did. Only extract facts about the USER. "
            "If a fact's subject would be 'Assistant' or 'The Assistant', skip it.\n"
            "- Do NOT include greetings, filler, or questions.\n\n"
            "## Entity rules\n"
            "- Each entity must have: name (canonical form), relationship (to the user: "
            "'sister', 'boss', 'friend', 'pet', 'coworker', etc. — empty string if "
            "unknown), aliases (nicknames or alternate names, empty array if none).\n"
            "- Only extract entities that are people, pets, or organizations — not "
            "abstract concepts or places.\n\n"
            "## Output format\n"
            "Return strict JSON: a single object with keys 'facts' and 'entities'.\n\n"
            "Example:\n"
            '  User says "My sister Sarah, we call her sis, started a new job."\n'
            "  {\n"
            '    "facts": [\n'
            '      {"subject": "User", "predicate": "has sister who started", '
            '"object": "a new job", "text": "User\'s sister Sarah started a new job"}\n'
            "    ],\n"
            '    "entities": [\n'
            '      {"name": "Sarah", "relationship": "sister", "aliases": ["sis"]}\n'
            "    ]\n"
            "  }\n\n"
            f"User message: {user_message}\n"
            f"Assistant response: {assistant_message}"
        )


def build_fact_extractor(settings: Settings) -> FactExtractor:
    provider = settings.analysis_provider.strip().lower()
    heuristic = HeuristicFactExtractor()
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
    return LLMFactExtractor(provider=llm_provider, fallback=heuristic)


def _to_declarative(sentence: str) -> str:
    text = sentence.strip().rstrip(".")
    text = re.sub(r"^I'm\b", "User is", text)
    text = re.sub(r"^I\b", "User", text)
    text = re.sub(r"\bI'm\b", "user is", text)
    text = re.sub(r"\bI\b", "user", text)
    text = re.sub(r"\b[Mm]y\b", "user's", text)
    text = re.sub(r"\bme\b", "user", text)
    text = re.sub(r"\bmyself\b", "user", text)
    text = re.sub(r"\bmine\b", "user's", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


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
            if text and subject and text not in seen_texts:
                seen_texts.add(text)
                facts.append(ExtractedFact(subject=subject, predicate=predicate, object=obj, text=text))
        elif isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in seen_texts:
                seen_texts.add(cleaned)
                facts.append(ExtractedFact(subject="User", predicate="", object="", text=cleaned))
    return facts


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
        raw_aliases = item.get("aliases", [])
        aliases = [str(a).strip() for a in raw_aliases if isinstance(a, str) and str(a).strip()]
        entities.append(EntityMention(name=name, relationship=relationship, aliases=aliases))
    return entities
