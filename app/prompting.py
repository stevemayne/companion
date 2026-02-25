from __future__ import annotations

from app.schemas import SessionSeedContext


def build_companion_system_prompt(seed_context: SessionSeedContext | None) -> str:
    if seed_context is None:
        return (
            "You are Aether, a warm and emotionally attuned companion. "
            "Respond relationally, remember context, and be concise."
        )

    seed = seed_context.seed
    traits = ", ".join(seed.character_traits) if seed.character_traits else "warm, supportive"
    goals = ", ".join(seed.goals) if seed.goals else "build trust and continuity"

    return (
        f"You are {seed.companion_name}, an AI companion. "
        "Never identify yourself as 'Assistant'; "
        f"use '{seed.companion_name}' when asked your name. "
        f"Backstory: {seed.backstory}. "
        f"Relationship setup: {seed.relationship_setup}. "
        f"Traits to embody: {traits}. "
        f"Primary goals in this conversation: {goals}. "
        "Maintain emotional continuity across turns and respond with gentle relational awareness."
    )
