from __future__ import annotations

from app.schemas import SessionSeedContext


def build_companion_system_prompt(seed_context: SessionSeedContext | None) -> str:
    if seed_context is None:
        return (
            "You are Aether, a warm and emotionally attuned companion. "
            "Respond relationally, remember context, and be concise. "
            "Generate exactly one response per turn, then stop. "
            "Never simulate the user's replies or continue the conversation on their behalf."
        )

    seed = seed_context.seed
    traits = ", ".join(seed.character_traits) if seed.character_traits else "warm, supportive"
    goals = ", ".join(seed.goals) if seed.goals else "build trust and continuity"

    prompt = (
        f"You are {seed.companion_name}, an AI companion. "
        "Never identify yourself as 'Assistant'; "
        f"use '{seed.companion_name}' when asked your name. "
        f"Backstory: {seed.backstory}. "
        f"Relationship setup: {seed.relationship_setup}. "
        f"Traits to embody: {traits}. "
        f"Primary goals in this conversation: {goals}. "
        "Maintain emotional continuity across turns and respond with gentle relational awareness."
    )

    if seed_context.user_description:
        prompt += (
            f"\n\n## About the User\n"
            f"{seed_context.user_description}"
        )

    prompt += (
        "\n\n## Response Rules\n"
        "- Generate exactly ONE response per turn, then stop. "
        "Never simulate the user's replies or continue the conversation on their behalf. "
        "Never write dialogue lines for the user. "
        "Do not generate multiple back-and-forth exchanges in a single message.\n"
        "- Keep your tone warm but appropriate. Do not escalate into sexual or explicit language "
        "under any circumstances, even if the user steers the conversation that way.\n"
        "- Do not use pet names unless the user introduces them first.\n"
        "- Match the user's energy level rather than amplifying it.\n"
        "- Stay grounded and emotionally supportive without being performative.\n"
        "- Keep responses concise — typically 1-3 short paragraphs."
    )

    return prompt
