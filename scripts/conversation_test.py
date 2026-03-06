#!/usr/bin/env python3
"""Run a multi-turn conversation test and assess extraction quality."""
import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
SESSION = "1442ccf0-b376-4994-b834-fd79e231ae0d"

MESSAGES = [
    "I've been working on a new invention - a portable water purifier that uses UV light. It's going to be really compact.",
    "Yeah, it's based on some research I read about UV-C wavelengths killing bacteria. I'm calling it the AquaPure.",
    "I actually got the idea while we were camping at Lake Tahoe last summer. Remember the dodgy water?",
    "How's your research going? Are you still working on that protein folding project?",
    "That sounds fascinating. Do you think it could lead to new drug treatments?",
    "I'm feeling a bit stressed about a deadline. The prototype needs to be ready for a demo next Friday.",
    "My business partner Tom thinks we should pivot to a different design. I'm not sure I agree.",
    "Tom wants to use chemical filtration instead of UV. I think UV is cleaner and more sustainable.",
    "Thanks for listening. You always know how to calm me down. What do you want to do tonight?",
    "I was thinking maybe we could cook dinner together. I picked up some fresh salmon from Pike Place Market.",
    "Oh, I also bumped into my sister Emma today. She's visiting from Portland next weekend.",
    "Emma's bringing her dog Rex. Hope you don't mind - I know you're allergic to cats but dogs are fine, right?",
    "She's also bringing her boyfriend Jake. They've been together about six months now.",
    "I've been thinking about taking a trip to Japan in the fall. Would you want to come?",
    "I'd love to see Kyoto and Tokyo. Maybe visit some onsen hot springs too.",
    "My mum called today - she's been asking about you. Wants us to come for Sunday dinner soon.",
    "Her name's Margaret, remember? She made that amazing roast last time we visited in Edinburgh.",
    "I had my annual checkup today. Doctor says I need to exercise more and watch my cholesterol.",
    "Maybe we should start running together in the mornings? There's a nice trail near Green Lake.",
    "This has been a really nice chat. I love that I can talk to you about anything.",
]


def _post_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def chat(message: str) -> dict:
    return _post_json(
        f"{BASE}/v1/chat",
        {"chat_session_id": SESSION, "message": message},
    )


def knowledge() -> dict:
    return _get_json(f"{BASE}/v1/knowledge/{SESSION}")


def print_separator():
    print("=" * 80)


def main():
    for i, msg in enumerate(MESSAGES, start=2):  # start=2 because turn 1 already done
        print_separator()
        print(f"TURN {i}: {msg}")
        print()

        resp = chat(msg)
        assistant = resp["assistant_message"]["content"]
        print(f"CHLOE: {assistant}")
        print()

        # Wait for background agents
        time.sleep(3)

        kn = knowledge()
        facts = kn["facts"]
        graph = kn["graph"]
        affect = kn.get("affect")
        monologue = kn.get("monologue")

        user_facts = [f for f in facts if f["kind"] != "companion"]
        companion_facts = [f for f in facts if f["kind"] == "companion"]

        print(f"--- KNOWLEDGE STATE ---")
        print(f"User facts: {len(user_facts)}  |  Companion facts: {len(companion_facts)}  |  Graph: {len(graph)}")

        if affect:
            print(f"Affect: mood={affect['mood']}, trust={affect['trust']:.1f}, engagement={affect['engagement']:.1f}, comfort={affect['comfort_level']:.1f}")

        # Show new facts (last 3)
        print(f"Recent user facts:")
        for f in user_facts[-3:]:
            print(f"  - {f['content']}")

        print(f"Recent companion facts:")
        for f in companion_facts[-3:]:
            print(f"  - {f['content']}")

        print(f"Graph relations ({len(graph)}):")
        for g in graph:
            print(f"  {g['source']} --{g['relation']}--> {g['target']}")

        if monologue:
            print(f"Monologue: {monologue[:120]}")
        print()

    # Final summary
    print_separator()
    print("FINAL SUMMARY")
    print_separator()
    kn = knowledge()
    facts = kn["facts"]
    graph = kn["graph"]
    affect = kn.get("affect")

    user_facts = [f for f in facts if f["kind"] != "companion"]
    companion_facts = [f for f in facts if f["kind"] == "companion"]

    print(f"\nTotal user facts: {len(user_facts)}")
    print(f"Total companion facts: {len(companion_facts)}")
    print(f"Total graph relations: {len(graph)}")

    if affect:
        print(f"\nFinal affect state:")
        for k, v in affect.items():
            if k != "recent_triggers":
                print(f"  {k}: {v}")
            else:
                print(f"  triggers: {v}")

    print(f"\nAll user facts:")
    for f in user_facts:
        print(f"  [{f['kind']}] {f['content']}")

    print(f"\nAll companion facts:")
    for f in companion_facts:
        print(f"  [{f['kind']}] {f['content']}")

    print(f"\nAll graph relations:")
    for g in graph:
        print(f"  {g['source']} --{g['relation']}--> {g['target']}")

    # Check companion_id attribution
    all_items = facts + graph
    missing_cid = [item for item in all_items if not item.get("companion_id")]
    if missing_cid:
        print(f"\nWARNING: {len(missing_cid)} items missing companion_id!")
    else:
        print(f"\nAll {len(all_items)} items have companion_id attribution.")


if __name__ == "__main__":
    main()
