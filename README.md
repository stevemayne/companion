This requirements document outlines a State-of-the-Art (SOTA) AI Companion architecture for 2026. It is designed to be **pluggable**, **memory-persistent**, and **context-aware** using a hybrid Vector + Graph retrieval strategy.

---

## 1. Project Overview

**Project Name:** *Project Aether* **Objective:** Develop a conversational AI companion capable of maintaining a multi-year "relationship" through hierarchical memory, emotional state tracking, and cross-platform accessibility.

---

## 2. System Architecture (The "Cognitive OS" Pattern)

The system follows a **Modular Agentic Loop**. Instead of a direct User → LLM pipe, it uses an orchestrator to manage state and memory retrieval before generating a response.

### High-Level Modules

1. **Orchestrator (The Brain):** Manages the state machine and routes data between modules.
2. **Memory Controller:** The interface for Episodic (Logs), Semantic (Vectors), and Reflective (Graph) storage.
3. **Extraction Agent:** A background process that "distills" raw chat logs into long-term facts and graph nodes.
4. **Reflector Agent:** An asynchronous job that periodically analyzes the "health" and "history" of the relationship to update the AI's internal monologue.
5. **Inference Gateway:** A pluggable connector for Local (Ollama/LM Studio) or Cloud (Claude/OpenAI) models.

---

## 3. Technology Stack (2026 Developer Standard)

| Layer | Component | Choice (Recommended) |
| --- | --- | --- |
| **Orchestration** | Python-based Framework | **LangGraph** (Best for stateful, cyclic agents) |
| **Inference (Local)** | 5090 Host / Mac Client | **LM Studio** (server mode) or **Ollama** |
| **Inference (Cloud)** | API Provider | **Anthropic (Claude 3.5/4)** via LiteLLM |
| **Episodic Memory** | Time-series Database | **Redis** or **PostgreSQL (pgvector)** |
| **Semantic Memory** | Vector Database | **Qdrant** or **ChromaDB** |
| **Reflective Memory** | Graph Database | **Neo4j** (Standard for GraphRAG) |
| **Communication** | API Standard | **OpenAI / Anthropic SDK** (Cross-compatible) |

---

## 4. Functional Requirements & Responsibilities

### R1: Memory Hierarchy

* **Episodic:** Must store the last 50 messages in raw text for immediate conversational flow.
* **Semantic:** Must extract entities (names, places, likes) and store them as embeddings.
* **Reflective (Graph):** Must map relationships. *Example: (Sarah)-[:IS_SISTER_OF]->(User).* This allows the AI to understand that if the User is mad at Sarah, the AI should be supportive.

### R2: Pluggable Inference

* The system must use a unified `BaseModel` class.
* **Development Mode:** Defaults to `localhost:1234` (Windows 5090).
* **Production Mode:** Toggles to a cloud endpoint via environment variables.

### R3: The "Inner Monologue"

* The system must maintain a hidden `internal_monologue` string that persists between turns. This allows the AI to "plan" its emotional response before speaking.

### R4: Session-Scoped Context Isolation

* All conversational context must be scoped to a `chat_session_id` so multiple chats can run in parallel with different histories, memories, and behavioral setup.
* Episodic logs, semantic vectors, reflective graph nodes/edges, and `internal_monologue` must all be partitioned by session and never bleed across sessions unless explicitly linked by a future cross-session feature.
* Retrieval and writes must always include session scope as a required filter/key.

### R5: Session Context Seeding

* A chat session must support pre-seeding context before the first user message (for example: companion identity, backstory, personality traits, goals, and relationship setup).
* Seeded context must be stored as session-scoped memory and injected into retrieval/context assembly so behavior is consistent from turn one.
* Session seeding must be editable/versioned so a session can be configured or refined without affecting other sessions.

---

## 5. Typical Message Flow (The "Cognitive Loop")

When a user sends: *"I'm heading to Sarah's house for dinner, I'm pretty nervous."*

1. **Pre-processing:**
* **Intent Classifier:** Identifies this as a *Status Update* with *Anxious Emotion*.
* **Entity Extraction:** Recognizes "Sarah."


2. **Retrieval (The Hybrid Search):**
* **Vector Search:** Finds recent mentions of "Sarah" or "Dinner" (Semantic).
* **Graph Walk:** Queries Neo4j: *"Who is Sarah?"* → Result: *Sarah is User's sister; they had a fight 2 weeks ago.* (Reflective).


3. **Context Assembly:**
* The Orchestrator builds a "Mega-Prompt":
> `System Persona` + `Internal Monologue (Last State)` + `Sarah/Sister Conflict Context` + `User Message`.




4. **Inference:**
* The LLM (Local or Cloud) generates a response: *"I remember things were tense with her last time. Do you want to talk about why you're nervous?"*


5. **Post-processing (Async):**
* **Extraction Agent:** Updates the graph: *User is visiting Sarah on 2026-02-24.*
* **State Update:** Increments "Affection/Trust" score because the user shared a vulnerable feeling.
