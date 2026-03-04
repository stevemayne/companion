# Ideas from RMM Paper (ACL 2025)

**Paper:** "In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents"
**Authors:** Tan et al. (Arizona State / Google Cloud AI Research)
**Link:** https://aclanthology.org/2025.acl-long.413.pdf

---

## 1. Topic-Based Memory Organization (Prospective Reflection)

**Paper idea:** Instead of storing memories at a fixed granularity (per-turn or per-session), decompose conversation into topic-based units that can span multiple turns. Each memory entry is a `(topic_summary, raw_dialogue)` pair. The topic summary serves as the search key for retrieval.

**Our current approach:** We extract individual atomic facts via `FactExtractor` (e.g., "User has a cat named Luna"). Each fact is a standalone `MemoryItem` with no grouping.

**What we could do:**
- Add a `topic` or `cluster_id` field to `MemoryItem` so related facts are grouped (e.g., "user's pets", "user's family", "user's job").
- At consolidation time, cluster related memories and generate a topic summary that becomes the primary search key.
- Store both the summary (for retrieval) and the raw facts (for context injection).
- This would improve retrieval because a query like "tell me about my family" would match a topic summary rather than needing to match individual scattered facts.

**Effort:** Medium. Extends consolidation + changes how memories are stored and queried.

---

## 2. Memory Merge vs. Add

**Paper idea:** When new memories are extracted at session end, each is compared against existing memories. An LLM decides whether to **add** (new topic) or **merge** (update existing topic with new information). Merging produces a combined summary.

**Our current approach:** Consolidation can _reinforce_ (bump importance) or _supersede_ (archive old, create replacement). Supersede is close to merge, but it replaces rather than combining.

**What we could do:**
- Add a `merge` operation to consolidation alongside reinforce/supersede. Rather than archiving the old memory, merge would combine the old and new content into a single updated memory, preserving the original `memory_id`.
- This reduces memory count while keeping information coherent.
- Example: old: "User has a cat named Luna" + new: "User's cat Luna is 3 years old" → merged: "User has a 3-year-old cat named Luna".

**Effort:** Low-medium. Extends `_parse_consolidation_response` and the LLM consolidation prompt.

---

## 3. Retrieval Reranking

**Paper idea:** After the base retriever returns top-K candidates, a lightweight reranker (linear layer + residual connection) refines relevance scores and selects top-M. The reranker is much cheaper to update than the full retriever.

**Our current approach:** We retrieve top-5 from the vector store by cosine similarity. No reranking step.

**What we could do:**
- After `query_similar()` returns candidates, apply a reranking step before injecting into context. Options:
  - **Heuristic reranker:** Boost memories that share entities with the current message, have higher importance scores, or were recently accessed. Simple weighted combination of similarity + importance + recency + entity overlap.
  - **LLM reranker:** Small LLM call: "Given this user message and these 10 candidate memories, which 5 are most relevant?" — expensive but high quality.
- Retrieve top-10 (or top-20) then rerank to top-5. The wider initial retrieval net catches more relevant candidates; reranking filters noise.

**Effort:** Low for heuristic reranker. Medium for LLM reranker.

---

## 4. LLM Attribution as Retrieval Feedback

**Paper idea:** When generating a response, the LLM also cites which memories it actually used (inline `[i]` references). Memories that get cited receive +1 reward; uncited memories get -1. This signal is used to improve retrieval over time via REINFORCE.

**Our current approach:** We retrieve memories and inject them into context, but we never know which ones the LLM actually used. No feedback loop.

**What we could do:**
- **Lightweight version (no RL):** After generating a response, do a quick check — does the response reference content from each retrieved memory? Use token overlap or embedding similarity between each memory and the response. Track a `usefulness_score` per memory over time.
- **Use usefulness for decay:** Memories that are repeatedly retrieved but never cited should decay faster (their importance should drop). Memories that are consistently cited should be reinforced.
- **Use usefulness for retrieval:** Factor historical usefulness into the reranking score — memories that have been useful in past retrievals should be preferred.
- This creates an organic feedback loop: good memories surface more, irrelevant ones fade.

**Effort:** Medium. Requires post-generation analysis step + tracking citation/usefulness per memory.

---

## 5. Multi-Granularity Memory is Better Than Fixed

**Paper finding:** Turn-level retrieval is too fragmented; session-level is too coarse. Topic-level (from Prospective Reflection) approaches the oracle "best-per-instance" granularity. Mixed (both turn + session in the same pool) actually performs _worse_ due to noise.

**Takeaway for us:**
- Our current atomic-fact granularity is similar to "turn-level" — very fine-grained. Works well for precise questions ("what's my cat's name?") but may fragment broader topics.
- We should avoid mixing granularities in the same retrieval pool (e.g., don't mix atomic facts with session summaries in one vector store).
- The consolidation topic-clustering from idea #1 would be our path to adaptive granularity.

---

## 6. Stronger Models May Abstain on Personal Information

**Paper finding:** Gemini-1.5-Pro performed _worse_ than 1.5-Flash when using RMM. The hypothesis: stronger models with more alignment training are more likely to refuse or hedge when asked to use personal information from memory.

**Takeaway for us:**
- When choosing the inference model, don't automatically assume bigger = better for a companion use case that relies on personal recall.
- Our current setup allows model selection via config (`inference_model`). Worth A/B testing if we switch models.
- Prompt engineering may be needed to give stronger models explicit permission to use retrieved personal context.

---

## 7. Memory Extraction Prompt Design

**Paper approach:** The extraction prompt asks the LLM to:
1. Decompose the session into distinct topics
2. For each topic, produce a `summary` and list of `reference` turn IDs
3. Return structured JSON with `extracted_memories` array

**Our current approach:** `FactExtractor` extracts atomic `(subject, predicate, object)` triples + entity names.

**What we could steal:**
- The idea of extracting topic-scoped summaries alongside (or instead of) atomic triples. A topic summary like "User is considering joining a local gym because of rainy weather" carries more context than separate triples ("user→wants→gym", "user→dislikes→rain").
- We could run both: atomic facts for precise retrieval + topic summaries for broader context.

---

## 8. Memory Update with Merge Semantics

**Paper prompt design:** The memory update prompt gives the LLM:
- The list of existing (history) summaries
- A new summary to integrate
- Two possible actions: `Add()` or `Merge(index, merged_summary)`

**Mapping to our consolidation:**
- Our current consolidation prompt asks for `reinforce`, `supersede`, `new_facts`. The paper's `Merge(index, merged_summary)` maps closest to `supersede` but is more nuanced — it explicitly generates the combined text.
- We could adopt this: our supersede operation already supports `replacement_text`, so it's conceptually the same as merge. The difference is framing — "merge" implies combining, "supersede" implies replacing. The LLM may produce better results with the merge framing.

---

## Priority Ranking for Implementation

| # | Idea | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 3 | Heuristic reranking (importance + recency + entity overlap) | High | Low | **Do first** |
| 2 | Memory merge in consolidation | Medium | Low | **Do second** |
| 4 | Usefulness tracking (token overlap between response and memories) | High | Medium | **Do third** |
| 1 | Topic-based memory clustering | High | Medium | Phase 5+ |
| 7 | Topic-scoped extraction alongside atomic facts | Medium | Medium | Phase 5+ |
| 5 | Avoid mixing granularities | Low | Low | Keep in mind |
| 6 | Model selection awareness | Low | N/A | Keep in mind |
| 8 | Merge framing in consolidation prompt | Low | Low | Next consolidation touch |
