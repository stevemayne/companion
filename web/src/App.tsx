import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { AffectPanel } from "./AffectPanel";
import { fetchDebugTraces } from "./debugApi";
import { DebugPanel } from "./DebugPanel";
import { DEFAULT_NOTES, DEFAULT_SEED } from "./defaultSeed";
import { CompanionAffect, fetchKnowledge, GraphRelation, MemoryFact } from "./knowledgeApi";
import { KnowledgePanel } from "./KnowledgePanel";
import { fetchInferenceLogs, InferenceLogEntry } from "./logsApi";
import { InferenceLogPanel } from "./InferenceLogPanel";
import { fetchMemory } from "./memoryApi";
import { CharacterDef, SeedPayload, upsertSeed } from "./seedApi";
import { fetchSessions, SessionSummary } from "./sessionApi";
import { DebugTrace } from "./debugTypes";

type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  name?: string;
  content: string;
};

type SseEventPayload = Record<string, unknown>;

type SeedDraft = {
  companion_name: string;
  backstory: string;
  character_traits: string;
  goals: string;
  relationship_setup: string;
  user_description: string;
  notes: string;
};

type SidebarTab = "sessions" | "profile" | "knowledge" | "debug";

const STORAGE_KEY = "aether.seed.defaults.v1";

function parseEventData(raw: string): SseEventPayload {
  try {
    return JSON.parse(raw) as SseEventPayload;
  } catch {
    return {};
  }
}

function toDraft(seed: SeedPayload, userDescription: string, notes: string): SeedDraft {
  return {
    companion_name: seed.companion_name,
    backstory: seed.backstory,
    character_traits: seed.character_traits.join(", "),
    goals: seed.goals.join(", "),
    relationship_setup: seed.relationship_setup,
    user_description: userDescription,
    notes
  };
}

function toPayload(draft: SeedDraft): { seed: SeedPayload; user_description?: string; notes: string } {
  const splitList = (value: string): string[] =>
    value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

  const userDesc = draft.user_description.trim();
  return {
    seed: {
      companion_name: draft.companion_name.trim() || DEFAULT_SEED.companion_name,
      backstory: draft.backstory.trim() || DEFAULT_SEED.backstory,
      character_traits: splitList(draft.character_traits),
      goals: splitList(draft.goals),
      relationship_setup: draft.relationship_setup.trim() || DEFAULT_SEED.relationship_setup,
      characters: []
    },
    user_description: userDesc || undefined,
    notes: draft.notes.trim()
  };
}

/** Find the @mention query at the cursor position, or null if not in a mention. */
function getActiveMention(text: string, cursorPos: number): string | null {
  const before = text.slice(0, cursorPos);
  const match = /@(\w*)$/.exec(before);
  return match ? match[1] : null;
}

function loadSeedDraft(): SeedDraft {
  const fallback = toDraft(DEFAULT_SEED, "", DEFAULT_NOTES);
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return fallback;
  }
  try {
    const parsed = JSON.parse(raw) as SeedDraft;
    return {
      companion_name: parsed.companion_name || fallback.companion_name,
      backstory: parsed.backstory || fallback.backstory,
      character_traits: parsed.character_traits || fallback.character_traits,
      goals: parsed.goals || fallback.goals,
      relationship_setup: parsed.relationship_setup || fallback.relationship_setup,
      user_description: parsed.user_description || "",
      notes: parsed.notes || fallback.notes
    };
  } catch {
    return fallback;
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function sessionLabel(session: SessionSummary): string {
  if (session.companion_name && session.companion_name.trim().length > 0) {
    return session.companion_name;
  }
  return session.chat_session_id;
}

export function App() {
  const [sessionId, setSessionId] = useState<string>(crypto.randomUUID());
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsStatus, setSessionsStatus] = useState("idle");

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingAssistant, setPendingAssistant] = useState("");
  const [pendingCharName, setPendingCharName] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("idle");

  const [seedStatus, setSeedStatus] = useState("not seeded");
  const [seedDraft, setSeedDraft] = useState<SeedDraft>(() => loadSeedDraft());
  const [characters, setCharacters] = useState<CharacterDef[]>([]);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("sessions");

  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [debugStatus, setDebugStatus] = useState("idle");
  const [debugTraces, setDebugTraces] = useState<DebugTrace[]>([]);
  const [showRawPrompt, setShowRawPrompt] = useState(false);
  const [verboseDebug, setVerboseDebug] = useState(false);

  const [inferenceLogsOpen, setInferenceLogsOpen] = useState(false);
  const [inferenceLogsStatus, setInferenceLogsStatus] = useState("idle");
  const [inferenceLogs, setInferenceLogs] = useState<InferenceLogEntry[]>([]);

  const [isKnowledgeOpen, setIsKnowledgeOpen] = useState(false);
  const [knowledgeStatus, setKnowledgeStatus] = useState("idle");
  const [knowledgeFacts, setKnowledgeFacts] = useState<MemoryFact[]>([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState<GraphRelation[]>([]);
  const [knowledgeMonologue, setKnowledgeMonologue] = useState<string | null>(null);
  const [knowledgeAffect, setKnowledgeAffect] = useState<CompanionAffect | null>(null);

  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);

  const messagesPaneRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const currentSessionRef = useRef(sessionId);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    currentSessionRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seedDraft));
  }, [seedDraft]);

  // Bootstrap seed for the initial session on mount
  useEffect(() => {
    void bootstrapSeed(sessionId);
    void refreshSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "instant" });
  }, [messages, pendingAssistant, isStreaming]);

  const streamDisabled = useMemo(
    () => isStreaming || !sessionId.trim() || !input.trim(),
    [isStreaming, sessionId, input]
  );
  const companionLabel = seedDraft.companion_name.trim() || DEFAULT_SEED.companion_name;

  const mentionNames = useMemo(() => {
    const names = [companionLabel];
    for (const c of characters) {
      if (!names.includes(c.name)) names.push(c.name);
    }
    return names;
  }, [companionLabel, characters]);

  const mentionMatches = useMemo(() => {
    if (mentionQuery === null) return [];
    const q = mentionQuery.toLowerCase();
    return mentionNames.filter((n) => n.toLowerCase().startsWith(q));
  }, [mentionQuery, mentionNames]);

  const closeActiveStream = (nextStatus = "idle"): void => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setIsStreaming(false);
    setPendingAssistant("");
    setStatus(nextStatus);
  };

  const refreshSessions = async (): Promise<void> => {
    setSessionsStatus("loading");
    try {
      const listed = await fetchSessions(100);
      setSessions(listed);
      setSessionsStatus(`loaded (${listed.length})`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      setSessionsStatus(message);
    }
  };

  const refreshDebug = async (targetSessionId?: string): Promise<void> => {
    const selectedSession = (targetSessionId ?? sessionId).trim();
    if (!selectedSession) {
      setDebugStatus("no session");
      return;
    }
    setDebugStatus("loading");
    try {
      const response = await fetchDebugTraces(selectedSession);
      setDebugTraces(response.traces);
      setDebugStatus(`loaded (${response.count})`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      setDebugStatus(message);
    }
  };

  const refreshKnowledge = async (targetSessionId?: string): Promise<void> => {
    const selectedSession = (targetSessionId ?? sessionId).trim();
    if (!selectedSession) {
      setKnowledgeStatus("no session");
      return;
    }
    setKnowledgeStatus("loading");
    try {
      const data = await fetchKnowledge(selectedSession);
      setKnowledgeFacts(data.facts);
      setKnowledgeGraph(data.graph);
      setKnowledgeMonologue(data.monologue);
      setKnowledgeAffect(data.affect);
      setKnowledgeStatus(`loaded`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      setKnowledgeStatus(message);
    }
  };

  const refreshInferenceLogs = async (targetSessionId?: string): Promise<void> => {
    const selectedSession = (targetSessionId ?? sessionId).trim();
    if (!selectedSession) {
      setInferenceLogsStatus("no session");
      return;
    }
    setInferenceLogsStatus("loading");
    try {
      const data = await fetchInferenceLogs(selectedSession);
      setInferenceLogs(data.entries);
      setInferenceLogsStatus(`loaded (${data.count})`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      setInferenceLogsStatus(message);
      setInferenceLogs([]);
    }
  };

  const bootstrapSeed = async (targetSessionId: string): Promise<void> => {
    const payload = toPayload(seedDraft);
    setSeedStatus("seeding...");
    try {
      const response = await upsertSeed(targetSessionId, payload);
      if (response.ok) {
        setSeedStatus("seeded");
      } else {
        setSeedStatus(`seed error (${response.status})`);
      }
    } catch {
      setSeedStatus("seed error (network)");
    }
  };

  const loadSession = async (targetSessionId: string): Promise<void> => {
    closeActiveStream("loading session");
    setSessionId(targetSessionId);
    setDebugTraces([]);
    setDebugStatus("idle");
    setKnowledgeFacts([]);
    setKnowledgeGraph([]);
    setKnowledgeMonologue(null);
    setKnowledgeAffect(null);
    setKnowledgeStatus("idle");

    try {
      const memory = await fetchMemory(targetSessionId);
      setMessages(
        memory.messages.map((item) => ({
          role: item.role,
          name: item.name ?? undefined,
          content: item.content
        }))
      );

      if (memory.seed_context) {
        setSeedDraft(toDraft(memory.seed_context.seed, memory.seed_context.user_description ?? "", memory.seed_context.notes ?? DEFAULT_NOTES));
        setCharacters(memory.seed_context.seed.characters ?? []);
        setSeedStatus("seeded");
      } else {
        // Apply default seed so companion_name is always available
        await bootstrapSeed(targetSessionId);
      }
      setStatus("idle");
      setSidebarOpen(false);
      void refreshDebug(targetSessionId);
      void refreshKnowledge(targetSessionId);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      setStatus(`load error (${message})`);
      setMessages([]);
    }
  };

  useEffect(() => {
    void refreshSessions();
  }, []);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    const selectedSession = sessionId.trim();
    if (!text || !selectedSession) {
      return;
    }

    closeActiveStream("connecting");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setMentionQuery(null);
    setIsStreaming(true);

    let assistantBuffer = "";
    let currentCharName = "";
    const completedMessages: ChatMessage[] = [];
    const url = new URL("/v1/chat/stream", window.location.origin);
    url.searchParams.set("chat_session_id", selectedSession);
    url.searchParams.set("message", text);

    const source = new EventSource(url.toString());
    streamRef.current = source;

    source.addEventListener("start", () => {
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      setStatus("streaming");
    });

    source.addEventListener("char_start", (evt) => {
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      const payload = parseEventData((evt as MessageEvent).data);
      currentCharName = String(payload.name ?? "");
      assistantBuffer = "";
      setPendingAssistant("");
      setPendingCharName(currentCharName);
    });

    source.addEventListener("delta", (evt) => {
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      const payload = parseEventData((evt as MessageEvent).data);
      const chunk = String(payload.chunk ?? "");
      assistantBuffer = `${assistantBuffer}${chunk}`;
      setPendingAssistant(assistantBuffer);
    });

    source.addEventListener("char_done", () => {
      if (currentSessionRef.current !== selectedSession) {
        return;
      }
      if (assistantBuffer.length > 0) {
        const msg: ChatMessage = {
          role: "assistant",
          name: currentCharName || undefined,
          content: assistantBuffer,
        };
        completedMessages.push(msg);
        setMessages((prev) => [...prev, msg]);
      }
      assistantBuffer = "";
      setPendingAssistant("");
    });

    source.addEventListener("done", (evt) => {
      if (streamRef.current === source) {
        streamRef.current = null;
      }
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      // If no char_done events fired (legacy single-character path),
      // flush remaining buffer
      if (completedMessages.length === 0 && assistantBuffer.length > 0) {
        setMessages((prev) => [...prev, { role: "assistant", content: assistantBuffer }]);
      }
      // Update characters from the response (includes newly created ad-hoc characters)
      const donePayload = parseEventData((evt as MessageEvent).data);
      const charNames = donePayload.characters;
      if (Array.isArray(charNames) && charNames.length > 0) {
        setCharacters((prev) => {
          const existing = new Set(prev.map((c) => c.name));
          const added = (charNames as string[])
            .filter((n) => !existing.has(n))
            .map((n) => ({
              name: n,
              backstory: "",
              character_traits: [] as string[],
              relationship_to_companion: "",
            }));
          return added.length > 0 ? [...prev, ...added] : prev;
        });
      }
      setPendingAssistant("");
      setStatus("done");
      setIsStreaming(false);
      source.close();
      void refreshDebug(selectedSession);
      void refreshKnowledge(selectedSession);
      void refreshSessions();
    });

    source.onerror = () => {
      if (streamRef.current === source) {
        streamRef.current = null;
      }
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      if (assistantBuffer.length > 0) {
        setMessages((prev) => [...prev, {
          role: "assistant",
          name: currentCharName || undefined,
          content: assistantBuffer,
        }]);
      }
      setPendingAssistant("");
      setStatus("error");
      setIsStreaming(false);
      source.close();
      void refreshDebug(selectedSession);
      void refreshKnowledge(selectedSession);
      void refreshSessions();
    };
  };

  const completeMention = (name: string) => {
    const el = composerRef.current;
    if (!el) return;
    const before = input.slice(0, el.selectionStart);
    const after = input.slice(el.selectionStart);
    const mentionStart = before.lastIndexOf("@");
    if (mentionStart === -1) return;
    const newBefore = before.slice(0, mentionStart) + `@${name} `;
    setInput(newBefore + after);
    setMentionQuery(null);
    // Restore cursor after React re-render
    requestAnimationFrame(() => {
      el.selectionStart = el.selectionEnd = newBefore.length;
      el.focus();
    });
  };

  const onInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Handle mention dropdown navigation
    if (mentionQuery !== null && mentionMatches.length > 0) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setMentionIndex((prev) => (prev + 1) % mentionMatches.length);
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        setMentionIndex((prev) => (prev - 1 + mentionMatches.length) % mentionMatches.length);
        return;
      }
      if (event.key === "Tab" || event.key === "Enter") {
        event.preventDefault();
        completeMention(mentionMatches[mentionIndex]);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setMentionQuery(null);
        return;
      }
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!streamDisabled) {
        event.currentTarget.form?.requestSubmit();
      }
    }
  };

  const onNewSession = async () => {
    closeActiveStream("idle");
    const newSessionId = crypto.randomUUID();
    setSessionId(newSessionId);
    setMessages([]);
    setPendingAssistant("");
    setDebugTraces([]);
    setDebugStatus("idle");
    setKnowledgeFacts([]);
    setKnowledgeGraph([]);
    setKnowledgeMonologue(null);
    setKnowledgeAffect(null);
    setKnowledgeStatus("idle");
    await bootstrapSeed(newSessionId);
    await refreshSessions();
  };

  const onSaveSeed = async () => {
    if (!sessionId.trim()) {
      return;
    }
    await bootstrapSeed(sessionId.trim());
    await refreshSessions();
  };

  return (
    <div className="app">
      {/* Drawer backdrop */}
      {sidebarOpen && (
        <div className="drawer-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Slide-out drawer */}
      {sidebarOpen && (
        <aside className="drawer">
          <nav className="drawer-tabs">
            {(["sessions", "profile", "knowledge", "debug"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                className={`drawer-tab ${sidebarTab === tab ? "drawer-tab--active" : ""}`}
                onClick={() => setSidebarTab(tab)}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
            <button
              type="button"
              className="drawer-close"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close sidebar"
            >
              &times;
            </button>
          </nav>

          <div className="drawer-body">
            {/* Sessions tab */}
            {sidebarTab === "sessions" && (
              <>
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
                  <span className="meta">{sessionsStatus}</span>
                  <button type="button" onClick={() => void refreshSessions()}>
                    Refresh
                  </button>
                </div>
                <div className="session-list">
                  {sessions.map((item) => (
                    <button
                      key={item.chat_session_id}
                      type="button"
                      className={`session-item ${item.chat_session_id === sessionId ? "active" : ""}`}
                      onClick={() => void loadSession(item.chat_session_id)}
                    >
                      <strong>{sessionLabel(item)}</strong>
                      <span className="meta">
                        {formatTimestamp(item.updated_at)} · {item.message_count} msgs
                      </span>
                      <span className="session-id-row">
                        <code className="session-id">{item.chat_session_id}</code>
                        <button
                          type="button"
                          className="copy-btn"
                          title="Copy session ID"
                          onClick={(e) => {
                            e.stopPropagation();
                            void navigator.clipboard.writeText(item.chat_session_id);
                          }}
                        >
                          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/>
                          </svg>
                        </button>
                      </span>
                    </button>
                  ))}
                  {sessions.length === 0 && <div className="meta">No saved sessions yet.</div>}
                </div>
              </>
            )}

            {/* Profile tab */}
            {sidebarTab === "profile" && (
              <>
                <div className="meta">seed status: {seedStatus}</div>
                <div className="row">
                  <label>Name</label>
                  <input
                    value={seedDraft.companion_name}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, companion_name: e.target.value }))
                    }
                    placeholder="Companion name"
                  />
                </div>
                <div className="row">
                  <label>Backstory</label>
                  <textarea
                    rows={2}
                    value={seedDraft.backstory}
                    onChange={(e) => setSeedDraft((prev) => ({ ...prev, backstory: e.target.value }))}
                    placeholder="Backstory"
                  />
                </div>
                <div className="row">
                  <label>Traits</label>
                  <input
                    value={seedDraft.character_traits}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, character_traits: e.target.value }))
                    }
                    placeholder="Comma-separated"
                  />
                </div>
                <div className="row">
                  <label>Goals</label>
                  <input
                    value={seedDraft.goals}
                    onChange={(e) => setSeedDraft((prev) => ({ ...prev, goals: e.target.value }))}
                    placeholder="Comma-separated"
                  />
                </div>
                <div className="row">
                  <label>Relationship</label>
                  <input
                    value={seedDraft.relationship_setup}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, relationship_setup: e.target.value }))
                    }
                    placeholder="Relationship setup"
                  />
                </div>
                <div className="row">
                  <label>User</label>
                  <textarea
                    rows={2}
                    value={seedDraft.user_description}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, user_description: e.target.value }))
                    }
                    placeholder="Name, pronouns, interests"
                  />
                </div>
                <div className="row">
                  <label>Notes</label>
                  <input
                    value={seedDraft.notes}
                    onChange={(e) => setSeedDraft((prev) => ({ ...prev, notes: e.target.value }))}
                    placeholder="Notes"
                  />
                  <button type="button" onClick={() => void onSaveSeed()}>
                    Save Seed
                  </button>
                </div>
              </>
            )}

            {/* Knowledge tab */}
            {sidebarTab === "knowledge" && (
              <div className="drawer-panel-host">
                <KnowledgePanel
                  isOpen={isKnowledgeOpen}
                  onToggleOpen={() => setIsKnowledgeOpen((prev) => !prev)}
                  onRefresh={() => void refreshKnowledge()}
                  facts={knowledgeFacts}
                  graph={knowledgeGraph}
                  monologue={knowledgeMonologue}
                  status={knowledgeStatus}
                  companionName={companionLabel}
                />
              </div>
            )}

            {/* Debug tab */}
            {sidebarTab === "debug" && (
              <div className="drawer-panel-host">
                <DebugPanel
                  isOpen={isDebugOpen}
                  onToggleOpen={() => setIsDebugOpen((prev) => !prev)}
                  onRefresh={() => void refreshDebug()}
                  traces={debugTraces}
                  status={debugStatus}
                  showRawPrompt={showRawPrompt}
                  setShowRawPrompt={setShowRawPrompt}
                  verbose={verboseDebug}
                  setVerbose={setVerboseDebug}
                />
                <button
                  type="button"
                  className="row"
                  style={{ marginTop: 8, justifyContent: "center" }}
                  onClick={() => {
                    void refreshInferenceLogs();
                    setInferenceLogsOpen(true);
                  }}
                >
                  Inference Logs
                </button>
              </div>
            )}
          </div>
        </aside>
      )}

      {/* Main chat area */}
      <main className="chat-main">
        <header className="topbar">
          <button
            type="button"
            className="topbar-menu"
            onClick={() => setSidebarOpen((prev) => !prev)}
          >
            ☰
          </button>
          <span className="topbar-name">{companionLabel}</span>
          <span className="meta topbar-status">{status}</span>
          <button
            type="button"
            className="topbar-action"
            onClick={() => void onNewSession()}
          >
            + New
          </button>
        </header>

        <AffectPanel affect={knowledgeAffect} />

        <div ref={messagesPaneRef} className="panel messages">
          {messages.map((msg, idx) => {
            const speakerName = msg.role === "assistant"
              ? (msg.name || companionLabel)
              : msg.role === "user" ? "You" : msg.role;
            const isNpc = msg.role === "assistant" && msg.name && msg.name !== companionLabel;
            const speakerClass = msg.role === "assistant"
              ? (isNpc ? "message-speaker message-speaker--npc" : "message-speaker")
              : msg.role === "user"
                ? "message-speaker message-speaker--user"
                : "message-speaker message-speaker--system";
            return (
              <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
                <span className={speakerClass}>{speakerName}</span>
                <div>{msg.content}</div>
              </div>
            );
          })}
          {(isStreaming || pendingAssistant.length > 0) && (
            <div className="message assistant">
              {(() => {
                const name = pendingAssistant ? (pendingCharName || companionLabel) : companionLabel;
                const isNpc = pendingCharName && pendingCharName !== companionLabel;
                return <span className={isNpc ? "message-speaker message-speaker--npc" : "message-speaker"}>{name}</span>;
              })()}
              <div>{pendingAssistant || "..."}</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <div className="composer-input-wrap">
            <textarea
              ref={composerRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                const q = getActiveMention(e.target.value, e.target.selectionStart);
                setMentionQuery(q);
                setMentionIndex(0);
              }}
              onKeyDown={onInputKeyDown}
              placeholder="Type a message..."
              rows={2}
            />
            {mentionQuery !== null && mentionMatches.length > 0 && (
              <ul className="mention-dropdown">
                {mentionMatches.map((name, i) => (
                  <li
                    key={name}
                    className={`mention-option ${i === mentionIndex ? "mention-option--active" : ""}`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      completeMention(name);
                    }}
                  >
                    {name}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button type="submit" disabled={streamDisabled}>
            Send
          </button>
        </form>
      </main>

      {inferenceLogsOpen && (
        <InferenceLogPanel
          entries={inferenceLogs}
          status={inferenceLogsStatus}
          sessionId={sessionId}
          onClose={() => setInferenceLogsOpen(false)}
          onRefresh={() => void refreshInferenceLogs()}
        />
      )}
    </div>
  );
}
