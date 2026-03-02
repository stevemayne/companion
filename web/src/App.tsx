import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { AffectPanel } from "./AffectPanel";
import { fetchDebugTraces } from "./debugApi";
import { DebugPanel } from "./DebugPanel";
import { DEFAULT_NOTES, DEFAULT_SEED } from "./defaultSeed";
import { CompanionAffect, fetchKnowledge, GraphRelation, MemoryFact } from "./knowledgeApi";
import { KnowledgePanel } from "./KnowledgePanel";
import { fetchMemory } from "./memoryApi";
import { SeedPayload, upsertSeed } from "./seedApi";
import { fetchSessions, SessionSummary } from "./sessionApi";
import { DebugTrace } from "./debugTypes";

type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
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
      relationship_setup: draft.relationship_setup.trim() || DEFAULT_SEED.relationship_setup
    },
    user_description: userDesc || undefined,
    notes: draft.notes.trim()
  };
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
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("idle");

  const [seedStatus, setSeedStatus] = useState("not seeded");
  const [seedDraft, setSeedDraft] = useState<SeedDraft>(() => loadSeedDraft());

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("sessions");

  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [debugStatus, setDebugStatus] = useState("idle");
  const [debugTraces, setDebugTraces] = useState<DebugTrace[]>([]);
  const [showRawPrompt, setShowRawPrompt] = useState(false);
  const [verboseDebug, setVerboseDebug] = useState(false);

  const [isKnowledgeOpen, setIsKnowledgeOpen] = useState(false);
  const [knowledgeStatus, setKnowledgeStatus] = useState("idle");
  const [knowledgeFacts, setKnowledgeFacts] = useState<MemoryFact[]>([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState<GraphRelation[]>([]);
  const [knowledgeMonologue, setKnowledgeMonologue] = useState<string | null>(null);
  const [knowledgeAffect, setKnowledgeAffect] = useState<CompanionAffect | null>(null);

  const messagesPaneRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const currentSessionRef = useRef(sessionId);

  useEffect(() => {
    currentSessionRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seedDraft));
  }, [seedDraft]);

  useEffect(() => {
    const pane = messagesPaneRef.current;
    if (!pane) {
      return;
    }
    pane.scrollTop = pane.scrollHeight;
  }, [messages, pendingAssistant, isStreaming]);

  const streamDisabled = useMemo(
    () => isStreaming || !sessionId.trim() || !input.trim(),
    [isStreaming, sessionId, input]
  );
  const companionLabel = seedDraft.companion_name.trim() || DEFAULT_SEED.companion_name;

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
          content: item.content
        }))
      );

      if (memory.seed_context) {
        setSeedDraft(toDraft(memory.seed_context.seed, memory.seed_context.user_description ?? "", memory.seed_context.notes ?? DEFAULT_NOTES));
        setSeedStatus("seeded");
      } else {
        setSeedStatus("not seeded");
        setSidebarOpen(true);
        setSidebarTab("profile");
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
    setIsStreaming(true);

    let assistantBuffer = "";
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

    source.addEventListener("done", () => {
      if (streamRef.current === source) {
        streamRef.current = null;
      }
      if (currentSessionRef.current !== selectedSession) {
        source.close();
        return;
      }
      if (assistantBuffer.length > 0) {
        setMessages((prev) => [...prev, { role: "assistant", content: assistantBuffer }]);
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
        setMessages((prev) => [...prev, { role: "assistant", content: assistantBuffer }]);
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

  const onInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
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
                  <input
                    value={seedDraft.companion_name}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, companion_name: e.target.value }))
                    }
                    placeholder="Companion name"
                  />
                </div>
                <div className="row">
                  <textarea
                    rows={2}
                    value={seedDraft.backstory}
                    onChange={(e) => setSeedDraft((prev) => ({ ...prev, backstory: e.target.value }))}
                    placeholder="Backstory"
                  />
                </div>
                <div className="row">
                  <input
                    value={seedDraft.character_traits}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, character_traits: e.target.value }))
                    }
                    placeholder="Traits (comma-separated)"
                  />
                </div>
                <div className="row">
                  <input
                    value={seedDraft.goals}
                    onChange={(e) => setSeedDraft((prev) => ({ ...prev, goals: e.target.value }))}
                    placeholder="Goals (comma-separated)"
                  />
                </div>
                <div className="row">
                  <input
                    value={seedDraft.relationship_setup}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, relationship_setup: e.target.value }))
                    }
                    placeholder="Relationship setup"
                  />
                </div>
                <div className="row">
                  <textarea
                    rows={2}
                    value={seedDraft.user_description}
                    onChange={(e) =>
                      setSeedDraft((prev) => ({ ...prev, user_description: e.target.value }))
                    }
                    placeholder="About the user (e.g. name, pronouns, interests)"
                  />
                </div>
                <div className="row">
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
          {messages.map((msg, idx) => (
            <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
              {msg.role === "assistant" && <strong>{companionLabel}</strong>}
              {msg.role === "system" && <strong>system</strong>}
              {msg.role === "tool" && <strong>tool</strong>}
              <div>{msg.content}</div>
            </div>
          ))}
          {(isStreaming || pendingAssistant.length > 0) && (
            <div className="message assistant">
              <strong>{companionLabel}</strong>
              <div>{pendingAssistant || "..."}</div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="Type a message..."
            rows={2}
          />
          <button type="submit" disabled={streamDisabled}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
