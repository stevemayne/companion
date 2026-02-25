import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";

import { DEFAULT_NOTES, DEFAULT_SEED } from "./defaultSeed";
import { SeedPayload, upsertSeed } from "./seedApi";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type SseEventPayload = Record<string, unknown>;

type SeedDraft = {
  companion_name: string;
  backstory: string;
  character_traits: string;
  goals: string;
  relationship_setup: string;
  notes: string;
};

const STORAGE_KEY = "aether.seed.defaults.v1";

function parseEventData(raw: string): SseEventPayload {
  try {
    return JSON.parse(raw) as SseEventPayload;
  } catch {
    return {};
  }
}

function toDraft(seed: SeedPayload, notes: string): SeedDraft {
  return {
    companion_name: seed.companion_name,
    backstory: seed.backstory,
    character_traits: seed.character_traits.join(", "),
    goals: seed.goals.join(", "),
    relationship_setup: seed.relationship_setup,
    notes
  };
}

function toPayload(draft: SeedDraft): { seed: SeedPayload; notes: string } {
  const splitList = (value: string): string[] =>
    value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

  return {
    seed: {
      companion_name: draft.companion_name.trim() || DEFAULT_SEED.companion_name,
      backstory: draft.backstory.trim() || DEFAULT_SEED.backstory,
      character_traits: splitList(draft.character_traits),
      goals: splitList(draft.goals),
      relationship_setup: draft.relationship_setup.trim() || DEFAULT_SEED.relationship_setup
    },
    notes: draft.notes.trim()
  };
}

function loadSeedDraft(): SeedDraft {
  const fallback = toDraft(DEFAULT_SEED, DEFAULT_NOTES);
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
      notes: parsed.notes || fallback.notes
    };
  } catch {
    return fallback;
  }
}

export function App() {
  const [sessionId, setSessionId] = useState<string>(crypto.randomUUID());
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingAssistant, setPendingAssistant] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("idle");
  const [seedStatus, setSeedStatus] = useState("not seeded");
  const [seedDraft, setSeedDraft] = useState<SeedDraft>(() => loadSeedDraft());
  const [isProfileOpen, setIsProfileOpen] = useState(true);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seedDraft));
  }, [seedDraft]);

  const streamDisabled = useMemo(
    () => isStreaming || !sessionId.trim() || !input.trim(),
    [isStreaming, sessionId, input]
  );

  const bootstrapSeed = async (targetSessionId: string): Promise<void> => {
    const payload = toPayload(seedDraft);
    setSeedStatus("seeding...");
    try {
      const response = await upsertSeed(targetSessionId, payload);
      if (response.ok) {
        setSeedStatus("seeded");
        setIsProfileOpen(false);
      } else {
        setSeedStatus(`seed error (${response.status})`);
      }
    } catch {
      setSeedStatus("seed error (network)");
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || !sessionId.trim()) {
      return;
    }

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setPendingAssistant("");
    setInput("");
    setIsStreaming(true);
    setStatus("connecting");
    let assistantBuffer = "";

    const url = new URL("/v1/chat/stream", window.location.origin);
    url.searchParams.set("chat_session_id", sessionId.trim());
    url.searchParams.set("message", text);

    const source = new EventSource(url.toString());

    source.addEventListener("start", () => {
      setStatus("streaming");
    });

    source.addEventListener("delta", (evt) => {
      const payload = parseEventData((evt as MessageEvent).data);
      const chunk = String(payload.chunk ?? "");
      assistantBuffer = `${assistantBuffer}${chunk}`;
      setPendingAssistant(assistantBuffer);
    });

    source.addEventListener("done", () => {
      if (assistantBuffer.length > 0) {
        setMessages((prev) => [...prev, { role: "assistant", content: assistantBuffer }]);
      }
      setPendingAssistant("");
      setStatus("done");
      setIsStreaming(false);
      source.close();
    });

    source.onerror = () => {
      if (assistantBuffer.length > 0) {
        setMessages((prev) => [...prev, { role: "assistant", content: assistantBuffer }]);
      }
      setPendingAssistant("");
      setStatus("error");
      setIsStreaming(false);
      source.close();
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
    const newSessionId = crypto.randomUUID();
    setSessionId(newSessionId);
    setMessages([]);
    setPendingAssistant("");
    setStatus("idle");
    await bootstrapSeed(newSessionId);
  };

  const onSaveSeed = async () => {
    if (!sessionId.trim()) {
      return;
    }
    await bootstrapSeed(sessionId.trim());
  };

  return (
    <div className="app">
      <div className="panel">
        <h1>Project Aether</h1>
        <p className="meta">SSE streaming chat client (React + FastAPI)</p>
        <div className="row">
          <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
          <button type="button" onClick={() => void onNewSession()}>
            New Session
          </button>
        </div>
        <div className="meta">seed status: {seedStatus}</div>
      </div>

      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Companion Profile</h2>
          <button type="button" onClick={() => setIsProfileOpen((prev) => !prev)}>
            {isProfileOpen ? "Collapse" : "Expand"}
          </button>
        </div>
        {isProfileOpen && (
          <>
            <div className="row">
              <input
                value={seedDraft.companion_name}
                onChange={(e) => setSeedDraft((prev) => ({ ...prev, companion_name: e.target.value }))}
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
                onChange={(e) => setSeedDraft((prev) => ({ ...prev, character_traits: e.target.value }))}
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
      </div>

      <div className="panel messages">
        {messages.map((msg, idx) => (
          <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
            <strong>{msg.role}</strong>
            <div>{msg.content}</div>
          </div>
        ))}
        {(isStreaming || pendingAssistant.length > 0) && (
          <div className="message assistant">
            <strong>assistant</strong>
            <div>{pendingAssistant || "..."}</div>
          </div>
        )}
      </div>

      <form className="panel" onSubmit={onSubmit}>
        <div className="row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="Type a message..."
            rows={3}
          />
        </div>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
          <span className="meta">status: {status}</span>
          <button type="submit" disabled={streamDisabled}>
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
