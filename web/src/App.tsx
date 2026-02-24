import { FormEvent, useMemo, useState } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type SseEventPayload = Record<string, unknown>;

function parseEventData(raw: string): SseEventPayload {
  try {
    return JSON.parse(raw) as SseEventPayload;
  } catch {
    return {};
  }
}

export function App() {
  const [sessionId, setSessionId] = useState(crypto.randomUUID());
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("idle");

  const streamDisabled = useMemo(
    () => isStreaming || !sessionId.trim() || !input.trim(),
    [isStreaming, sessionId, input]
  );

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || !sessionId.trim()) {
      return;
    }

    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);
    setStatus("connecting");

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
      setMessages((prev) => {
        if (prev.length === 0) {
          return prev;
        }
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role !== "assistant") {
          return prev;
        }
        updated[updated.length - 1] = { ...last, content: `${last.content}${chunk}` };
        return updated;
      });
    });

    source.addEventListener("done", () => {
      setStatus("done");
      setIsStreaming(false);
      source.close();
    });

    source.onerror = () => {
      setStatus("error");
      setIsStreaming(false);
      source.close();
    };
  };

  const onNewSession = () => {
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setStatus("idle");
  };

  return (
    <div className="app">
      <div className="panel">
        <h1>Project Aether</h1>
        <p className="meta">SSE streaming chat client (React + FastAPI)</p>
        <div className="row">
          <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
          <button type="button" onClick={onNewSession}>
            New Session
          </button>
        </div>
      </div>

      <div className="panel messages">
        {messages.map((msg, idx) => (
          <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
            <strong>{msg.role}</strong>
            <div>{msg.content || (msg.role === "assistant" && isStreaming ? "..." : "")}</div>
          </div>
        ))}
      </div>

      <form className="panel" onSubmit={onSubmit}>
        <div className="row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
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
