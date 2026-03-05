import { useState } from "react";

import { InferenceLogEntry } from "./logsApi";

type InferenceLogPanelProps = {
  entries: InferenceLogEntry[];
  status: string;
  sessionId: string;
  onClose: () => void;
  onRefresh: () => void;
};

function LogEntryCard({ entry, index }: { entry: InferenceLogEntry; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const usage = entry.usage;
  const truncated = entry.finish_reason === "length";

  return (
    <div className={`log-entry ${truncated ? "log-entry--truncated" : ""}`}>
      <button
        type="button"
        className="log-entry-header"
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span className="log-entry-index">#{index + 1}</span>
        <span className="log-meta-pill">{entry.model}</span>
        <span className={`log-meta-pill ${truncated ? "log-meta-pill--warn" : ""}`}>
          {entry.finish_reason ?? "unknown"}
        </span>
        <span className="log-meta-pill">
          {usage?.prompt_tokens ?? "?"} / {usage?.completion_tokens ?? "?"} tokens
        </span>
        <span className="log-meta-pill">{entry.duration_ms.toFixed(0)}ms</span>
        <span className="log-meta-pill">{entry.message_count} msgs</span>
        {entry.max_tokens != null && (
          <span className="log-meta-pill">max={entry.max_tokens}</span>
        )}
        <span className="log-entry-chevron">{expanded ? "\u25BC" : "\u25B6"}</span>
      </button>

      {expanded && (
        <div className="log-entry-body">
          <div className="log-section">
            <div className="log-section-heading">Request Messages</div>
            {entry.request_messages.map((msg, i) => (
              <div key={i} className="log-message">
                <span className="log-message-role">{msg.role}</span>
                <pre className="log-message-content">{msg.content}</pre>
              </div>
            ))}
          </div>
          <div className="log-section">
            <div className="log-section-heading">Response</div>
            <pre className="log-message-content">{entry.response_content}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export function InferenceLogPanel(props: InferenceLogPanelProps) {
  const { entries, status, sessionId, onClose, onRefresh } = props;

  return (
    <div className="log-overlay">
      <div className="log-header">
        <div className="log-header-left">
          <h2 style={{ margin: 0 }}>Inference Logs</h2>
          <code className="log-session-id">{sessionId}</code>
          <span className="meta">{status}</span>
        </div>
        <div className="log-header-right">
          <button type="button" onClick={onRefresh}>
            Refresh
          </button>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <div className="log-body">
        {entries.length === 0 && (
          <div className="meta" style={{ padding: 16 }}>
            No inference logs for this session.
          </div>
        )}
        {entries.map((entry, i) => (
          <LogEntryCard key={i} entry={entry} index={i} />
        ))}
      </div>
    </div>
  );
}
