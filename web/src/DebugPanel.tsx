import { DebugTrace } from "./debugTypes";

type DebugPanelProps = {
  isOpen: boolean;
  onToggleOpen: () => void;
  onRefresh: () => void;
  traces: DebugTrace[];
  status: string;
  showRawPrompt: boolean;
  setShowRawPrompt: (value: boolean) => void;
  verbose: boolean;
  setVerbose: (value: boolean) => void;
};

export function DebugPanel(props: DebugPanelProps) {
  const {
    isOpen,
    onToggleOpen,
    onRefresh,
    traces,
    status,
    showRawPrompt,
    setShowRawPrompt,
    verbose,
    setVerbose
  } = props;

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Debug</h2>
        <div className="row" style={{ width: "auto" }}>
          <button type="button" onClick={onRefresh}>
            Refresh
          </button>
          <button type="button" onClick={onToggleOpen}>
            {isOpen ? "Collapse" : "Expand"}
          </button>
        </div>
      </div>
      <div className="meta">debug status: {status}</div>

      {isOpen && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <label className="meta" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={showRawPrompt}
                onChange={(event) => setShowRawPrompt(event.target.checked)}
              />
              show raw prompt
            </label>
            <label className="meta" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={verbose}
                onChange={(event) => setVerbose(event.target.checked)}
              />
              verbose trace
            </label>
          </div>

          <div className="messages" style={{ maxHeight: 260 }}>
            {traces.length === 0 && <div className="meta">No traces for this session yet.</div>}
            {traces.map((trace) => {
              const turnTrace = trace.turn_trace as Record<string, unknown>;
              const prompt = turnTrace.prompt as Record<string, unknown> | undefined;
              const preprocess = turnTrace.preprocess as Record<string, unknown> | undefined;
              const retrieval = turnTrace.retrieval as Record<string, unknown> | undefined;
              const writes = turnTrace.writes as Record<string, unknown> | undefined;
              return (
                <div key={trace.trace_id} className="message assistant">
                  <div className="meta">trace_id: {trace.trace_id}</div>
                  <div className="meta">seed_version: {String(trace.seed_version)}</div>
                  <div className="meta">safety: {(trace.safety_transforms || []).join(",") || "none"}</div>
                  <div>
                    <strong>user</strong>: {trace.user_message}
                  </div>
                  <div>
                    <strong>assistant</strong>: {trace.assistant_message}
                  </div>
                  {preprocess && (
                    <div className="meta">
                      preprocess: intent={String(preprocess.intent)} emotion={String(preprocess.emotion)}
                      entities={JSON.stringify(preprocess.entities)}
                    </div>
                  )}
                  {retrieval && (
                    <div className="meta">
                      retrieval: semantic={JSON.stringify(retrieval.semantic_items)} graph=
                      {JSON.stringify(retrieval.graph_relations)}
                    </div>
                  )}
                  {prompt && (
                    <div className="meta">
                      prompt: {showRawPrompt ? String(prompt.raw ?? "") : String(prompt.summary ?? "")}
                    </div>
                  )}
                  {verbose && writes && (
                    <div className="meta">writes: {JSON.stringify(writes)}</div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
