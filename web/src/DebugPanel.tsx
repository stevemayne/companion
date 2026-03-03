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
              const raw = trace as Record<string, unknown>;
              const isAgentTrace = "agent" in raw && !("turn_trace" in raw);

              if (isAgentTrace) {
                const facts = raw.facts as string[] | undefined;
                return (
                  <div key={trace.trace_id} className="message system">
                    <div className="meta">trace_id: {trace.trace_id}</div>
                    <div className="meta">agent: {String(raw.agent)}</div>
                    <div className="meta">
                      provider: {String(raw.provider)}
                      {raw.fallback_reason ? ` (fallback: ${String(raw.fallback_reason)})` : ""}
                    </div>
                    {raw.latency_ms != null && (
                      <div className="meta">latency: {String(raw.latency_ms)}ms</div>
                    )}
                    {facts && facts.length > 0 && (
                      <div className="meta">
                        facts:{facts.map((f, i) => <div key={i}>  - {f}</div>)}
                      </div>
                    )}
                    {facts && facts.length === 0 && (
                      <div className="meta">facts: (none extracted)</div>
                    )}
                  </div>
                );
              }

              const turnTrace = raw.turn_trace as Record<string, unknown> | undefined;
              const prompt = turnTrace?.prompt as Record<string, unknown> | undefined;
              const preprocess = turnTrace?.preprocess as Record<string, unknown> | undefined;
              const retrieval = turnTrace?.retrieval as Record<string, unknown> | undefined;
              const writes = turnTrace?.writes as Record<string, unknown> | undefined;
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
                  {writes?.affect != null && (
                    <div className="meta">
                      affect: mood={String((writes.affect as Record<string, unknown>).mood)}
                      {" "}valence={String((writes.affect as Record<string, unknown>).valence)}
                      {" "}trust={String((writes.affect as Record<string, unknown>).trust)}
                      {" "}comfort={String((writes.affect as Record<string, unknown>).comfort_level)}
                      {" "}engagement={String((writes.affect as Record<string, unknown>).engagement)}
                      {" "}shyness={String((writes.affect as Record<string, unknown>).shyness)}
                      {" "}patience={String((writes.affect as Record<string, unknown>).patience)}
                      {" "}curiosity={String((writes.affect as Record<string, unknown>).curiosity)}
                      {" "}vulnerability={String((writes.affect as Record<string, unknown>).vulnerability)}
                      {(() => {
                        const triggers = (writes.affect as Record<string, unknown>).recent_triggers as string[] | undefined;
                        return triggers && triggers.length > 0 ? ` triggers=[${triggers.join("; ")}]` : "";
                      })()}
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
