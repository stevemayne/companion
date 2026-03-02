import { GraphRelation, MemoryFact } from "./knowledgeApi";

type KnowledgePanelProps = {
  isOpen: boolean;
  onToggleOpen: () => void;
  onRefresh: () => void;
  facts: MemoryFact[];
  graph: GraphRelation[];
  monologue: string | null;
  status: string;
};

function relationLabel(rel: GraphRelation): string {
  return `${rel.source} —${rel.relation}→ ${rel.target}`;
}

export function KnowledgePanel(props: KnowledgePanelProps) {
  const { isOpen, onToggleOpen, onRefresh, facts, graph, monologue, status } = props;

  // Group graph relations by type for readability
  const byType: Record<string, GraphRelation[]> = {};
  for (const rel of graph) {
    const bucket = byType[rel.relation] ?? [];
    bucket.push(rel);
    byType[rel.relation] = bucket;
  }
  const relationTypes = Object.keys(byType).sort();

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Knowledge</h2>
        <div className="row" style={{ width: "auto" }}>
          <button type="button" onClick={onRefresh}>
            Refresh
          </button>
          <button type="button" onClick={onToggleOpen}>
            {isOpen ? "Collapse" : "Expand"}
          </button>
        </div>
      </div>
      <div className="meta">
        {status} · {facts.length} facts · {graph.length} relations
      </div>

      {isOpen && (
        <div style={{ maxHeight: 320, overflowY: "auto", marginTop: 8 }}>
          {monologue && (
            <div className="knowledge-section" style={{ marginTop: 0 }}>
              <div className="knowledge-heading">Internal Monologue</div>
              <div className="knowledge-mono">{monologue}</div>
            </div>
          )}

          <div className="knowledge-section">
            <div className="knowledge-heading">Facts ({facts.length})</div>
            {facts.length === 0 && <div className="meta">No facts extracted yet.</div>}
            {facts.map((fact, idx) => (
              <div key={fact.memory_id ?? idx} className="knowledge-item fact">
                {fact.content}
              </div>
            ))}
          </div>

          <div className="knowledge-section">
            <div className="knowledge-heading">Graph ({graph.length})</div>
            {graph.length === 0 && <div className="meta">No relations stored yet.</div>}
            {relationTypes.map((type) => (
              <div key={type} className="knowledge-group">
                <div className="meta">{type}</div>
                {byType[type].map((rel, idx) => (
                  <div key={`${type}-${idx}`} className="knowledge-item relation">
                    {relationLabel(rel)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
