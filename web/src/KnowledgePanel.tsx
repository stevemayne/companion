import { CharacterState, GraphRelation, MemoryFact, WorldState } from "./knowledgeApi";

type KnowledgePanelProps = {
  isOpen: boolean;
  onToggleOpen: () => void;
  onRefresh: () => void;
  facts: MemoryFact[];
  graph: GraphRelation[];
  monologue: string | null;
  world: WorldState | null;
  status: string;
};

function relationLabel(rel: GraphRelation): string {
  return `${rel.source} —${rel.relation}→ ${rel.target}`;
}

function characterSummary(state: CharacterState): string | null {
  const parts: string[] = [];
  if (state.activity) parts.push(state.activity);
  if (state.position) parts.push(state.position);
  if (state.clothing) parts.push(`wearing ${state.clothing}`);
  if (state.mood_apparent) parts.push(`looks ${state.mood_apparent}`);
  for (const note of state.appearance) parts.push(note);
  if (state.location) parts.push(`in ${state.location}`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function KnowledgePanel(props: KnowledgePanelProps) {
  const { isOpen, onToggleOpen, onRefresh, facts, graph, monologue, world, status } = props;

  const userFacts = facts.filter((f) => f.kind !== "companion");
  const companionFacts = facts.filter((f) => f.kind === "companion");

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
        {status} · {userFacts.length} facts · {companionFacts.length} self-facts · {graph.length} relations
      </div>

      {isOpen && (
        <div className="knowledge-scroll">
          {monologue && (
            <div className="knowledge-section" style={{ marginTop: 0 }}>
              <div className="knowledge-heading">Internal Monologue</div>
              <div className="knowledge-mono">{monologue}</div>
            </div>
          )}

          {world && (
            <div className="knowledge-section">
              <div className="knowledge-heading">Scene</div>
              {world.environment && (
                <div className="knowledge-item">{world.environment}</div>
              )}
              {world.time_of_day && (
                <div className="knowledge-item">Time: {world.time_of_day}</div>
              )}
              {characterSummary(world.self_state) && (
                <div className="knowledge-item">
                  <strong>Self:</strong> {characterSummary(world.self_state)}
                </div>
              )}
              {characterSummary(world.user_state) && (
                <div className="knowledge-item">
                  <strong>User:</strong> {characterSummary(world.user_state)}
                </div>
              )}
              {Object.entries(world.other_characters).map(([name, state]) => {
                const summary = characterSummary(state);
                return summary ? (
                  <div key={name} className="knowledge-item">
                    <strong>{name}:</strong> {summary}
                  </div>
                ) : null;
              })}
              {world.recent_events.length > 0 && (
                <div style={{ marginTop: "0.25rem" }}>
                  <div className="meta">Recent events</div>
                  {world.recent_events.map((evt, i) => (
                    <div key={i} className="knowledge-item">{evt}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="knowledge-section">
            <div className="knowledge-heading">Facts ({userFacts.length})</div>
            {userFacts.length === 0 && <div className="meta">No facts extracted yet.</div>}
            {userFacts.map((fact, idx) => (
              <div key={fact.memory_id ?? idx} className="knowledge-item fact">
                {fact.content}
              </div>
            ))}
          </div>

          {companionFacts.length > 0 && (
            <div className="knowledge-section">
              <div className="knowledge-heading">Companion Self-Facts ({companionFacts.length})</div>
              {companionFacts.map((fact, idx) => (
                <div key={fact.memory_id ?? idx} className="knowledge-item companion">
                  {fact.content}
                </div>
              ))}
            </div>
          )}

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
