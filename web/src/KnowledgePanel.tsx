import { GraphRelation, MemoryFact } from "./knowledgeApi";

type KnowledgePanelProps = {
  isOpen: boolean;
  onToggleOpen: () => void;
  onRefresh: () => void;
  facts: MemoryFact[];
  graph: GraphRelation[];
  monologue: string | null;
  status: string;
  companionName: string;
};

function relationLabel(rel: GraphRelation): string {
  return `${rel.source} —${rel.relation}→ ${rel.target}`;
}

/** Try to extract the character name a companion-kind fact is about. */
function factSubject(content: string): string {
  // Companion facts are stored as "{Name} did something" — grab the first word(s)
  // before common verb patterns.
  const match = content.match(/^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s/);
  return match ? match[1] : "";
}

export function KnowledgePanel(props: KnowledgePanelProps) {
  const { isOpen, onToggleOpen, onRefresh, facts, graph, monologue, status, companionName } = props;

  const userFacts = facts.filter((f) => f.kind !== "companion");
  const allCompanionFacts = facts.filter((f) => f.kind === "companion");

  // Split companion facts: primary companion vs NPC characters
  const primaryFacts: MemoryFact[] = [];
  const npcFactsByName: Record<string, MemoryFact[]> = {};

  for (const fact of allCompanionFacts) {
    const subject = factSubject(fact.content);
    if (subject === companionName || subject === "Companion" || subject === "") {
      primaryFacts.push(fact);
    } else {
      const bucket = npcFactsByName[subject] ?? [];
      bucket.push(fact);
      npcFactsByName[subject] = bucket;
    }
  }
  const npcNames = Object.keys(npcFactsByName).sort();

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
        {status} · {userFacts.length} facts · {allCompanionFacts.length} self-facts · {graph.length} relations
      </div>

      {isOpen && (
        <div className="knowledge-scroll">
          {monologue && (
            <div className="knowledge-section" style={{ marginTop: 0 }}>
              <div className="knowledge-heading">Internal Monologue</div>
              <div className="knowledge-mono">{monologue}</div>
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

          {primaryFacts.length > 0 && (
            <div className="knowledge-section">
              <div className="knowledge-heading">{companionName} Self-Facts ({primaryFacts.length})</div>
              {primaryFacts.map((fact, idx) => (
                <div key={fact.memory_id ?? idx} className="knowledge-item companion">
                  {fact.content}
                </div>
              ))}
            </div>
          )}

          {npcNames.map((name) => (
            <div key={name} className="knowledge-section">
              <div className="knowledge-heading">{name} Facts ({npcFactsByName[name].length})</div>
              {npcFactsByName[name].map((fact, idx) => (
                <div key={fact.memory_id ?? idx} className="knowledge-item npc">
                  {fact.content}
                </div>
              ))}
            </div>
          ))}

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
