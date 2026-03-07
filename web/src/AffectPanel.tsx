import { CompanionAffect } from "./knowledgeApi";

type AffectPanelProps = {
  affect: CompanionAffect | null;
};

type BarDef = {
  label: string;
  value: number;
  max: number;
  color: string;
};

function AffectBar({ label, value, max, color }: BarDef) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="affect-bar-row">
      <span className="affect-bar-label">{label}</span>
      <div className="affect-bar-track">
        <div className="affect-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="affect-bar-value">{value.toFixed(1)}</span>
    </div>
  );
}

export function AffectPanel({ affect }: AffectPanelProps) {
  if (!affect) {
    return null;
  }

  const bars: BarDef[] = [
    { label: "Trust", value: affect.trust, max: 10, color: "#4a7ab5" },
    { label: "Closeness", value: affect.closeness, max: 10, color: "#5a9a54" },
    { label: "Engagement", value: affect.engagement, max: 10, color: "#b5994a" },
    { label: "Dominance", value: affect.dominance, max: 1, color: "#7a8ab5" },
    { label: "Arousal", value: affect.arousal, max: 1, color: "#8a5ab5" },
  ];

  const valenceLabel = affect.valence > 0.2
    ? "positive"
    : affect.valence < -0.2
      ? "negative"
      : "neutral";

  return (
    <div className="panel affect-panel">
      <h2 style={{ margin: 0 }}>Companion State</h2>

      <div className="affect-mood">
        <span className="affect-mood-label">{affect.mood}</span>
        <span className="meta">valence: {affect.valence.toFixed(2)} ({valenceLabel})</span>
      </div>

      <div className="affect-bars">
        {bars.map((bar) => (
          <AffectBar key={bar.label} {...bar} />
        ))}
      </div>

      {affect.recent_triggers.length > 0 && (
        <div className="affect-triggers">
          <div className="meta">recent triggers:</div>
          {affect.recent_triggers.map((trigger, i) => (
            <div key={i} className="affect-trigger-item">{trigger}</div>
          ))}
        </div>
      )}
    </div>
  );
}
