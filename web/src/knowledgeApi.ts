export type MemoryFact = {
  memory_id: string;
  chat_session_id: string;
  companion_id: string | null;
  kind: string;
  content: string;
  score: number | null;
  created_at: string;
};

export type GraphRelation = {
  chat_session_id: string;
  companion_id: string | null;
  source: string;
  relation: string;
  target: string;
  confidence: number;
};

export type CompanionAffect = {
  mood: string;
  valence: number;
  arousal: number;
  dominance: number;
  trust: number;
  closeness: number;
  engagement: number;
  recent_triggers: string[];
};

export type CharacterState = {
  clothing: string | null;
  location: string | null;
  activity: string | null;
  position: string | null;
  appearance: string[];
  mood_apparent: string | null;
};

export type WorldState = {
  self_state: CharacterState;
  user_state: CharacterState;
  other_characters: Record<string, CharacterState>;
  environment: string | null;
  time_of_day: string | null;
  recent_events: string[];
};

export type KnowledgeResponse = {
  chat_session_id: string;
  companion_id: string | null;
  facts: MemoryFact[];
  graph: GraphRelation[];
  monologue: string | null;
  affect: CompanionAffect | null;
  world: WorldState | null;
};

export async function fetchKnowledge(chatSessionId: string): Promise<KnowledgeResponse> {
  const response = await fetch(`/v1/knowledge/${chatSessionId}`);
  if (!response.ok) {
    throw new Error(`knowledge fetch failed (${response.status})`);
  }
  return (await response.json()) as KnowledgeResponse;
}
