export type MemoryFact = {
  memory_id: string;
  chat_session_id: string;
  kind: string;
  content: string;
  score: number | null;
  created_at: string;
};

export type GraphRelation = {
  chat_session_id: string;
  source: string;
  relation: string;
  target: string;
  confidence: number;
};

export type CompanionAffect = {
  mood: string;
  valence: number;
  arousal: number;
  comfort_level: number;
  trust: number;
  attraction: number;
  engagement: number;
  shyness: number;
  patience: number;
  curiosity: number;
  vulnerability: number;
  recent_triggers: string[];
};

export type KnowledgeResponse = {
  chat_session_id: string;
  facts: MemoryFact[];
  graph: GraphRelation[];
  monologue: string | null;
  affect: CompanionAffect | null;
};

export async function fetchKnowledge(chatSessionId: string): Promise<KnowledgeResponse> {
  const response = await fetch(`/v1/knowledge/${chatSessionId}`);
  if (!response.ok) {
    throw new Error(`knowledge fetch failed (${response.status})`);
  }
  return (await response.json()) as KnowledgeResponse;
}
