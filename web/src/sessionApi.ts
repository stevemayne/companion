export type SessionSummary = {
  chat_session_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  companion_name: string | null;
};

type SessionListResponse = {
  sessions: SessionSummary[];
};

export async function fetchSessions(limit = 50): Promise<SessionSummary[]> {
  const response = await fetch(`/v1/sessions?limit=${encodeURIComponent(String(limit))}`);
  if (!response.ok) {
    throw new Error(`session list failed (${response.status})`);
  }
  const payload = (await response.json()) as SessionListResponse;
  return payload.sessions;
}
