export type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type InferenceLogEntry = {
  model: string;
  max_tokens: number | null;
  message_count: number;
  finish_reason: string | null;
  usage: TokenUsage | null;
  duration_ms: number;
  request_messages: { role: string; content: string }[];
  response_content: string;
};

export type InferenceLogsResponse = {
  chat_session_id: string;
  count: number;
  entries: InferenceLogEntry[];
};

export async function fetchInferenceLogs(
  chatSessionId: string,
  tail = 50
): Promise<InferenceLogsResponse> {
  const response = await fetch(`/v1/logs/${chatSessionId}?tail=${tail}`);
  if (!response.ok) {
    throw new Error(`logs fetch failed (${response.status})`);
  }
  return (await response.json()) as InferenceLogsResponse;
}
