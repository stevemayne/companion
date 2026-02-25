export type DebugTrace = {
  trace_id: string;
  created_at: string;
  idempotency_replay: boolean;
  seed_version: number | null;
  safety_transforms: string[];
  user_message: string;
  assistant_message: string;
  turn_trace: Record<string, unknown>;
};

export type DebugTraceResponse = {
  chat_session_id: string;
  count: number;
  traces: DebugTrace[];
};
