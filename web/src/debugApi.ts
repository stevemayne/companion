import { DebugTraceResponse } from "./debugTypes";

export async function fetchDebugTraces(chatSessionId: string): Promise<DebugTraceResponse> {
  const response = await fetch(`/v1/debug/${chatSessionId}`);
  if (!response.ok) {
    throw new Error(`debug fetch failed (${response.status})`);
  }
  return (await response.json()) as DebugTraceResponse;
}
