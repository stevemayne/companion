import { SeedPayload } from "./seedApi";

type MemoryMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

type SeedContextResponse = {
  seed: SeedPayload;
  user_description: string | null;
  notes: string | null;
};

type MemoryResponse = {
  messages: MemoryMessage[];
  seed_context: SeedContextResponse | null;
};

export async function fetchMemory(chatSessionId: string): Promise<MemoryResponse> {
  const response = await fetch(`/v1/memory/${chatSessionId}`);
  if (!response.ok) {
    throw new Error(`memory fetch failed (${response.status})`);
  }
  return (await response.json()) as MemoryResponse;
}
