export type SeedPayload = {
  companion_name: string;
  backstory: string;
  character_traits: string[];
  goals: string[];
  relationship_setup: string;
};

export type SeedUpsertRequest = {
  seed: SeedPayload;
  user_description?: string;
  notes?: string;
};

async function requestSeed(
  url: string,
  method: "POST" | "PUT",
  payload: SeedUpsertRequest
): Promise<Response> {
  return fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
}

export async function createSeed(
  chatSessionId: string,
  payload: SeedUpsertRequest
): Promise<Response> {
  return requestSeed(`/v1/sessions/${chatSessionId}/seed`, "POST", payload);
}

export async function updateSeed(
  chatSessionId: string,
  payload: SeedUpsertRequest
): Promise<Response> {
  return requestSeed(`/v1/sessions/${chatSessionId}/seed`, "PUT", payload);
}

export async function upsertSeed(
  chatSessionId: string,
  payload: SeedUpsertRequest
): Promise<Response> {
  const createResponse = await createSeed(chatSessionId, payload);
  if (createResponse.status === 409) {
    return updateSeed(chatSessionId, payload);
  }
  return createResponse;
}
