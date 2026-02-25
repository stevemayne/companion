import { afterEach, describe, expect, it, vi } from "vitest";

import { createSeed, upsertSeed, updateSeed } from "./seedApi";

const payload = {
  seed: {
    companion_name: "Ari",
    backstory: "Backstory",
    character_traits: ["warm"],
    goals: ["trust"],
    relationship_setup: "Companion"
  },
  notes: "note"
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("seedApi", () => {
  it("sends POST create seed request", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(null, { status: 201 }));

    const response = await createSeed("abc", payload);

    expect(response.status).toBe(201);
    expect(fetchMock).toHaveBeenCalledWith(
      "/v1/sessions/abc/seed",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("sends PUT update seed request", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(null, { status: 200 }));

    const response = await updateSeed("xyz", payload);

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledWith(
      "/v1/sessions/xyz/seed",
      expect.objectContaining({ method: "PUT" })
    );
  });

  it("falls back to PUT when POST returns 409", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(null, { status: 409 }))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));

    const response = await upsertSeed("seed-1", payload);

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/v1/sessions/seed-1/seed",
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/v1/sessions/seed-1/seed",
      expect.objectContaining({ method: "PUT" })
    );
  });
});
