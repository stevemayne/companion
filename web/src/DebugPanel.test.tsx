import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { afterEach } from "vitest";

import { DebugPanel } from "./DebugPanel";
import { DebugTrace } from "./debugTypes";

const trace: DebugTrace = {
  trace_id: "t-1",
  created_at: "now",
  idempotency_replay: false,
  seed_version: 1,
  safety_transforms: ["pii_redaction"],
  user_message: "u",
  assistant_message: "a",
  turn_trace: {
    prompt: { summary: "summary prompt", raw: "raw prompt" },
    preprocess: { intent: "statement", emotion: "neutral", entities: [] },
    retrieval: { semantic_items: [], graph_relations: [] },
    writes: { semantic_upserts: ["fact"] }
  }
};

afterEach(() => {
  cleanup();
});

describe("DebugPanel", () => {
  it("renders trace summary by default", () => {
    render(
      <DebugPanel
        isOpen={true}
        onToggleOpen={vi.fn()}
        onRefresh={vi.fn()}
        traces={[trace]}
        status="loaded"
        showRawPrompt={false}
        setShowRawPrompt={vi.fn()}
        verbose={false}
        setVerbose={vi.fn()}
      />
    );

    expect(screen.getByText(/debug status: loaded/i)).toBeTruthy();
    expect(screen.getByText(/summary prompt/i)).toBeTruthy();
  });

  it("toggles raw prompt and verbose flags", () => {
    const setRaw = vi.fn();
    const setVerbose = vi.fn();

    render(
      <DebugPanel
        isOpen={true}
        onToggleOpen={vi.fn()}
        onRefresh={vi.fn()}
        traces={[trace]}
        status="loaded"
        showRawPrompt={true}
        setShowRawPrompt={setRaw}
        verbose={true}
        setVerbose={setVerbose}
      />
    );

    expect(screen.getByText(/prompt:\s*raw prompt/i)).toBeTruthy();
    expect(screen.getByText(/writes:/i)).toBeTruthy();

    const checkboxes = screen.getAllByRole("checkbox");
    fireEvent.click(checkboxes[0]);
    fireEvent.click(checkboxes[1]);

    expect(setRaw).toHaveBeenCalled();
    expect(setVerbose).toHaveBeenCalled();
  });
});
