import { test, expect } from "@playwright/test";

test("navigate to inference logs overlay", async ({ page, request }) => {
  await page.goto("/");

  // Wait for the app to render
  await expect(page.locator(".topbar-name")).toBeVisible();

  // Open sidebar
  await page.click(".topbar-menu");
  await expect(page.locator(".drawer")).toBeVisible();

  // Switch to debug tab
  await page.click('.drawer-tab:has-text("Debug")');
  await expect(page.locator('.drawer-tab--active:has-text("Debug")')).toBeVisible();

  // Click inference logs button
  await page.click('button:has-text("Inference Logs")');

  // Verify the full-screen overlay appears
  const overlay = page.locator(".log-overlay");
  await expect(overlay).toBeVisible();
  await expect(overlay.locator(".log-header")).toBeVisible();
  await expect(overlay.locator("h2:has-text('Inference Logs')")).toBeVisible();

  // Verify header controls are present
  await expect(overlay.locator('button:has-text("Refresh")')).toBeVisible();
  await expect(overlay.locator('button:has-text("Close")')).toBeVisible();

  // Session ID should be displayed
  await expect(overlay.locator(".log-session-id")).toBeVisible();

  // Grab the session ID for cleanup
  const displayedSessionId = await overlay.locator(".log-session-id").textContent();

  // Take a screenshot for visual verification
  await page.screenshot({ path: "e2e/screenshots/inference-logs.png", fullPage: true });

  // Close the overlay
  await overlay.locator('button:has-text("Close")').click();
  await expect(overlay).not.toBeVisible();

  // Clean up: delete the test session via API
  if (displayedSessionId?.trim()) {
    const resp = await request.delete(`/v1/sessions/${displayedSessionId.trim()}`);
    expect(resp.status()).toBe(204);
  }
});
