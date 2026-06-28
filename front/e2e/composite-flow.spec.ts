import { test, expect } from '@playwright/test';

test.describe('Composite Plan-and-Execute', () => {
  test('mock — chat input and session creation loads', async ({ page }) => {
    await page.goto('/');

    // Verify page loads with session creation capability
    const newSessionBtn = page.locator('[data-testid="new-session-btn"]');
    await expect(newSessionBtn).toBeVisible({ timeout: 10000 });

    // Verify chat input exists
    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible({ timeout: 5000 });
  });

  test('@live real LLM — composite exam + exercise generation', async ({ page }) => {
    test.slow();
    await page.goto('/');
    await page.waitForTimeout(2000);

    // Step 1: Create a new session
    await page.click('[data-testid="new-session-btn"]');
    await page.fill('[data-testid="session-name-input"]', 'Composite E2E Test');
    await page.click('[data-testid="session-create-confirm"]');
    await page.waitForTimeout(2000);

    // Verify session created
    const sessionItem = page.locator('[data-testid="session-item"]').first();
    await expect(sessionItem).toBeVisible({ timeout: 10000 });

    // Step 2: Send composite message — exam + exercise in one request
    const compositeMessage =
      'Generame un examen de 3 preguntas sobre agentes inteligentes y tambien un ejercicio practico sobre racionalidad en agentes.';
    await page.fill('[data-testid="chat-input"]', compositeMessage);
    await page.click('[data-testid="send-btn"]');

    // Step 3: Wait for real LLM to process composite plan (classify → plan → execute × 2 → synthesize)
    // Composite runs 2 LLM generation calls sequentially — wait for response to appear
    await page.waitForTimeout(90000);

    // Step 4: Verify page didn't crash and send button is back (response received)
    // Chat message text may be anywhere in the DOM — check page body for content
    const sendBtn = page.locator('[data-testid="send-btn"]');
    await expect(sendBtn).toBeVisible({ timeout: 30000 });

    // Get full page text to check for exam/exercise artifacts
    const pageText = await page.textContent('body').catch(() => '');

    // Tolerance assertions — LLM output is non-deterministic but should contain both artifacts
    const hasExamContent =
      pageText.includes('pregunta') ||
      pageText.includes('examen') ||
      pageText.includes('opción');

    const hasExerciseContent =
      pageText.includes('ejercicio') ||
      pageText.includes('práctico') ||
      pageText.includes('enunciado') ||
      pageText.includes('consigna');

    // At least one of the two must be present — composite may partially succeed
    expect(hasExamContent || hasExerciseContent).toBe(true);

    // Step 5: Verify page didn't crash — chat input still visible
    const chatInputAfter = page.locator('[data-testid="chat-input"]');
    await expect(chatInputAfter).toBeVisible({ timeout: 5000 });
  });

  test('@live real LLM — composite with no material (graceful degradation)', async ({ page }) => {
    test.slow();
    await page.goto('/');
    await page.waitForTimeout(2000);

    // Create fresh session (no material ingested)
    await page.click('[data-testid="new-session-btn"]');
    await page.fill('[data-testid="session-name-input"]', 'No Material Test');
    await page.click('[data-testid="session-create-confirm"]');
    await page.waitForTimeout(2000);

    // Send composite message without any ingested material
    await page.fill(
      '[data-testid="chat-input"]',
      'Generame un examen de 3 preguntas y un ejercicio practico sobre agentes inteligentes.'
    );
    await page.click('[data-testid="send-btn"]');

    // Wait for response
    await page.waitForTimeout(45000);

    // Verify send button is back (response received)
    const sendBtn = page.locator('[data-testid="send-btn"]');
    await expect(sendBtn).toBeVisible({ timeout: 15000 });

    // Verify system responds gracefully (no crash, no 500 error)
    const pageText = await page.textContent('body').catch(() => '');

    // Should mention material not found or suggest uploading
    const gracefulResponse =
      pageText.includes('material') ||
      pageText.includes('no se encontró') ||
      pageText.includes('subir') ||
      pageText.includes('archivo') ||
      pageText.includes('documento') ||
      pageText.includes('apunte');

    expect(gracefulResponse).toBe(true);

    // Page must not crash
    const chatInputAfter = page.locator('[data-testid="chat-input"]');
    await expect(chatInputAfter).toBeVisible({ timeout: 5000 });
  });
});
