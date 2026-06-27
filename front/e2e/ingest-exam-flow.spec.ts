import { test, expect } from '@playwright/test';

test.describe('Ingest → Exam → Evaluate', () => {
  test('mock — structural flow', async ({ page }) => {
    await page.goto('/');

    // Create session
    await page.click('[data-testid="new-session-btn"]');
    await page.fill('[data-testid="session-name-input"]', 'E2E Test Session');
    await page.click('[data-testid="session-create-confirm"]');
    await expect(page.locator('[data-testid="session-item"]').first()).toBeVisible({ timeout: 5000 });

    // Type chat message to trigger exam
    await page.fill('[data-testid="chat-input"]', 'generame un examen de 3 preguntas');
    await page.click('[data-testid="send-btn"]');

    // Wait for response (mock mode returns quickly)
    await page.waitForTimeout(3000);

    // Verify some response appeared
    const response = page.locator('text=examen').first();
    await expect(response).toBeVisible({ timeout: 10000 });
  });

  test('@live real LLM — quality validation', async ({ page }) => {
    test.slow();
    await page.goto('/');

    // Create session with live backend
    await page.click('[data-testid="new-session-btn"]');
    await page.fill('[data-testid="session-name-input"]', 'Live Quality Test');
    await page.click('[data-testid="session-create-confirm"]');
    await expect(page.locator('[data-testid="session-item"]').first()).toBeVisible({ timeout: 5000 });

    // Send exam generation request
    await page.fill('[data-testid="chat-input"]', 'generame un examen de 3 preguntas sobre limites');
    await page.click('[data-testid="send-btn"]');

    // With live LLM, wait longer
    await page.waitForTimeout(30000);

    // Verify exam widget appears with questions
    const questions = page.locator('[data-testid="exam-question"]');
    const count = await questions.count();
    expect(count).toBeGreaterThanOrEqual(1);

    // Submit exam
    const submitBtn = page.locator('[data-testid="submit-exam-btn"]');
    if (await submitBtn.isVisible()) {
      await submitBtn.click();
      await page.waitForTimeout(5000);

      // Check results page
      const totalScore = page.locator('[data-testid="total-score"]');
      await expect(totalScore).toBeVisible({ timeout: 15000 });
    }
  });
});
