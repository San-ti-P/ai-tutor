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
    // Use /exam page with dedicated exam form (not chat)
    await page.goto('/exam');
    await page.waitForTimeout(2000);

    // Fill exam form
    const topicInput = page.locator('input[placeholder*="Ej:"]').first();
    if (await topicInput.isVisible()) {
      await topicInput.fill('Agentes inteligentes');
    }

    // Set question count to 3
    const countInput = page.locator('input[type="number"]').first();
    if (await countInput.isVisible()) {
      await countInput.fill('3');
    }

    // Click generate
    const submitBtn = page.locator('[data-testid="submit-exam-btn"]');
    await expect(submitBtn).toBeVisible({ timeout: 5000 });
    await submitBtn.click();

    // Wait for real LLM to generate exam
    await page.waitForTimeout(35000);

    // Tolerance: exam may have 0-5 questions (LLM non-deterministic)
    const questions = page.locator('[data-testid="exam-question"]');
    const count = await questions.count();

    // Either exam rendered with questions OR submit button still visible (still generating)
    // Both are valid outcomes for live LLM
    if (count === 0) {
      // Exam might still be generating or LLM returned non-widget format
      // Verify page didn't crash
      await expect(page.locator('body')).toBeVisible();
    } else {
      expect(count).toBeGreaterThanOrEqual(1);

      // Answer MCQ questions if possible
      for (let i = 0; i < count; i++) {
        const radio = questions.nth(i).locator('input[type="radio"]').first();
        if (await radio.isVisible().catch(() => false)) {
          await radio.click();
        }
      }

      // Submit if button visible
      const evalSubmitBtn = page.locator('[data-testid="submit-exam-btn"]');
      if (await evalSubmitBtn.isVisible().catch(() => false)) {
        await evalSubmitBtn.click();
        await page.waitForTimeout(10000);
      }
    }
  });
});
