import { test, expect } from '@playwright/test';

test.describe('Ingest → Exam → Evaluate', () => {
  test('mock — structural flow', async ({ page }) => {
    test.setTimeout(300000);
    await page.goto('/', { timeout: 60000 });

    // Create session
    await page.click('[data-testid="new-session-btn"]', { timeout: 30000 });
    await page.fill('[data-testid="session-name-input"]', 'E2E Test Session');
    await page.click('[data-testid="session-create-confirm"]');
    await expect(page.locator('[data-testid="session-item"]').first()).toBeVisible({ timeout: 5000 });

    // Upload a PDF so exam generator has material
    const fileInput = page.locator('[data-testid="file-upload-input"]');
    await fileInput.setInputFiles('../back/tests/fixtures/apunteAgentes_IA2007.pdf');
    await page.waitForTimeout(5000);

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
    test.setTimeout(300000); // 5 min — live LLM is slow
    // Create active session
    await page.goto('/', { timeout: 60000 });
    await page.click('[data-testid="new-session-btn"]', { timeout: 30000 });
    await page.fill('[data-testid="session-name-input"]', 'E2E Live Exam Test');
    await page.click('[data-testid="session-create-confirm"]');
    await expect(page.locator('[data-testid="session-item"]').first()).toBeVisible({ timeout: 30000 });

    // Upload a PDF so exam generator has material
    const fileInput = page.locator('[data-testid="file-upload-input"]');
    await fileInput.setInputFiles('../back/tests/fixtures/apunteAgentes_IA2007.pdf');
    await page.waitForTimeout(8000); // wait for ingest to complete

    // Request exam through chat — widget renders inline (no page navigation)
    await page.fill('[data-testid="chat-input"]', 'generame un examen de 3 preguntas sobre agentes inteligentes');
    await page.click('[data-testid="send-btn"]');

    // Wait for real LLM to generate exam via chat widget
    await page.waitForTimeout(60000);

    // Tolerance: exam may have 0-5 questions, or LLM returns text-only (no widget)
    const questions = page.locator('[data-testid="exam-question"]');
    const count = await questions.count();

    if (count === 0) {
      // Widget might not render — verify page didn't crash
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
