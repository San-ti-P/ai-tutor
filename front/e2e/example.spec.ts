import { test, expect } from '@playwright/test';

test('app loads', async ({ page }) => {
  await page.goto('/');
  // Verify the app renders (minimal sanity check)
  await expect(page).toHaveTitle(/Tutor/);
});
