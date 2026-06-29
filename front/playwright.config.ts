import { defineConfig, devices } from '@playwright/test';

const LIVE_LLM = process.env.E2E_LIVE_LLM === 'true';
const RECORD_MODE = process.env.E2E_RECORD_MODE === 'true';

export default defineConfig({
  testDir: './e2e',
  timeout: LIVE_LLM ? 120000 : 30000,
  retries: LIVE_LLM ? 2 : 1,
  expect: {
    timeout: LIVE_LLM ? 30000 : 10000,
  },
  use: {
    baseURL: 'http://localhost:3000',
    headless: true,
    screenshot: LIVE_LLM ? 'on' : 'only-on-failure',
    trace: 'on-first-retry',
    video: LIVE_LLM ? 'on' : 'off',
  },
  grep: process.env.E2E_LIVE_ONLY ? /@live/ : undefined,
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: 'cd ../back && uv run uvicorn src.main:app --host 0.0.0.0 --port 8000',
      port: 8000,
      reuseExistingServer: !RECORD_MODE,
      timeout: 30000,
      env: {
        E2E_TEST_MODE: LIVE_LLM ? 'false' : 'true',
        E2E_LIVE_LLM: LIVE_LLM ? 'true' : 'false',
        E2E_RECORD_MODE: RECORD_MODE ? 'true' : 'false',
      },
    },
    {
      command: 'npx next dev --port 3000',
      port: 3000,
      reuseExistingServer: !RECORD_MODE,
      timeout: 60000,
    },
  ],
});
