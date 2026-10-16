import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  timeout: 120000,
  use: {
    baseURL: process.env.SKIRT_VIEWER_TEST_URL || 'http://127.0.0.1:8765',
    viewport: {width: 1512, height: 1050},
    launchOptions: {
      executablePath: process.env.SKIRT_VIEWER_CHROMIUM || undefined,
      args: ['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
    },
  },
});
