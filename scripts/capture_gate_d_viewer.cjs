const path = require('path');
const fs = require('fs');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8080;
const BASE_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(BASE_DIR, 'outputs/indoor/distortion_investigation_20261031/gate_d_full_duration');
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

async function captureGateDViewer() {
  console.log('[*] Launching Headless Edge browser with WebGL...');
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: [
      '--enable-webgl',
      '--use-gl=angle',
      '--use-angle=default',
      '--ignore-gpu-blocklist',
      '--no-sandbox'
    ]
  });

  const page = await browser.newPage({
    viewport: { width: 1680, height: 980 }
  });

  try {
    console.log(`\n[*] Navigating to production viewer: http://127.0.0.1:${PORT}`);
    await page.goto(`http://127.0.0.1:${PORT}`, { waitUntil: 'networkidle', timeout: 35000 });
    await page.waitForTimeout(5000);

    const shotPcd = path.join(OUT_DIR, 'viewer_gate_d_pre_tsdf_point_cloud.png');
    await page.screenshot({ path: shotPcd });
    console.log(`[+] Captured Gate D Pre-TSDF Point Cloud in Viewer: ${path.basename(shotPcd)}`);

    if (fs.existsSync(ARTIFACT_DIR)) {
      fs.copyFileSync(shotPcd, path.join(ARTIFACT_DIR, 'viewer_gate_d_pre_tsdf_point_cloud.png'));
      console.log(`[+] Copied to artifacts directory.`);
    }

    // Also top-down plan view
    const topBtn = page.locator('button[title*="Top Overhead View"], button:has-text("Top")').first();
    if (await topBtn.count() > 0) {
      await topBtn.click();
      await page.waitForTimeout(1500);
      const shotTop = path.join(OUT_DIR, 'viewer_gate_d_pre_tsdf_top_plan.png');
      await page.screenshot({ path: shotTop });
      console.log(`[+] Captured Gate D Pre-TSDF Top Plan in Viewer: ${path.basename(shotTop)}`);
      if (fs.existsSync(ARTIFACT_DIR)) {
        fs.copyFileSync(shotTop, path.join(ARTIFACT_DIR, 'viewer_gate_d_pre_tsdf_top_plan.png'));
      }
    }
  } catch (err) {
    console.error('[-] Error during viewer capture:', err);
  } finally {
    await browser.close();
  }
}

captureGateDViewer();

