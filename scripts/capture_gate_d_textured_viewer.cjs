const path = require('path');
const fs = require('fs');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8080;
const BASE_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(BASE_DIR, 'outputs/indoor/distortion_investigation_20261031/gate_d_full_duration/textured_tsdf_model');
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

async function captureTexturedViewer() {
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

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('error') || text.includes('Error') || text.includes('loaded') || text.includes('GLB')) {
      console.log('    [VIEWER ' + msg.type() + ']:', text);
    }
  });

  try {
    console.log(`\n[*] Navigating to production viewer: http://127.0.0.1:${PORT}`);
    await page.goto(`http://127.0.0.1:${PORT}`, { waitUntil: 'networkidle', timeout: 35000 });
    // Allow GLB to fully load into WebGL
    await page.waitForTimeout(6000);

    // 1. Full-Room Overview of the Textured Model
    const shotOverview = path.join(OUT_DIR, 'viewer_gate_d_textured_overview.png');
    await page.screenshot({ path: shotOverview });
    console.log(`[+] 1. Captured Textured Model Overview: ${path.basename(shotOverview)}`);
    if (fs.existsSync(ARTIFACT_DIR)) {
      fs.copyFileSync(shotOverview, path.join(ARTIFACT_DIR, 'viewer_gate_d_textured_overview.png'));
    }

    // 2. Top-down Floorplan
    const topBtn = page.locator('button[title*="Top Overhead View"], button:has-text("Top")').first();
    if (await topBtn.count() > 0) {
      await topBtn.click();
      await page.waitForTimeout(2000);
      const shotTop = path.join(OUT_DIR, 'viewer_gate_d_textured_top_plan.png');
      await page.screenshot({ path: shotTop });
      console.log(`[+] 2. Captured Textured Model Top Plan: ${path.basename(shotTop)}`);
      if (fs.existsSync(ARTIFACT_DIR)) {
        fs.copyFileSync(shotTop, path.join(ARTIFACT_DIR, 'viewer_gate_d_textured_top_plan.png'));
      }
    }

    // 3. Furniture Close-Up (Orbit & Zoom)
    console.log('[*] Orbiting and zooming into furniture...');
    await page.mouse.move(840, 500);
    await page.mouse.down({ button: 'left' });
    await page.mouse.move(960, 420, { steps: 15 });
    await page.mouse.up({ button: 'left' });
    await page.mouse.wheel(0, -600); // Zoom in
    await page.waitForTimeout(2000);
    const shotClose = path.join(OUT_DIR, 'viewer_gate_d_textured_furniture_close.png');
    await page.screenshot({ path: shotClose });
    console.log(`[+] 3. Captured Furniture Close-up: ${path.basename(shotClose)}`);
    if (fs.existsSync(ARTIFACT_DIR)) {
      fs.copyFileSync(shotClose, path.join(ARTIFACT_DIR, 'viewer_gate_d_textured_furniture_close.png'));
    }

    // 4. Toggle Wireframe
    const wireBtn = page.locator('button[title*="Toggle wireframe"], button:has-text("Wireframe")').first();
    if (await wireBtn.count() > 0) {
      await wireBtn.click();
      await page.waitForTimeout(1500);
      const shotWire = path.join(OUT_DIR, 'viewer_gate_d_textured_wireframe.png');
      await page.screenshot({ path: shotWire });
      console.log(`[+] 4. Captured Wireframe Inspection: ${path.basename(shotWire)}`);
      if (fs.existsSync(ARTIFACT_DIR)) {
        fs.copyFileSync(shotWire, path.join(ARTIFACT_DIR, 'viewer_gate_d_textured_wireframe.png'));
      }
    }

    console.log('\n[+] All Three.js viewer verification renders successfully captured!');
  } catch (err) {
    console.error('[-] Error during viewer capture:', err);
  } finally {
    await browser.close();
  }
}

captureTexturedViewer();

