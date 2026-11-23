const path = require('path');
const fs = require('fs');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8080;
const BASE_DIR = path.resolve(__dirname, '..');
const OUT_DIR = path.join(BASE_DIR, 'outputs/indoor/distortion_investigation_20261031/gate_c_extended');

async function captureGateCViewer() {
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
    await page.waitForTimeout(4000);

    // 1. CANDIDATE A — CORRECTED SIM(3) (Already active by default as Card 0)
    console.log('\n[1/3] Capturing CANDIDATE A — CORRECTED SIM(3)...');
    const shotAOverview = path.join(OUT_DIR, 'viewer_candidate_a_overview.png');
    await page.screenshot({ path: shotAOverview });
    console.log(`  [+] Candidate A Overview saved: ${path.basename(shotAOverview)}`);

    // Top view
    const topBtn = page.locator('button[title*="Top Overhead View"], button:has-text("Top")').first();
    if (await topBtn.count() > 0) {
      await topBtn.click();
      await page.waitForTimeout(1500);
      const shotATop = path.join(OUT_DIR, 'viewer_candidate_a_top_plan.png');
      await page.screenshot({ path: shotATop });
      console.log(`  [+] Candidate A Top Plan saved: ${path.basename(shotATop)}`);
    }

    // Reset to Iso/Front view
    const isoBtn = page.locator('button:has-text("Iso")').first();
    if (await isoBtn.count() > 0) {
      await isoBtn.click();
      await page.waitForTimeout(1500);
      const shotAIso = path.join(OUT_DIR, 'viewer_candidate_a_isometric.png');
      await page.screenshot({ path: shotAIso });
      console.log(`  [+] Candidate A Isometric saved: ${path.basename(shotAIso)}`);
    }

    // Switch to Point Cloud representation for Candidate A
    const plyBtn = page.locator('button:has-text("Photo PLY")').first();
    if (await plyBtn.count() > 0) {
      await plyBtn.click();
      await page.waitForTimeout(3000);
      const shotAPly = path.join(OUT_DIR, 'viewer_candidate_a_point_cloud.png');
      await page.screenshot({ path: shotAPly });
      console.log(`  [+] Candidate A TSDF Point Cloud saved: ${path.basename(shotAPly)}`);
    }

    // 2. BASELINE — DEFECTIVE (Card 1)
    console.log('\n[2/3] Capturing BASELINE — DEFECTIVE...');
    await page.locator('.scene-card').nth(1).click();
    await page.waitForTimeout(4000);
    const shotBaseOverview = path.join(OUT_DIR, 'viewer_baseline_defective_overview.png');
    await page.screenshot({ path: shotBaseOverview });
    console.log(`  [+] Baseline Overview saved: ${path.basename(shotBaseOverview)}`);

    if (await topBtn.count() > 0) {
      await topBtn.click();
      await page.waitForTimeout(1500);
      const shotBaseTop = path.join(OUT_DIR, 'viewer_baseline_defective_top_plan.png');
      await page.screenshot({ path: shotBaseTop });
      console.log(`  [+] Baseline Top Plan saved: ${path.basename(shotBaseTop)}`);
    }

    // 3. CANDIDATE B — RIGID SE(3) (Card 2)
    console.log('\n[3/3] Capturing CANDIDATE B — RIGID SE(3)...');
    await page.locator('.scene-card').nth(2).click();
    await page.waitForTimeout(4000);
    const shotBOverview = path.join(OUT_DIR, 'viewer_candidate_b_overview.png');
    await page.screenshot({ path: shotBOverview });
    console.log(`  [+] Candidate B Overview saved: ${path.basename(shotBOverview)}`);

    if (await topBtn.count() > 0) {
      await topBtn.click();
      await page.waitForTimeout(1500);
      const shotBTop = path.join(OUT_DIR, 'viewer_candidate_b_top_plan.png');
      await page.screenshot({ path: shotBTop });
      console.log(`  [+] Candidate B Top Plan saved: ${path.basename(shotBTop)}`);
    }

    console.log('\n[+] All 3 models captured in Three.js viewer!');
  } catch (err) {
    console.error('[-] Error during viewer capture:', err);
  } finally {
    await browser.close();
  }
}

captureGateCViewer();
