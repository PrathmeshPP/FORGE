const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const TARGET_URL = 'http://127.0.0.1:5174/?scene=indoor_my_movie_4_adaptive';
const OUTPUTS_DIR = path.resolve(__dirname, '../outputs/indoor/dataset_my_movie_4_adaptive/walkthrough_renders');
const ARTIFACTS_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

if (!fs.existsSync(OUTPUTS_DIR)) {
  fs.mkdirSync(OUTPUTS_DIR, { recursive: true });
}

async function run() {
  console.log('[*] Capturing walkthrough verification views for My Movie 4 Adaptive Digital Twin...');
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: [
      '--enable-webgl',
      '--use-gl=angle',
      '--use-angle=default',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--window-size=1600,1000'
    ]
  });

  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForSelector('canvas', { timeout: 20000 });

  await page.waitForFunction(() => {
    const overlay = document.querySelector('.loading-overlay');
    return !overlay || overlay.style.display === 'none';
  }, { timeout: 60000 });

  await page.waitForTimeout(2500);

  // View 1: Inspection Overview
  const p1 = path.join(OUTPUTS_DIR, '01_initial_inspection.png');
  await page.screenshot({ path: p1 });
  console.log('[+] View 1 captured: Initial Inspection Overview');

  // View 2: Enter Walkthrough
  const enterBtn = page.locator('button:has-text("Enter Walkthrough")');
  if (await enterBtn.isVisible()) {
    await enterBtn.click();
    await page.waitForTimeout(1000);
  }
  const p2 = path.join(OUTPUTS_DIR, '02_walkthrough_spawn.png');
  await page.screenshot({ path: p2 });
  console.log('[+] View 2 captured: Walkthrough Spawn');

  // View 3: Walk Forward into Room
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(1200);
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(600);
  const p3 = path.join(OUTPUTS_DIR, '03_walk_forward.png');
  await page.screenshot({ path: p3 });
  console.log('[+] View 3 captured: Walk Forward into Room');

  // View 4: Look Left towards Furniture & Wardrobes
  await page.keyboard.down('KeyA');
  await page.waitForTimeout(800);
  await page.keyboard.up('KeyA');
  await page.waitForTimeout(600);
  const p4 = path.join(OUTPUTS_DIR, '04_wardrobes_and_bed.png');
  await page.screenshot({ path: p4 });
  console.log('[+] View 4 captured: Wardrobes and Bed Detail');

  // View 5: Look Right towards Cooler & Desk
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(1400);
  await page.keyboard.up('KeyD');
  await page.waitForTimeout(600);
  const p5 = path.join(OUTPUTS_DIR, '05_cooler_and_desk.png');
  await page.screenshot({ path: p5 });
  console.log('[+] View 5 captured: Air Cooler and Study Area');

  // View 6: Exit and Switch to Performance Preset
  const exitBtn = page.locator('button:has-text("Exit Walkthrough")');
  if (await exitBtn.isVisible()) {
    await exitBtn.click();
    await page.waitForTimeout(800);
  }

  // View 7: Top-down Preset
  const topBtn = page.locator('button:has-text("Top")');
  if (await topBtn.isVisible()) {
    await topBtn.click();
    await page.waitForTimeout(1000);
  }
  const p6 = path.join(OUTPUTS_DIR, '06_top_down_view.png');
  await page.screenshot({ path: p6 });
  console.log('[+] View 6 captured: Top-down Floor Plan View');

  await browser.close();

  // Copy to brain artifacts
  for (const f of fs.readdirSync(OUTPUTS_DIR)) {
    if (f.endsWith('.png')) {
      fs.copyFileSync(path.join(OUTPUTS_DIR, f), path.join(ARTIFACTS_DIR, `my_movie_4_adaptive_${f}`));
    }
  }
  console.log('[+] All walkthrough verification renders copied to brain artifacts directory.');
}

run().catch(err => {
  console.error('[!] Capture failed:', err);
  process.exit(1);
});
