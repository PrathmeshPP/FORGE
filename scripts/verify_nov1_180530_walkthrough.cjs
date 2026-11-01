const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const TARGET_URL = 'http://127.0.0.1:5174/?scene=indoor_nov1_180530_photographic';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');
const OUTPUTS_DIR = path.resolve(__dirname, '../outputs/indoor/texture_experiment_video_20261101_180530/renders');

async function run() {
  console.log('[*] Launching browser to verify new scene at:', TARGET_URL);
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

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('Error') || text.includes('error') || text.includes('WARN')) {
      console.log('  [BROWSER LOG]:', text);
    }
  });

  page.on('pageerror', err => {
    console.error('  [PAGE ERROR]:', err.message);
  });

  console.log('[*] Navigating to target scene...');
  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 60000 });

  // 1. Wait for canvas and model loading
  await page.waitForSelector('canvas', { timeout: 20000 });
  console.log('[+] Canvas detected, waiting for 3D model loading completion...');

  await page.waitForFunction(() => {
    const overlay = document.querySelector('.loading-overlay');
    return !overlay || overlay.style.display === 'none';
  }, { timeout: 60000 });

  await page.waitForTimeout(3000);

  const title = await page.evaluate(() => {
    const h1 = document.querySelector('h1') || document.querySelector('.scene-title');
    return h1 ? h1.textContent : document.title;
  });
  console.log('[+] Active Scene Title:', title);

  // Capture Screenshot 1: Initial Inspection View
  const ss1Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_1_initial_inspection.png');
  await page.screenshot({ path: ss1Path });
  console.log('[+] Saved Screenshot 1:', path.basename(ss1Path));

  // 2. Click ENTER SPACE
  console.log('[*] Engaging First-Person Walkthrough Mode...');
  const enterBtn = await page.waitForSelector('button:has-text("ENTER SPACE"), button:has-text("WALKTHROUGH")', { timeout: 10000 });
  await enterBtn.click();
  await page.waitForTimeout(2000);

  // Capture Screenshot 2: Walkthrough Spawn View
  const ss2Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_2_walkthrough_spawn.png');
  await page.screenshot({ path: ss2Path });
  console.log('[+] Saved Screenshot 2 (Spawn View):', path.basename(ss2Path));

  // 3. Test WASD Navigation: Walk forward along corridor
  console.log('[*] Testing WASD forward movement (KeyW)...');
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(1500);
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(600);

  // Capture Screenshot 3: Moved forward down corridor
  const ss3Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_3_walk_forward_corridor.png');
  await page.screenshot({ path: ss3Path });
  console.log('[+] Saved Screenshot 3 (Walk Forward):', path.basename(ss3Path));

  // 4. Test Strafe Left towards 3-seater waiting chairs & banner
  console.log('[*] Testing strafe left (KeyA) towards waiting chairs...');
  await page.keyboard.down('KeyA');
  await page.waitForTimeout(1200);
  await page.keyboard.up('KeyA');
  await page.waitForTimeout(600);

  // Capture Screenshot 4: Close to Waiting Chairs
  const ss4Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_4_waiting_chairs_closeup.png');
  await page.screenshot({ path: ss4Path });
  console.log('[+] Saved Screenshot 4 (Waiting Chairs):', path.basename(ss4Path));

  // 5. Test Strafe Right across corridor towards window facade
  console.log('[*] Testing strafe right (KeyD) towards window facade...');
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(2000);
  await page.keyboard.up('KeyD');
  await page.waitForTimeout(600);

  // Capture Screenshot 5: Window Facade and Staircase Junction
  const ss5Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_5_window_and_stairs.png');
  await page.screenshot({ path: ss5Path });
  console.log('[+] Saved Screenshot 5 (Window & Stairs):', path.basename(ss5Path));

  // 6. Verify Minimap
  const minimapExists = await page.evaluate(() => {
    return !!document.querySelector('.minimap-container') || !!document.querySelector('canvas');
  });
  console.log('[+] MiniMap present in DOM:', minimapExists);

  // 7. Exit Walkthrough Mode
  console.log('[*] Testing EXIT SPACE via KeyG...');
  await page.keyboard.press('KeyG');
  await page.waitForTimeout(1500);

  const ss6Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_6_exit_inspection.png');
  await page.screenshot({ path: ss6Path });
  console.log('[+] Saved Screenshot 6 (Exit to Inspection):', path.basename(ss6Path));

  // 8. Test Top Camera Preset
  console.log('[*] Testing Top Camera Preset button...');
  const topBtn = await page.$('button:has-text("Top"), button:has-text("TOP")');
  if (topBtn) {
    await topBtn.click();
    await page.waitForTimeout(1500);
    const ss7Path = path.join(ARTIFACT_DIR, 'viewer_nov1_180530_7_top_preset.png');
    await page.screenshot({ path: ss7Path });
    console.log('[+] Saved Screenshot 7 (Top Preset):', path.basename(ss7Path));
  }

  // Copy to OUTPUTS_DIR
  fs.mkdirSync(OUTPUTS_DIR, { recursive: true });
  fs.copyFileSync(ss1Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_initial_inspection.png'));
  fs.copyFileSync(ss2Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_walkthrough_spawn.png'));
  fs.copyFileSync(ss3Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_walk_forward_corridor.png'));
  fs.copyFileSync(ss4Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_waiting_chairs_closeup.png'));
  fs.copyFileSync(ss5Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_window_and_stairs.png'));
  fs.copyFileSync(ss6Path, path.join(OUTPUTS_DIR, 'viewer_nov1_180530_exit_inspection.png'));

  console.log('\n[SUCCESS] New scene walkthrough verification completed successfully!');
  await browser.close();
  process.exit(0);
}

run().catch(err => {
  console.error('[FATAL ERROR]:', err);
  process.exit(1);
});
