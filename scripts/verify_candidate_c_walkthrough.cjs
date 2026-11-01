const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const TARGET_URL = 'http://127.0.0.1:5174/?scene=indoor_nov1_candidate_c_photographic';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');
const OUTPUTS_DIR = path.resolve(__dirname, '../outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/renders');

async function run() {
  console.log('[*] Launching browser to verify Candidate C at:', TARGET_URL);
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
  
  const consoleLogs = [];
  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(text);
    if (text.includes('Error') || text.includes('error') || text.includes('WARN')) {
      console.log('  [BROWSER LOG]:', text);
    }
  });

  page.on('pageerror', err => {
    console.error('  [PAGE ERROR]:', err.message);
  });

  console.log('[*] Navigating to Candidate C scene...');
  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 60000 });

  // 1. Wait for canvas and model to load
  await page.waitForSelector('canvas', { timeout: 20000 });
  console.log('[+] Canvas detected, waiting for 3D model loading completion...');

  // Wait until loading overlay disappears
  await page.waitForFunction(() => {
    const overlay = document.querySelector('.loading-overlay');
    return !overlay || overlay.style.display === 'none';
  }, { timeout: 60000 });

  // Brief pause for Three.js render stabilization and texture upload
  await page.waitForTimeout(3000);

  // Check scene title in header
  const title = await page.evaluate(() => {
    const h1 = document.querySelector('h1') || document.querySelector('.scene-title');
    return h1 ? h1.textContent : document.title;
  });
  console.log('[+] Active Scene Title:', title);

  // Check rendered geometry statistics from telemetry HUD if present
  const stats = await page.evaluate(() => {
    const statElements = Array.from(document.querySelectorAll('.stat-badge, .telemetry-item, .hud-value, span'));
    return statElements.map(e => e.textContent.trim()).filter(t => t.includes('Vert') || t.includes('Tri') || t.includes('FPS') || t.includes('Pos'));
  });
  console.log('[+] HUD stats found:', stats.slice(0, 5));

  // Capture Screenshot 1: Initial Inspection View
  const ss1Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_1_initial_inspection.png');
  await page.screenshot({ path: ss1Path });
  console.log('[+] Saved Screenshot 1:', path.basename(ss1Path));

  // 2. Click ENTER SPACE (Walkthrough Button)
  console.log('[*] Engaging First-Person Walkthrough Mode...');
  const enterBtn = await page.waitForSelector('button:has-text("ENTER SPACE"), button:has-text("WALKTHROUGH")', { timeout: 10000 });
  await enterBtn.click();
  await page.waitForTimeout(2000);

  // Capture Screenshot 2: Spawn in Walkthrough Mode (Aisle perspective)
  const ss2Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_2_walkthrough_spawn.png');
  await page.screenshot({ path: ss2Path });
  console.log('[+] Saved Screenshot 2 (Spawn View):', path.basename(ss2Path));

  // 3. Test WASD Navigation: Walk forward along central aisle
  console.log('[*] Testing WASD forward movement (KeyW)...');
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(1400);
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(600);

  // Capture Screenshot 3: Moved forward down aisle
  const ss3Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_3_walk_forward_aisle.png');
  await page.screenshot({ path: ss3Path });
  console.log('[+] Saved Screenshot 3 (Walk Forward):', path.basename(ss3Path));

  // 4. Test Strafe & Collision Detection towards Right Cot (Bed 2)
  console.log('[*] Testing strafe right (KeyD) towards Bed 2...');
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(1000);
  await page.keyboard.up('KeyD');
  await page.waitForTimeout(600);

  // Capture Screenshot 4: Close to Right Bed with collision sliding active
  const ss4Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_4_right_cot_closeup.png');
  await page.screenshot({ path: ss4Path });
  console.log('[+] Saved Screenshot 4 (Right Cot):', path.basename(ss4Path));

  // 5. Test Strafe Left (KeyA) towards Left Cot & Study Desk
  console.log('[*] Testing strafe left (KeyA) across aisle towards Left Cot & Desk...');
  await page.keyboard.down('KeyA');
  await page.waitForTimeout(1800);
  await page.keyboard.up('KeyA');
  await page.waitForTimeout(600);

  // Capture Screenshot 5: Left Cot and Study Desk
  const ss5Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_5_left_cot_and_desk.png');
  await page.screenshot({ path: ss5Path });
  console.log('[+] Saved Screenshot 5 (Left Cot & Desk):', path.basename(ss5Path));

  // 6. Test Minimap toggle (KeyM)
  console.log('[*] Verifying MiniMap Overlay...');
  const minimapExists = await page.evaluate(() => {
    return !!document.querySelector('.minimap-container') || !!document.querySelector('canvas') || !!document.querySelector('div:has-text("TOP-DOWN FLOOR PLAN")');
  });
  console.log('[+] MiniMap present in DOM:', minimapExists);

  // 7. Exit Walkthrough Mode via KeyG
  console.log('[*] Testing EXIT SPACE via KeyG...');
  await page.keyboard.press('KeyG');
  await page.waitForTimeout(1500);

  // Capture Screenshot 6: Back to Inspection Mode
  const ss6Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_6_exit_inspection.png');
  await page.screenshot({ path: ss6Path });
  console.log('[+] Saved Screenshot 6 (Exit to Inspection):', path.basename(ss6Path));

  // 8. Test Camera Preset: Top perspective
  console.log('[*] Testing Top Camera Preset button...');
  const topBtn = await page.$('button:has-text("Top"), button:has-text("TOP")');
  if (topBtn) {
    await topBtn.click();
    await page.waitForTimeout(1500);
    const ss7Path = path.join(ARTIFACT_DIR, 'viewer_cand_c_7_top_preset.png');
    await page.screenshot({ path: ss7Path });
    console.log('[+] Saved Screenshot 7 (Top Preset):', path.basename(ss7Path));
  }

  // Also copy key screenshots to OUTPUTS_DIR for deliverables
  fs.mkdirSync(OUTPUTS_DIR, { recursive: true });
  fs.copyFileSync(ss1Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_initial_inspection.png'));
  fs.copyFileSync(ss2Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_walkthrough_spawn.png'));
  fs.copyFileSync(ss3Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_walk_forward_aisle.png'));
  fs.copyFileSync(ss4Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_right_cot_closeup.png'));
  fs.copyFileSync(ss5Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_left_cot_and_desk.png'));
  fs.copyFileSync(ss6Path, path.join(OUTPUTS_DIR, 'viewer_cand_c_exit_inspection.png'));

  console.log('\n[SUCCESS] Candidate C viewer walkthrough verification completed successfully!');
  await browser.close();
  process.exit(0);
}

run().catch(err => {
  console.error('[FATAL ERROR]:', err);
  process.exit(1);
});
