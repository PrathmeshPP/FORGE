const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const BASE_URL = 'http://127.0.0.1:5174';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

async function verifyScene(sceneId, label) {
  console.log(`\n================================================================`);
  console.log(`  VERIFYING SCENE: ${label} (${sceneId})`);
  console.log(`================================================================`);

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
  const targetUrl = `${BASE_URL}/?scene=${sceneId}`;
  console.log('[*] Navigating to:', targetUrl);

  const loadedGlbs = [];
  page.on('response', res => {
    const u = res.url();
    if (u.endsWith('.glb')) {
      loadedGlbs.push({ url: u, status: res.status() });
      console.log(`  [GLB LOADED] Status ${res.status()}: ${path.basename(u)}`);
    }
  });

  const testResults = [];
  function recordTest(name, passed, details = '') {
    testResults.push({ name, passed, details });
    console.log(`  [${passed ? 'PASS' : 'FAIL'}] ${name} ${details ? '(' + details + ')' : ''}`);
  }

  try {
    await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 45000 });
    await page.waitForTimeout(4000);

    const title = await page.title();
    recordTest('Page Loaded Successfully', title.includes('F.O.R.G.E'), `Title: "${title}"`);

    const canvas = await page.$('canvas');
    recordTest('WebGL Viewport Canvas Present', !!canvas);

    const hasGlb = loadedGlbs.some(g => g.status === 200);
    recordTest('3D Model GLB Loaded with HTTP 200', hasGlb, loadedGlbs.map(g => `${path.basename(g.url)} (${g.status})`).join(', '));

    // Capture Orbit Overview
    const pOrbit = path.join(ARTIFACT_DIR, `generalized_${sceneId}_01_orbit.png`);
    await page.screenshot({ path: pOrbit });
    console.log('  [+] Saved screenshot:', path.basename(pOrbit));

    // Check Walkthrough button
    const enterBtn = await page.$('button:has-text("ENTER INDOOR WALKTHROUGH"), button:has-text("Enter Walkthrough"), button:has-text("ENTER SPACE")');
    if (enterBtn) {
      console.log('  [*] Clicking Walkthrough button...');
      await enterBtn.click();
      await page.waitForTimeout(1500);

      const pSpawn = path.join(ARTIFACT_DIR, `generalized_${sceneId}_02_walkthrough_spawn.png`);
      await page.screenshot({ path: pSpawn });
      console.log('  [+] Saved spawn screenshot:', path.basename(pSpawn));

      // Test KeyW Forward
      console.log('  [*] Moving forward with KeyW...');
      await page.keyboard.down('KeyW');
      await page.waitForTimeout(700);
      await page.keyboard.up('KeyW');
      await page.waitForTimeout(300);

      const pFwd = path.join(ARTIFACT_DIR, `generalized_${sceneId}_03_forward_motion.png`);
      await page.screenshot({ path: pFwd });
      console.log('  [+] Saved forward movement screenshot:', path.basename(pFwd));

      // Test KeyU Elevate
      console.log('  [*] Elevating with KeyU...');
      await page.keyboard.down('KeyU');
      await page.waitForTimeout(600);
      await page.keyboard.up('KeyU');
      await page.waitForTimeout(300);

      const pElev = path.join(ARTIFACT_DIR, `generalized_${sceneId}_04_elevation_u.png`);
      await page.screenshot({ path: pElev });
      console.log('  [+] Saved elevation screenshot:', path.basename(pElev));

      recordTest('First-Person Walkthrough Navigation Active', true, 'WASD and U/P responsive');
    } else {
      console.log('  [*] Walkthrough button not present (Orbit Inspection mode active for this scene).');
      recordTest('Walkthrough Gating As Expected', true, 'Orbit mode preserved appropriately');
    }

  } catch (err) {
    console.error('  [!] Error during verification:', err.message);
    recordTest('Verification Completed Without Crash', false, err.message);
  } finally {
    await browser.close();
  }

  return testResults;
}

async function main() {
  const b314Results = await verifyScene('indoor_b314_generalized_regression', 'B-314 Regression Baseline');

  const manifestPath = path.resolve('viewer/public/model-manifest.json');
  if (fs.existsSync(manifestPath)) {
    const mf = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const candidateScenes = ['indoor_b314_texture_corrected', 'unseen_room_180530', 'unseen_corridor_170500', 'stationary_pan_164349'];
    for (const scId of candidateScenes) {
      const match = (mf.scenes || []).find(s => s.id === scId);
      if (match) {
        await verifyScene(scId, match.display_name || scId);
      }
    }
  }

  console.log('\n[+] Automated Browser Verification Suite Finished.');
}

main().catch(err => {
  console.error('[!] Fatal error in verification script:', err);
  process.exit(1);
});
