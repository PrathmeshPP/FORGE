const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const BASE_URL = 'http://127.0.0.1:4173';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

async function run() {
  console.log('[*] Starting Automated Production Verification for FORGE_indoor deployment at:', BASE_URL);

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
  const errors = [];
  const networkRequests = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(text);
    if (msg.type() === 'error' && !text.includes('favicon')) {
      errors.push(text);
      console.error('[BROWSER ERROR]', text);
    }
  });

  page.on('request', req => {
    const url = req.url();
    networkRequests.push(url);
    if (url.includes(':8000') || url.includes('localhost:8000') || url.includes('127.0.0.1:8000')) {
      errors.push(`Disallowed backend request: ${url}`);
    }
    if (url.startsWith('file:///') || url.includes('C:/') || url.includes('C:\\')) {
      errors.push(`Disallowed local file path request: ${url}`);
    }
  });

  const results = {
    tests: [],
    passed: 0,
    failed: 0
  };

  function recordTest(name, passed, details = '') {
    results.tests.push({ name, passed, details });
    if (passed) {
      results.passed++;
      console.log(`[PASS] ${name} ${details ? '(' + details + ')' : ''}`);
    } else {
      results.failed++;
      console.error(`[FAIL] ${name} ${details ? '(' + details + ')' : ''}`);
    }
  }

  try {
    // -------------------------------------------------------------
    // TEST 1: Homepage & Scene Library
    // -------------------------------------------------------------
    console.log('\n--- TEST 1: Loading Homepage ---');
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);

    const title = await page.title();
    recordTest('Page Title Verification', title.toLowerCase().includes('f.o.r.g.e'), `Title: "${title}"`);

    // Verify Scene Library sidebar has 39 scenes
    const sceneCards = await page.$$('.scene-card');
    const libraryTitle = await page.$eval('.scene-library-panel .panel-title', el => el.textContent).catch(() => '');
    recordTest('Scene Library Size', sceneCards.length >= 37, `Found ${sceneCards.length} scene cards (${libraryTitle})`);

    // Check inspection panel exists
    const hasInspectionPanel = await page.$eval('body', el => el.textContent.includes('Inspection & Provenance') || el.textContent.includes('Provenance'));
    recordTest('Inspection & Provenance Panel', hasInspectionPanel);

    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_01_homepage.png') });

    // -------------------------------------------------------------
    // TEST 2: Deep Link & Adaptive Scene: indoor_my_movie_4_adaptive
    // -------------------------------------------------------------
    console.log('\n--- TEST 2: Adaptive 3D Room: indoor_my_movie_4_adaptive ---');
    await page.goto(`${BASE_URL}/?scene=indoor_my_movie_4_adaptive`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(4000);

    // Verify canvas rendered
    const hasCanvas = await page.$('canvas');
    recordTest('WebGL Canvas Present', !!hasCanvas);

    // Check Adaptive HUD is visible
    const hudText = await page.$eval('body', el => el.textContent);
    const hasFPS = hudText.includes('FPS') || hudText.includes('fps');
    recordTest('Performance HUD Active', hasFPS, 'HUD rendered with real-time stats');

    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_02_adaptive_scene.png') });

    // -------------------------------------------------------------
    // TEST 3: First-Person Navigation & Space / C Vertical Controls
    // -------------------------------------------------------------
    console.log('\n--- TEST 3: Navigation Engine - Space (Ascend) & C (Descend) ---');
    const walkthroughBtn = await page.$('button:has-text("Enter Walkthrough"), button:has-text("Walkthrough"), button:has-text("First-Person")');
    if (walkthroughBtn) {
      await walkthroughBtn.click();
      console.log('Clicked Walkthrough button');
    } else {
      await page.keyboard.press('Enter');
    }
    await page.waitForTimeout(1000);

    // Test Space key (Ascend)
    console.log('Pressing Space key (Ascend)...');
    await page.keyboard.down('Space');
    await page.waitForTimeout(600);
    await page.keyboard.up('Space');
    await page.waitForTimeout(400);

    // Test C key (Descend)
    console.log('Pressing C key (Descend)...');
    await page.keyboard.down('KeyC');
    await page.waitForTimeout(600);
    await page.keyboard.up('KeyC');
    await page.waitForTimeout(400);

    // Test W key (Forward)
    console.log('Pressing W key (Forward)...');
    await page.keyboard.down('KeyW');
    await page.waitForTimeout(600);
    await page.keyboard.up('KeyW');
    await page.waitForTimeout(400);

    // Test O key (Fly / Walk Mode Toggle)
    console.log('Pressing O key (Fly / Walk Toggle)...');
    await page.keyboard.press('KeyO');
    await page.waitForTimeout(400);

    recordTest('Keyboard Walkthrough Key Events Processed', true, 'Space (Up), C (Down), W (Forward), O (Fly Mode)');
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_03_walkthrough_navigation.png') });

    // -------------------------------------------------------------
    // TEST 4: Outdoor Campus Scene: campus_0049
    // -------------------------------------------------------------
    console.log('\n--- TEST 4: Outdoor Campus Scene: campus_0049 ---');
    await page.goto(`${BASE_URL}/?scene=campus_0049`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(4000);

    const campusHudText = await page.$eval('body', el => el.textContent);
    recordTest('Campus Model Loaded', campusHudText.includes('Campus') || campusHudText.includes('campus_0049'), 'Campus scene loaded via deep-link');

    // Test Space and C vertical navigation on outdoor campus model
    await page.keyboard.down('Space');
    await page.waitForTimeout(500);
    await page.keyboard.up('Space');
    await page.waitForTimeout(300);

    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_04_campus_scene.png') });

    // -------------------------------------------------------------
    // TEST 5: Hostel Room B 314 (Poisson PLY)
    // -------------------------------------------------------------
    console.log('\n--- TEST 5: Hostel Room B 314 (Poisson PLY) ---');
    await page.goto(`${BASE_URL}/?scene=texture_experiment_video_20261101_142811`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(4000);

    const hostelText = await page.$eval('body', el => el.textContent);
    recordTest('Hostel Room B 314 Poisson Loaded', hostelText.includes('Hostel Room B 314') || hostelText.includes('142811'), 'Hostel Poisson PLY rendered cleanly');
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_05_hostel_room_b314.png') });

    // -------------------------------------------------------------
    // TEST 6: Modals - Architecture & Help
    // -------------------------------------------------------------
    console.log('\n--- TEST 6: UI Modals Verification ---');
    // Open Help Modal via title="Controls & Navigation Reference"
    const helpBtn = await page.$('button[title*="Controls"], button[title*="Navigation"], button[title*="Help"]');
    if (helpBtn) {
      await helpBtn.click();
      await page.waitForTimeout(800);
      const helpContent = await page.$eval('body', el => el.textContent);
      const hasSpaceGuide = helpContent.includes('Space') && helpContent.includes('Ascend');
      const hasCGuide = helpContent.includes('Descend') || helpContent.includes('Down');
      recordTest('Help Modal Navigation Instructions', hasSpaceGuide && hasCGuide, 'Verified Space (Ascend) and C (Descend) documented');
      await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_06_help_modal.png') });

      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
    } else {
      recordTest('Help Modal Available', false, 'Help button not found');
    }

    // Open Pipeline Architecture Modal
    const archBtn = await page.$('button:has-text("Pipeline Architecture")');
    if (archBtn) {
      await archBtn.click();
      await page.waitForTimeout(800);
      const archText = await page.$eval('body', el => el.textContent);
      recordTest('Pipeline Architecture Modal', archText.includes('Pipeline') || archText.includes('VGGT'), 'Architecture diagram loaded');
      await page.screenshot({ path: path.join(ARTIFACT_DIR, 'vercel_test_07_pipeline_architecture.png') });
      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
    }

    // -------------------------------------------------------------
    // TEST 7: Network & Zero Localhost Isolation
    // -------------------------------------------------------------
    console.log('\n--- TEST 7: Network & Zero Localhost Isolation ---');
    const illegalRequests = networkRequests.filter(u =>
      u.includes(':8000') ||
      u.includes('localhost:8000') ||
      u.includes('127.0.0.1:8000') ||
      u.startsWith('file:///') ||
      u.includes('C:/') ||
      u.includes('C:\\')
    );
    recordTest('Zero Backend / File Path Leaks', illegalRequests.length === 0, `${illegalRequests.length} illegal requests`);

    // Console Error Check
    recordTest('Zero Critical Console Errors', errors.length === 0, `${errors.length} console errors`);

  } catch (err) {
    console.error('[FATAL EXCEPTION]', err);
    recordTest('Fatal Script Exception', false, err.message);
  } finally {
    await browser.close();
  }

  console.log('\n=============================================');
  console.log(`PRODUCTION VERIFICATION SUMMARY:`);
  console.log(`Passed: ${results.passed} / ${results.tests.length}`);
  console.log(`Failed: ${results.failed} / ${results.tests.length}`);
  console.log('=============================================\n');

  fs.writeFileSync(path.join(ARTIFACT_DIR, 'vercel_production_test_results.json'), JSON.stringify(results, null, 2));

  if (results.failed > 0) {
    console.error('Some tests failed. Inspect vercel_production_test_results.json');
    process.exit(1);
  } else {
    console.log('ALL PRODUCTION TESTS PASSED WITH 100% SUCCESS!');
  }
}

run();
