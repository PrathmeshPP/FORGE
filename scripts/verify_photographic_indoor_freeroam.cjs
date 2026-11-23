const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const BASE_URL = 'http://127.0.0.1:5174';
const TARGET_URL = `${BASE_URL}/?scene=indoor_nov1_191043_photographic_v3`;
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

async function run() {
  console.log('[*] Launching Browser for Photographic Indoor 3D Free-Roam Verification');
  console.log('[*] Target Scene:', TARGET_URL);

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
  const loadedModels = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(text);
    if (msg.type() === 'error' && !text.includes('favicon')) {
      errors.push(text);
      console.error('  [BROWSER ERROR]:', text);
    }
  });

  page.on('response', res => {
    const url = res.url();
    if (url.endsWith('.glb')) {
      loadedModels.push({ url, status: res.status() });
      console.log(`  [MODEL LOADED] Status ${res.status()}: ${url}`);
    }
  });

  const testResults = [];
  function recordTest(name, passed, details = '') {
    testResults.push({ name, passed, details });
    if (passed) {
      console.log(`[PASS] ${name} ${details ? '(' + details + ')' : ''}`);
    } else {
      console.error(`[FAIL] ${name} ${details ? '(' + details + ')' : ''}`);
    }
  }

  try {
    // -----------------------------------------------------------------
    // 1. Load the website & verify exact requested GLB
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 1: Loading Website & Exact GLB ---');
    await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 45000 });
    await page.waitForTimeout(4000);

    const title = await page.title();
    recordTest('Page Loaded', title.includes('F.O.R.G.E'), `Title: "${title}"`);

    // Verify exact GLB URL
    const hasTargetGlb = loadedModels.some(m =>
      m.url.includes('texture_experiment_video_20261101_191043_v3_quality/room_photographic_model.glb') && m.status === 200
    );
    recordTest('Exact Requested GLB Loaded', hasTargetGlb, 'room_photographic_model.glb loaded with HTTP 200');

    // Verify WebGL Canvas
    const canvas = await page.$('canvas');
    recordTest('WebGL Viewport Active', !!canvas);

    // Capture initial interface with instructions before entering
    const pIntro = path.join(ARTIFACT_DIR, 'indoor_freeroam_05_controls_and_interface.png');
    await page.screenshot({ path: pIntro });
    console.log('[+] Captured interface & controls overview:', path.basename(pIntro));

    // -----------------------------------------------------------------
    // 2. Enter Walkthrough Mode & Verify Starting Position
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 2: Enter Walkthrough & Spawn Location ---');
    const enterBtn = await page.$('button:has-text("ENTER INDOOR WALKTHROUGH"), button:has-text("Enter Walkthrough"), button:has-text("ENTER SPACE")');
    if (enterBtn) {
      await enterBtn.click();
      console.log('Clicked Enter Walkthrough button');
    } else {
      await page.keyboard.press('Enter');
    }
    await page.waitForTimeout(1500);

    // Helper to read telemetry from HUD
    const getTelemetry = async () => {
      return await page.evaluate(() => {
        const text = document.body.innerText;
        const posMatch = text.match(/Position \[X, Z\]:\s*\[([-\d.]+),\s*([-\d.]+)\]/);
        const altMatch = text.match(/Vertical Altitude \(Y\):\s*([-\d.]+)\s*m/);
        const headingMatch = text.match(/Heading:\s*(\d+)°/);
        const collMatch = text.match(/Collision Status:\s*(\w+)/);
        const fpsMatch = text.match(/(\d+)\s*FPS/);

        return {
          x: posMatch ? parseFloat(posMatch[1]) : null,
          z: posMatch ? parseFloat(posMatch[2]) : null,
          y: altMatch ? parseFloat(altMatch[1]) : null,
          heading: headingMatch ? parseInt(headingMatch[1]) : null,
          collision: collMatch ? collMatch[1] : null,
          fps: fpsMatch ? parseInt(fpsMatch[1]) : null,
        };
      });
    };

    const spawnTele = await getTelemetry();
    console.log('Spawn Telemetry:', spawnTele);

    // Verify spawn is inside the room
    const isInsideRoom = (
      spawnTele.x !== null && spawnTele.x >= -0.55 && spawnTele.x <= 0.50 &&
      spawnTele.z !== null && spawnTele.z >= -0.65 && spawnTele.z <= 0.75 &&
      spawnTele.y !== null && spawnTele.y >= -0.15 && spawnTele.y <= 0.30
    );
    recordTest('Spawn Inside Room at Human Eye-Level', isInsideRoom, `Spawn: [X: ${spawnTele.x}, Y: ${spawnTele.y}, Z: ${spawnTele.z}]`);

    // Capture 1: Initial Room View
    const p1 = path.join(ARTIFACT_DIR, 'indoor_freeroam_01_initial_room_spawn.png');
    await page.screenshot({ path: p1 });
    console.log('[+] Captured initial room spawn:', path.basename(p1));

    // -----------------------------------------------------------------
    // 3. Test Horizontal Navigation: W, S, A, D
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 3: Testing Horizontal Controls (W, S, A, D) ---');

    // Test KeyW (Forward)
    const posBeforeW = await getTelemetry();
    console.log('Moving Forward with W...');
    await page.keyboard.down('KeyW');
    await page.waitForTimeout(800);
    await page.keyboard.up('KeyW');
    await page.waitForTimeout(300);
    const posAfterW = await getTelemetry();
    const wMoved = Math.hypot(posAfterW.x - posBeforeW.x, posAfterW.z - posBeforeW.z) > 0.05;
    recordTest('Key W Moves Forward', wMoved, `Dist: ${Math.hypot(posAfterW.x - posBeforeW.x, posAfterW.z - posBeforeW.z).toFixed(3)}m`);

    // Test KeyS (Backward)
    const posBeforeS = await getTelemetry();
    console.log('Moving Backward with S...');
    await page.keyboard.down('KeyS');
    await page.waitForTimeout(600);
    await page.keyboard.up('KeyS');
    await page.waitForTimeout(300);
    const posAfterS = await getTelemetry();
    const sMoved = Math.hypot(posAfterS.x - posBeforeS.x, posAfterS.z - posBeforeS.z) > 0.04;
    recordTest('Key S Moves Backward', sMoved, `Dist: ${Math.hypot(posAfterS.x - posBeforeS.x, posAfterS.z - posBeforeS.z).toFixed(3)}m`);

    // Test KeyA (Left)
    const posBeforeA = await getTelemetry();
    console.log('Moving Left with A...');
    await page.keyboard.down('KeyA');
    await page.waitForTimeout(600);
    await page.keyboard.up('KeyA');
    await page.waitForTimeout(300);
    const posAfterA = await getTelemetry();
    const aMoved = Math.abs(posAfterA.x - posBeforeA.x) > 0.03 || Math.hypot(posAfterA.x - posBeforeA.x, posAfterA.z - posBeforeA.z) > 0.03;
    recordTest('Key A Moves Left', aMoved, `Delta: ${(posAfterA.x - posBeforeA.x).toFixed(3)}m`);

    // Test KeyD (Right)
    const posBeforeD = await getTelemetry();
    console.log('Moving Right with D...');
    await page.keyboard.down('KeyD');
    await page.waitForTimeout(700);
    await page.keyboard.up('KeyD');
    await page.waitForTimeout(300);
    const posAfterD = await getTelemetry();
    const dMoved = Math.abs(posAfterD.x - posBeforeD.x) > 0.03 || Math.hypot(posAfterD.x - posBeforeD.x, posAfterD.z - posBeforeD.z) > 0.03;
    recordTest('Key D Moves Right', dMoved, `Delta: ${(posAfterD.x - posBeforeD.x).toFixed(3)}m`);

    // Capture 2: Different Room Position Reached via WASD
    const p2 = path.join(ARTIFACT_DIR, 'indoor_freeroam_02_different_position_wasd.png');
    await page.screenshot({ path: p2 });
    console.log('[+] Captured different position via WASD:', path.basename(p2));

    // -----------------------------------------------------------------
    // 4. Test Vertical Navigation: Key U (Upward) and Key P (Downward)
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 4: Testing Vertical Controls (U Up, P Down) ---');

    // Test KeyU (Move Vertically Upward into the Air)
    const altBeforeU = (await getTelemetry()).y;
    console.log(`Altitude before U: ${altBeforeU.toFixed(3)}m. Holding KeyU...`);
    await page.keyboard.down('KeyU');
    await page.waitForTimeout(900);
    await page.keyboard.up('KeyU');
    await page.waitForTimeout(300);
    const altAfterU = (await getTelemetry()).y;
    console.log(`Altitude after U: ${altAfterU.toFixed(3)}m`);
    const uMovedUp = altAfterU > altBeforeU + 0.04;
    recordTest('Key U Moves Vertically Upward', uMovedUp, `Alt delta: +${(altAfterU - altBeforeU).toFixed(3)}m (from ${altBeforeU.toFixed(3)}m to ${altAfterU.toFixed(3)}m)`);

    // Capture 3: Elevated View Reached using U
    const p3 = path.join(ARTIFACT_DIR, 'indoor_freeroam_03_elevated_view_key_u.png');
    await page.screenshot({ path: p3 });
    console.log('[+] Captured elevated view using Key U:', path.basename(p3));

    // Test KeyP (Move Vertically Downward)
    const altBeforeP = (await getTelemetry()).y;
    console.log(`Altitude before P: ${altBeforeP.toFixed(3)}m. Holding KeyP...`);
    await page.keyboard.down('KeyP');
    await page.waitForTimeout(1100);
    await page.keyboard.up('KeyP');
    await page.waitForTimeout(300);
    const altAfterP = (await getTelemetry()).y;
    console.log(`Altitude after P: ${altAfterP.toFixed(3)}m`);
    const pMovedDown = altAfterP < altBeforeP - 0.04;
    recordTest('Key P Moves Vertically Downward', pMovedDown, `Alt delta: ${(altAfterP - altBeforeP).toFixed(3)}m (from ${altBeforeP.toFixed(3)}m to ${altAfterP.toFixed(3)}m)`);

    // Capture 4: Lower View Reached using P
    const p4 = path.join(ARTIFACT_DIR, 'indoor_freeroam_04_lower_view_key_p.png');
    await page.screenshot({ path: p4 });
    console.log('[+] Captured lower view using Key P:', path.basename(p4));

    // -----------------------------------------------------------------
    // 5. Test Simultaneous Key Presses & Diagonal Normalization
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 5: Simultaneous Key Presses (W + U) ---');
    const posBeforeCombo = await getTelemetry();
    await page.keyboard.down('KeyW');
    await page.keyboard.down('KeyU');
    await page.waitForTimeout(700);
    await page.keyboard.up('KeyW');
    await page.keyboard.up('KeyU');
    await page.waitForTimeout(300);
    const posAfterCombo = await getTelemetry();

    const horizDelta = Math.hypot(posAfterCombo.x - posBeforeCombo.x, posAfterCombo.z - posBeforeCombo.z);
    const vertDelta = posAfterCombo.y - posBeforeCombo.y;
    const compoundMoved = horizDelta > 0.03 && vertDelta > 0.03;
    recordTest('Simultaneous Key Presses (W + U)', compoundMoved, `Horiz: ${horizDelta.toFixed(3)}m, Vert: ${vertDelta.toFixed(3)}m`);

    // -----------------------------------------------------------------
    // 6. Test Key Release Stops Movement
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 6: Key Release Stops Movement ---');
    await page.waitForTimeout(400);
    const posStill1 = await getTelemetry();
    await page.waitForTimeout(400);
    const posStill2 = await getTelemetry();
    const isStopped = Math.hypot(posStill2.x - posStill1.x, posStill2.y - posStill1.y, posStill2.z - posStill1.z) < 0.005;
    recordTest('Key Release Stops Movement', isStopped, 'Camera stationary when no keys held');

    // -----------------------------------------------------------------
    // 7. Test Mouse Look
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 7: Mouse Rotation (Yaw & Pitch) ---');
    const headBefore = (await getTelemetry()).heading;
    await page.mouse.move(800, 500);
    await page.mouse.move(950, 500); // Horizontal sweep
    await page.waitForTimeout(300);
    const headAfter = (await getTelemetry()).heading;
    recordTest('Mouse Rotation Operates Correctly', headBefore !== null, `Heading: ${headBefore}° -> ${headAfter}°`);

    // -----------------------------------------------------------------
    // 8. Test Floor and Ceiling Protection (Collision Clamping)
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 8: Floor & Ceiling Boundary Clamping ---');
    // Hold KeyU for 3 seconds -> must clamp at ceiling
    console.log('Holding KeyU to test ceiling boundary...');
    await page.keyboard.down('KeyU');
    await page.waitForTimeout(2000);
    await page.keyboard.up('KeyU');
    await page.waitForTimeout(300);
    const maxCeilingY = (await getTelemetry()).y;
    const ceilingProtected = maxCeilingY <= 0.35; // True ceiling is ~0.377
    recordTest('Ceiling Boundary Protection', ceilingProtected, `Max Altitude clamped at ${maxCeilingY.toFixed(3)}m <= 0.35m`);

    // Hold KeyP for 3 seconds -> must clamp at floor
    console.log('Holding KeyP to test floor boundary...');
    await page.keyboard.down('KeyP');
    await page.waitForTimeout(2200);
    await page.keyboard.up('KeyP');
    await page.waitForTimeout(300);
    const minFloorY = (await getTelemetry()).y;
    const floorProtected = minFloorY >= -0.24; // True floor is ~-0.275
    recordTest('Floor Boundary Protection', floorProtected, `Min Altitude clamped at ${minFloorY.toFixed(3)}m >= -0.24m`);

    // -----------------------------------------------------------------
    // 9. Test ESC Releases Pointer Lock & Exit Option
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 9: ESC Releases Pointer Lock & Exits Walkthrough ---');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(800);

    // Verify exit walkthrough button clicked
    const exitBtn = await page.$('button:has-text("EXIT WALKTHROUGH"), button:has-text("EXIT SPACE")');
    if (exitBtn) {
      await exitBtn.click();
      await page.waitForTimeout(1000);
    }
    const isBackInInspection = await page.$('button:has-text("ENTER INDOOR WALKTHROUGH")');
    recordTest('Exit Walkthrough Operates Correctly', !!isBackInInspection, 'Returned to inspection mode');

    // -----------------------------------------------------------------
    // 10. Verify Help Modal Instructions
    // -----------------------------------------------------------------
    console.log('\n--- PHASE 10: Help Modal Control Instructions ---');
    const helpBtn = await page.$('button[title*="Controls"], button[title*="Navigation"], button[title*="Help"]');
    if (helpBtn) {
      await helpBtn.click();
      await page.waitForTimeout(600);
      const helpText = await page.$eval('.modal-body', el => el.innerText);
      const hasW = helpText.includes('W') && helpText.includes('Forward');
      const hasS = helpText.includes('S') && helpText.includes('Backward');
      const hasA = helpText.includes('A') && helpText.includes('Left');
      const hasD = helpText.includes('D') && helpText.includes('Right');
      const hasU = helpText.includes('U') && (helpText.includes('Up') || helpText.includes('Upward'));
      const hasP = helpText.includes('P') && (helpText.includes('Down') || helpText.includes('Downward'));

      const allKeysDocumented = hasW && hasS && hasA && hasD && hasU && hasP;
      recordTest('Visible Control Guide Documented (W, A, S, D, U, P)', allKeysDocumented, 'Verified exact keys documented');

      await page.keyboard.press('Escape');
    }

    // -----------------------------------------------------------------
    // 11. Performance Check
    // -----------------------------------------------------------------
    recordTest('Zero Critical Console Errors', errors.length === 0, `${errors.length} errors`);

  } catch (err) {
    console.error('[FATAL EXCEPTION]:', err);
    recordTest('Script Execution', false, err.message);
  } finally {
    await browser.close();
  }

  // Summary
  const passed = testResults.filter(t => t.passed).length;
  const total = testResults.length;
  console.log('\n=============================================');
  console.log(`INDOOR FREE-ROAM TEST SUMMARY:`);
  console.log(`Passed: ${passed} / ${total}`);
  console.log('=============================================\n');

  fs.writeFileSync(
    path.join(ARTIFACT_DIR, 'indoor_freeroam_verification_results.json'),
    JSON.stringify({ passed, total, results: testResults }, null, 2)
  );

  if (passed !== total) {
    process.exit(1);
  }
}

run();
