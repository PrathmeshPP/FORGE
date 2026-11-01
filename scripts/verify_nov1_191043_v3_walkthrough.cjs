const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const TARGET_URL = 'http://127.0.0.1:5174/?scene=indoor_nov1_191043_photographic_v3';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');
const OUTPUTS_DIR = path.resolve(__dirname, '../outputs/indoor/texture_experiment_video_20261101_191043_v3_quality/renders');

async function run() {
  console.log('[*] Launching browser to verify V3 quality scene at:', TARGET_URL);
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

  console.log('[*] Navigating to V3 quality target scene...');
  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(4000);

  const capturedImages = [];

  // Capture 1: Initial Scene Inspection
  const p1 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_1_initial_inspection.png');
  await page.screenshot({ path: p1 });
  capturedImages.push(p1);
  console.log('[+] Captured initial inspection:', path.basename(p1));

  // Enter Walkthrough
  const enterBtn = await page.$('button:has-text("Enter Walkthrough"), button:has-text("Walkthrough"), button:has-text("Explore")');
  if (enterBtn) {
    await enterBtn.click();
    await page.waitForTimeout(2000);
  }

  // Capture 2: Walkthrough Spawn
  const p2 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_2_walkthrough_spawn.png');
  await page.screenshot({ path: p2 });
  capturedImages.push(p2);
  console.log('[+] Captured walkthrough spawn:', path.basename(p2));

  // Move forward into room (KeyW)
  console.log('[*] Simulating forward navigation into room (KeyW)...');
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(1200);
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(1000);

  // Capture 3: Walk forward
  const p3 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_3_walk_forward_room.png');
  await page.screenshot({ path: p3 });
  capturedImages.push(p3);
  console.log('[+] Captured walk forward:', path.basename(p3));

  // Turn left to Bed 1 and Wardrobes (KeyA)
  console.log('[*] Turning towards Bed 1 and wardrobes (KeyA)...');
  await page.keyboard.down('KeyA');
  await page.waitForTimeout(700);
  await page.keyboard.up('KeyA');
  await page.waitForTimeout(1000);

  // Capture 4: Bed 1 and Wardrobes
  const p4 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_4_bed1_and_wardrobes.png');
  await page.screenshot({ path: p4 });
  capturedImages.push(p4);
  console.log('[+] Captured Bed 1 and wardrobes:', path.basename(p4));

  // Turn right towards Bed 2, Bed 3, and Cooler (KeyD)
  console.log('[*] Turning towards Bed 2 and cooler (KeyD)...');
  await page.keyboard.down('KeyD');
  await page.waitForTimeout(1400);
  await page.keyboard.up('KeyD');
  await page.waitForTimeout(1000);

  // Capture 5: Bed 2 and Cooler
  const p5 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_5_bed2_and_cooler.png');
  await page.screenshot({ path: p5 });
  capturedImages.push(p5);
  console.log('[+] Captured Bed 2 and cooler:', path.basename(p5));

  // Exit walkthrough mode
  await page.keyboard.press('Escape');
  await page.waitForTimeout(1000);

  // Capture 6: Exit inspection
  const p6 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_6_exit_inspection.png');
  await page.screenshot({ path: p6 });
  capturedImages.push(p6);
  console.log('[+] Captured exit inspection:', path.basename(p6));

  // Click Top Camera Preset
  const topPresetBtn = await page.$('button:has-text("Top"), button:has-text("Overview"), button:has-text("Floorplan")');
  if (topPresetBtn) {
    await topPresetBtn.click();
    await page.waitForTimeout(2000);
  }

  // Capture 7: Top preset
  const p7 = path.join(ARTIFACT_DIR, 'viewer_nov1_191043_v3_7_top_preset.png');
  await page.screenshot({ path: p7 });
  capturedImages.push(p7);
  console.log('[+] Captured top preset:', path.basename(p7));

  await browser.close();
  console.log('[+] Playwright walkthrough verification complete! Captured 7 viewports.');
}

run().catch(err => {
  console.error('[!] Verification failed:', err);
  process.exit(1);
});

