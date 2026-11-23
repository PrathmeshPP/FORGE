const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');
const path = require('path');
const fs = require('fs');

const OUT_DIR_VALIDATION = path.resolve(__dirname, '../outputs/indoor/room_001/attempt_4_new_video/viewer_validation');
const OUT_DIR_VIEWER = path.resolve(__dirname, '../viewer/screenshots/attempt_4_new_video');

fs.mkdirSync(OUT_DIR_VALIDATION, { recursive: true });
fs.mkdirSync(OUT_DIR_VIEWER, { recursive: true });

async function capture() {
  console.log('Launching Edge for Attempt 4 New Video Three.js Viewer Validation...');
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
    viewport: { width: 1440, height: 900 }
  });

  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error('BROWSER ERROR:', msg.text());
    } else {
      console.log('BROWSER LOG:', msg.text());
    }
  });

  const port = process.env.VIEWER_PORT || '8088';
  const url = `http://127.0.0.1:${port}/`;
  console.log(`Navigating to ${url} ...`);
  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  // 1. Click on the first scene: Computer Lab (New Video Source)
  console.log('\n[1/6] Selecting Computer Lab (New Video Source) from Scene Library...');
  const firstCard = await page.$('.scene-card:nth-child(1)');
  if (firstCard) {
    await firstCard.click();
  }
  await page.waitForTimeout(5000);

  const canvas = await page.$('canvas');
  if (!canvas) {
    throw new Error('Canvas not found on page');
  }

  // View 1: Full Room Overview
  const p1 = path.join(OUT_DIR_VALIDATION, 'viewer_room_overview.png');
  await page.screenshot({ path: p1 });
  fs.copyFileSync(p1, path.join(OUT_DIR_VIEWER, 'viewer_room_overview.png'));
  console.log(`[+] Saved View 1 (Room Overview): ${p1}`);

  // View 2: Oblique Angle Orbit
  console.log('\n[2/6] Performing Orbit Interaction to Oblique Perspective...');
  const box = await canvas.boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 150, box.y + box.height / 2 - 60, { steps: 12 });
    await page.mouse.up();
    await page.waitForTimeout(1500);
  }
  const p2 = path.join(OUT_DIR_VALIDATION, 'viewer_room_oblique.png');
  await page.screenshot({ path: p2 });
  fs.copyFileSync(p2, path.join(OUT_DIR_VIEWER, 'viewer_room_oblique.png'));
  console.log(`[+] Saved View 2 (Oblique Angle): ${p2}`);

  // View 3: Eye-level inspection
  console.log('\n[3/6] Navigating to Eye-level viewpoint...');
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 - 50, box.y + box.height / 2 + 80, { steps: 12 });
    await page.mouse.up();
    await page.waitForTimeout(1500);
  }
  const p3 = path.join(OUT_DIR_VALIDATION, 'viewer_room_eye_level.png');
  await page.screenshot({ path: p3 });
  fs.copyFileSync(p3, path.join(OUT_DIR_VIEWER, 'viewer_room_eye_level.png'));
  console.log(`[+] Saved View 3 (Eye Level): ${p3}`);

  // View 4: Top-Down Plan View
  console.log('\n[4/6] Switching to Top-Down Plan View...');
  const topPreset = await page.$('button[title*="Top" i], button:has-text("Top")');
  if (topPreset) {
    await topPreset.click();
    await page.waitForTimeout(1500);
  } else if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 + 180, { steps: 15 });
    await page.mouse.up();
    await page.waitForTimeout(1500);
  }
  const p4 = path.join(OUT_DIR_VALIDATION, 'viewer_room_top_plan.png');
  await page.screenshot({ path: p4 });
  fs.copyFileSync(p4, path.join(OUT_DIR_VIEWER, 'viewer_room_top_plan.png'));
  console.log(`[+] Saved View 4 (Top-Down Plan): ${p4}`);

  // View 5: Furniture Close-up (Zooming into Computer Desks and Chairs)
  console.log('\n[5/6] Zooming in to inspect furniture details...');
  if (box) {
    // Reset view
    const resetBtn = await page.$('button[title*="Reset" i], button:has-text("Reset")');
    if (resetBtn) await resetBtn.click();
    await page.waitForTimeout(1000);

    // Zoom in
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -600);
    await page.waitForTimeout(1500);
  }
  const p5 = path.join(OUT_DIR_VALIDATION, 'viewer_furniture_closeup.png');
  await page.screenshot({ path: p5 });
  fs.copyFileSync(p5, path.join(OUT_DIR_VIEWER, 'viewer_furniture_closeup.png'));
  console.log(`[+] Saved View 5 (Furniture Close-up): ${p5}`);

  // View 6: Switch to Photographic Vertex PLY Representation
  console.log('\n[6/7] Testing Representation switch to Photo PLY...');
  const plyBtn = await page.$('button:has-text("Photo PLY")');
  if (plyBtn) {
    await plyBtn.click();
    await page.waitForTimeout(5000);
    const p6 = path.join(OUT_DIR_VALIDATION, 'viewer_photographic_ply_view.png');
    await page.screenshot({ path: p6 });
    fs.copyFileSync(p6, path.join(OUT_DIR_VIEWER, 'viewer_photographic_ply_view.png'));
    console.log(`[+] Saved View 6 (Photographic PLY): ${p6}`);
  }

  // View 7: Switch to Interactive GLB Model Representation
  console.log('\n[7/7] Testing Representation switch to Interactive GLB Model...');
  const glbBtn = await page.$('button:has-text("Interactive GLB")');
  if (glbBtn) {
    await glbBtn.click();
    await page.waitForTimeout(5000);
    const p7 = path.join(OUT_DIR_VALIDATION, 'viewer_interactive_glb_view.png');
    await page.screenshot({ path: p7 });
    fs.copyFileSync(p7, path.join(OUT_DIR_VIEWER, 'viewer_interactive_glb_view.png'));
    console.log(`[+] Saved View 7 (Interactive GLB Model): ${p7}`);
  }

  await browser.close();
  console.log('\n[+] All viewer validation screenshots captured successfully!');
}

capture().catch(err => {
  console.error('Capture failed:', err);
  process.exit(1);
});
