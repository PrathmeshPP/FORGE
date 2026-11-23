const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');
const path = require('path');
const fs = require('fs');

const OUT_DIR_INDOOR = path.resolve(__dirname, '../outputs/indoor/room_001/final_release/screenshots');
const OUT_DIR_VIEWER = path.resolve(__dirname, '../viewer/screenshots');

fs.mkdirSync(OUT_DIR_INDOOR, { recursive: true });
fs.mkdirSync(OUT_DIR_VIEWER, { recursive: true });

async function capture() {
  console.log('Launching Microsoft Edge for F.O.R.G.E. Indoor Room Visual Validation...');
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

  const port = process.env.VIEWER_PORT || '8080';
  const url = `http://127.0.0.1:${port}/`;
  console.log(`Navigating to ${url} ...`);
  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  // 1. Select Indoor Room 001
  console.log('\n[1/4] Selecting Indoor Room 001 from Scene Library...');
  await page.click('.scene-card:nth-child(1)');
  await page.waitForTimeout(4000);

  const canvas = await page.$('canvas');
  if (!canvas) {
    throw new Error('Canvas not found');
  }

  // Save Overview Screenshot
  const path1 = path.join(OUT_DIR_INDOOR, 'viewer_indoor_room_overview.png');
  await page.screenshot({ path: path1 });
  fs.copyFileSync(path1, path.join(OUT_DIR_VIEWER, 'viewer_indoor_room_overview.png'));
  console.log(`Saved Overview: ${path1} (${fs.statSync(path1).size} bytes)`);

  // 2. Orbit interaction
  console.log('\n[2/4] Executing Orbit Interaction on Canvas...');
  const box = await canvas.boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 180, box.y + box.height / 2 - 80, { steps: 15 });
    await page.mouse.up();
    await page.waitForTimeout(1500);
  }

  const path2 = path.join(OUT_DIR_INDOOR, 'viewer_indoor_room_orbited.png');
  await page.screenshot({ path: path2 });
  fs.copyFileSync(path2, path.join(OUT_DIR_VIEWER, 'viewer_indoor_room_orbited.png'));
  console.log(`Saved Orbited View: ${path2} (${fs.statSync(path2).size} bytes)`);

  // 3. Zoom into Furniture (desk & chair)
  console.log('\n[3/4] Zooming in to inspect furniture details...');
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -500);
    await page.waitForTimeout(1500);
  }

  const path3 = path.join(OUT_DIR_INDOOR, 'viewer_indoor_room_furniture_inspection.png');
  await page.screenshot({ path: path3 });
  fs.copyFileSync(path3, path.join(OUT_DIR_VIEWER, 'viewer_indoor_room_furniture_inspection.png'));
  console.log(`Saved Furniture Inspection: ${path3} (${fs.statSync(path3).size} bytes)`);

  // 4. Switch to PLY representation
  console.log('\n[4/4] Testing representation switch...');
  const plyBtn = await page.$('button:has-text("Photographic Vertex PLY")');
  if (plyBtn) {
    await plyBtn.click();
    await page.waitForTimeout(3000);
  }

  const path4 = path.join(OUT_DIR_INDOOR, 'viewer_indoor_room_ply_view.png');
  await page.screenshot({ path: path4 });
  fs.copyFileSync(path4, path.join(OUT_DIR_VIEWER, 'viewer_indoor_room_ply_view.png'));
  console.log(`Saved PLY View: ${path4} (${fs.statSync(path4).size} bytes)`);

  await browser.close();
  console.log('\n[+] All viewer validation screenshots captured successfully!');
}

capture().catch(err => {
  console.error('Fatal capture error:', err);
  process.exit(1);
});
