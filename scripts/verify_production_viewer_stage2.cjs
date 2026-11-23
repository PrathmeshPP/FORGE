const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8089;
const BASE_DIR = path.resolve(__dirname, '..');

async function testProductionViewer() {
  console.log('[*] Starting F.O.R.G.E. Production Server on port ' + PORT + '...');
  const server = spawn('python', ['scripts/serve_viewer.py', '--port', String(PORT)], {
    cwd: BASE_DIR,
    stdio: 'inherit'
  });

  await new Promise(r => setTimeout(r, 2000));

  console.log('[*] Launching Headless Edge to test actual production viewer...');
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
    viewport: { width: 1600, height: 950 }
  });

  page.on('console', msg => console.log('    [PRODUCTION VIEWER ' + msg.type() + ']:', msg.text()));
  page.on('pageerror', err => console.error('    [PRODUCTION VIEWER ERROR]:', err.message));

  const outDirV1 = path.join(BASE_DIR, 'outputs/indoor/stage2_texture_video1/diagnostic_renders');
  const outDirV2 = path.join(BASE_DIR, 'outputs/indoor/stage2_texture_video2/diagnostic_renders');

  try {
    console.log(`\n[*] Navigating to production viewer: http://127.0.0.1:${PORT}`);
    await page.goto(`http://127.0.0.1:${PORT}`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(4000);

    // 1. Capture Room B 314 in Production Viewer (first scene)
    const shotRoomPath = path.join(outDirV1, 'production_viewer_room_b314.png');
    await page.screenshot({ path: shotRoomPath });
    console.log(`[+] Captured Production Viewer Room B 314: ${path.basename(shotRoomPath)}`);

    // 2. Toggle wireframe button
    const wireframeBtn = await page.$('button[title*="Wireframe"], button:has-text("Wireframe")');
    if (wireframeBtn) {
      await wireframeBtn.click();
      await page.waitForTimeout(1500);
      const shotWirePath = path.join(outDirV1, 'production_viewer_room_b314_wireframe.png');
      await page.screenshot({ path: shotWirePath });
      console.log(`[+] Captured Production Viewer Wireframe Toggle: ${path.basename(shotWirePath)}`);
      // Toggle back off
      await wireframeBtn.click();
      await page.waitForTimeout(1000);
    }

    // 3. Switch to Corridor Scene via Scene Card
    console.log('\n[*] Selecting Corridor scene in production viewer...');
    const corridorCard = await page.$('.scene-card:has-text("Corridor")');
    if (corridorCard) {
      await corridorCard.click();
      console.log('    Clicked Corridor card, waiting for model to load...');
      await page.waitForTimeout(5000);
      const shotCorridorPath = path.join(outDirV2, 'production_viewer_corridor.png');
      await page.screenshot({ path: shotCorridorPath });
      console.log(`[+] Captured Production Viewer Corridor: ${path.basename(shotCorridorPath)}`);
    } else {
      console.log('    [-] Could not find Corridor scene card');
    }

  } catch (err) {
    console.error('[-] Error during production viewer test:', err);
  } finally {
    await browser.close();
    server.kill();
    console.log('[+] Production server stopped.');
  }
}

testProductionViewer();
