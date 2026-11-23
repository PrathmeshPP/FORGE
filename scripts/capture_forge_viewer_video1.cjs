const path = require('path');
const fs = require('fs');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const RENDERS_DIR = path.resolve(__dirname, '../outputs/indoor/vggt_tsdf_video_20261030_170234/renders');
const ARTIFACT_DIR = 'C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953';

async function run() {
  console.log('[*] Launching browser to verify live F.O.R.G.E. viewer on http://127.0.0.1:8080 ...');
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: ['--enable-webgl', '--use-gl=angle', '--use-angle=default', '--ignore-gpu-blocklist', '--no-sandbox']
  });

  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 }
  });

  page.on('console', msg => console.log('  [VIEWER ' + msg.type() + ']:', msg.text()));

  await page.goto('http://127.0.0.1:8080/', { waitUntil: 'networkidle' });

  // Select video1_vggt_tsdf_candidate_a_textured if not default
  console.log('[*] Waiting for canvas...');
  await page.waitForSelector('canvas', { timeout: 30000 });
  await page.waitForTimeout(4000);

  // Take Overview screenshot
  const p1 = path.join(RENDERS_DIR, 'forge_viewer_video1_overview.png');
  await page.screenshot({ path: p1 });
  console.log(`[+] Captured F.O.R.G.E. overview: ${p1}`);
  fs.copyFileSync(p1, path.join(ARTIFACT_DIR, 'forge_viewer_video1_overview.png'));

  // Switch to top-down camera if possible or orbit
  const canvasBox = await page.$eval('canvas', el => {
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });

  // Drag canvas to orbit
  await page.mouse.move(canvasBox.x, canvasBox.y);
  await page.mouse.down();
  await page.mouse.move(canvasBox.x + 200, canvasBox.y - 120, { steps: 15 });
  await page.mouse.up();
  await page.waitForTimeout(2000);

  const p2 = path.join(RENDERS_DIR, 'forge_viewer_video1_orbited.png');
  await page.screenshot({ path: p2 });
  console.log(`[+] Captured F.O.R.G.E. orbited: ${p2}`);
  fs.copyFileSync(p2, path.join(ARTIFACT_DIR, 'forge_viewer_video1_orbited.png'));

  await browser.close();
  console.log('[+] Viewer capture complete.');
}

run().catch(err => {
  console.error('[-] Error:', err);
  process.exit(1);
});

