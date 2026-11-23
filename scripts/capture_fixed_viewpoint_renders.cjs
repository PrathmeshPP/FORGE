const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const BASE_URL = 'http://127.0.0.1:5174';
const SCENE_ID = process.argv[2] || 'indoor_b314_generalized_regression';
const PREFIX = process.argv[3] || 'render_baseline';
const TARGET_URL = `${BASE_URL}/?scene=${SCENE_ID}`;
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');
const OUT_DIR = path.resolve('outputs/indoor/b314_viewpoint_comparisons');

async function main() {
  console.log('[*] Capturing Fixed Visual Evaluation Viewpoints at 11.4s, 59.6s, 107.8s');
  const posesFile = path.join(OUT_DIR, 'threejs_target_poses.json');
  if (!fs.existsSync(posesFile)) {
    throw new Error(`Poses file not found: ${posesFile}`);
  }
  const poses = JSON.parse(fs.readFileSync(posesFile, 'utf8'));

  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: [
      '--enable-webgl',
      '--use-gl=angle',
      '--use-angle=default',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--window-size=1200,900'
    ]
  });

  // Native smartphone resolution: 480 x 854
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });

  console.log('[*] Navigating to:', TARGET_URL);
  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 45000 });
  await page.waitForTimeout(5000);

  // Wait for canvas
  const canvas = await page.$('canvas');
  if (!canvas) throw new Error('Canvas element not found in viewport');

  // Verify __setCameraExactPose is ready
  const isHelperReady = await page.evaluate(() => typeof window.__setCameraExactPose === 'function');
  console.log('[*] Viewport helper ready:', isHelperReady);

  for (const item of poses) {
    const ts = item.target_ts;
    console.log(`\n[*] Setting camera for Timestamp: ${ts}s...`);

    await page.evaluate(({ pos, quat, fov }) => {
      if (window.__setCameraExactPose) {
        window.__setCameraExactPose(pos, quat, fov);
      }
    }, { pos: item.position, quat: item.quaternion, fov: item.fov_y });

    await page.waitForTimeout(1000);

    const outRenderPath = path.join(OUT_DIR, `${PREFIX}_${ts.toFixed(1)}s.png`);
    const artRenderPath = path.join(ARTIFACT_DIR, `${PREFIX}_${ts.toFixed(1)}s.png`);

    // Screenshot the canvas element directly
    await canvas.screenshot({ path: outRenderPath });
    await canvas.screenshot({ path: artRenderPath });
    console.log(`[+] Captured render for ${ts}s: ${path.basename(outRenderPath)}`);
  }

  await browser.close();
  console.log('[+] Viewpoint renders successfully captured!');
}

main().catch(err => {
  console.error('[!] Error capturing viewpoint renders:', err);
  process.exit(1);
});
