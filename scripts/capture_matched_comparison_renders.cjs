const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const BASE_URL = 'http://127.0.0.1:5174';
const ARTIFACT_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');
const OUT_DIR = path.resolve('outputs/indoor/b314_geometry_investigation');

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function captureSceneViewpoints(sceneId, prefix) {
  console.log(`\n================================================================`);
  console.log(`  CAPTURING VIEWPOINTS FOR SCENE: ${sceneId} (${prefix})`);
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
      '--window-size=1200,900'
    ]
  });

  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const targetUrl = `${BASE_URL}/?scene=${sceneId}`;
  console.log('[*] Navigating to:', targetUrl);

  await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 45000 });
  await page.waitForTimeout(5000);

  const canvas = await page.$('canvas');
  if (!canvas) throw new Error('Canvas element not found in viewport');

  // Load target poses
  const posesFile = path.resolve('outputs/indoor/b314_viewpoint_comparisons/threejs_target_poses.json');
  const poses = JSON.parse(fs.readFileSync(posesFile, 'utf8'));

  // Define viewpoints: 11.4s (curtain/door), 59.6s (bed legs), 107.8s (rack)
  // Plus a Ceiling overview viewpoint
  const viewpoints = [
    { id: '11.4s', name: 'Curtain & Doorway', pose: poses.find(p => Math.abs(p.target_ts - 11.4) < 0.1) },
    { id: '59.6s', name: 'Red Bed Legs', pose: poses.find(p => Math.abs(p.target_ts - 59.6) < 0.1) },
    { id: '107.8s', name: 'Blue Furniture Rack', pose: poses.find(p => Math.abs(p.target_ts - 107.8) < 0.1) },
    {
      id: 'ceiling',
      name: 'Ceiling Gaps & Upper Boundary',
      pose: {
        position: [0.0, 0.15, 0.0],
        quaternion: [-0.60, 0.0, 0.0, 0.80], // pitched upward 74 degrees towards ceiling
        fov_y: 55.0
      }
    }
  ];

  for (const vp of viewpoints) {
    if (!vp.pose) continue;
    const vpId = vp.id;
    console.log(`\n[*] Setting camera for Viewpoint: ${vp.name} (${vpId})...`);

    // Set camera pose
    await page.evaluate(({ pos, quat, fov }) => {
      if (window.__setCameraExactPose) {
        window.__setCameraExactPose(pos, quat, fov);
      }
    }, { pos: vp.pose.position, quat: vp.pose.quaternion, fov: vp.pose.fov_y });
    await page.waitForTimeout(1000);

    // 1. Capture Textured Render
    const pTextured = path.join(OUT_DIR, `${prefix}_textured_${vpId}.png`);
    const pArtTextured = path.join(ARTIFACT_DIR, `${prefix}_textured_${vpId}.png`);
    await canvas.screenshot({ path: pTextured });
    await canvas.screenshot({ path: pArtTextured });
    console.log(`  [+] Captured textured render: ${path.basename(pTextured)}`);

    // 2. Set Untextured (Clay) Mode
    await page.evaluate(() => {
      if (window.__setClayMode) window.__setClayMode(true);
    });
    await page.waitForTimeout(600);

    // Capture Untextured (Clay) Render
    const pClay = path.join(OUT_DIR, `${prefix}_untextured_${vpId}.png`);
    const pArtClay = path.join(ARTIFACT_DIR, `${prefix}_untextured_${vpId}.png`);
    await canvas.screenshot({ path: pClay });
    await canvas.screenshot({ path: pArtClay });
    console.log(`  [+] Captured untextured clay render: ${path.basename(pClay)}`);

    // Restore Textured Material
    await page.evaluate(() => {
      if (window.__setClayMode) window.__setClayMode(false);
    });
    await page.waitForTimeout(400);
  }

  await browser.close();
  console.log(`[+] Viewpoints for ${sceneId} successfully captured!`);
}

async function main() {
  // Capture Baseline
  await captureSceneViewpoints('indoor_b314_texture_corrected', 'baseline');
  // Capture Candidate
  await captureSceneViewpoints('indoor_b314_geometry_improved', 'candidate');
  console.log('\n[+] Matched comparison capture complete!');
}

main().catch(err => {
  console.error('[!] Error capturing comparison renders:', err);
  process.exit(1);
});
