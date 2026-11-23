const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 9015;
const BASE_DIR = path.resolve(__dirname, '..');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');

const server = http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  let filePath = '';

  if (urlPath === '/' || urlPath === '/forensic_render_page.html') {
    filePath = path.join(__dirname, 'forensic_render_page.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else {
    filePath = path.join(BASE_DIR, urlPath.replace(/^\//, ''));
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.json': 'application/json',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.glb': 'model/gltf-binary',
      '.obj': 'text/plain',
      '.mtl': 'text/plain',
      '.ply': 'application/octet-stream'
    };
    const contentType = mimeTypes[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found: ' + urlPath);
  }
});

async function runDensityComparisonRenders() {
  server.listen(PORT, async () => {
    console.log(`[+] Density Render Server listening on http://127.0.0.1:${PORT}`);

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

    page.on('console', msg => console.log('    [BROWSER ' + msg.type() + ']:', msg.text()));
    page.on('pageerror', err => console.error('    [BROWSER ERROR]:', err.message));

    const expRoot = path.join(BASE_DIR, 'outputs/indoor/frame_density_experiment');
    const outComparisons = path.join(expRoot, 'comparisons');
    fs.mkdirSync(outComparisons, { recursive: true });

    const configs = [
      'baseline_1_5fps',
      'exp_a_2fps',
      'exp_b_3fps',
      'exp_c_4fps',
      'exp_d_dual_pool'
    ];

    // Views for Video 1 (Room B 314)
    const viewsV1 = [
      { name: 'overview', cam: { x: 0.0, y: 0.4, z: 2.2, tx: 0, ty: 0, tz: 0 } },
      { name: 'furniture_closeup', cam: { x: 0.35, y: 0.15, z: 0.95, tx: 0.25, ty: 0.0, tz: 0.2 } },
      { name: 'wall_closeup', cam: { x: -0.65, y: 0.35, z: 0.85, tx: -0.85, ty: 0.3, tz: 0.0 } },
      { name: 'floor_closeup', cam: { x: 0.1, y: -0.2, z: 0.9, tx: 0.1, ty: -0.5, tz: 0.2 } }
    ];

    console.log('\n============================================================');
    console.log('[*] Capturing Density Comparison Views for Video 1 (Room B 314)');
    console.log('============================================================');

    for (const cfg of configs) {
      const modelRel = `/outputs/indoor/frame_density_experiment/video_1_room/${cfg}/model_density_result.glb`;
      const modelAbs = path.join(expRoot, 'video_1_room', cfg, 'model_density_result.glb');
      if (!fs.existsSync(modelAbs)) {
        console.log(`    [-] Skipping missing model: ${cfg}`);
        continue;
      }

      for (const v of viewsV1) {
        const url = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelRel)}&shader=actual_texture&camX=${v.cam.x}&camY=${v.cam.y}&camZ=${v.cam.z}&targetX=${v.cam.tx}&targetY=${v.cam.ty}&targetZ=${v.cam.tz}`;
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
        await page.waitForTimeout(1000);

        const savePath = path.join(outComparisons, `v1_${cfg}_${v.name}.png`);
        await page.screenshot({ path: savePath });
        console.log(`    [+] Captured: ${path.basename(savePath)}`);
      }
    }

    // Views for Video 2 (Corridor)
    const viewsV2 = [
      { name: 'overview', cam: { x: 0.0, y: 0.3, z: 2.4, tx: 0, ty: 0, tz: 0 } },
      { name: 'door_closeup', cam: { x: -0.4, y: 0.2, z: 1.2, tx: -0.6, ty: 0.2, tz: 0.5 } },
      { name: 'wall_closeup', cam: { x: 0.5, y: 0.3, z: 1.1, tx: 0.7, ty: 0.3, tz: 0.5 } },
      { name: 'floor_closeup', cam: { x: 0.0, y: -0.3, z: 1.2, tx: 0.0, ty: -0.8, tz: 0.3 } }
    ];

    console.log('\n============================================================');
    console.log('[*] Capturing Density Comparison Views for Video 2 (Corridor)');
    console.log('============================================================');

    for (const cfg of configs) {
      const modelRel = `/outputs/indoor/frame_density_experiment/video_2_corridor/${cfg}/model_density_result.glb`;
      const modelAbs = path.join(expRoot, 'video_2_corridor', cfg, 'model_density_result.glb');
      if (!fs.existsSync(modelAbs)) {
        console.log(`    [-] Skipping missing model: ${cfg}`);
        continue;
      }

      for (const v of viewsV2) {
        const url = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelRel)}&shader=actual_texture&camX=${v.cam.x}&camY=${v.cam.y}&camZ=${v.cam.z}&targetX=${v.cam.tx}&targetY=${v.cam.ty}&targetZ=${v.cam.tz}`;
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
        await page.waitForTimeout(1000);

        const savePath = path.join(outComparisons, `v2_${cfg}_${v.name}.png`);
        await page.screenshot({ path: savePath });
        console.log(`    [+] Captured: ${path.basename(savePath)}`);
      }
    }

    await browser.close();
    server.close();
    console.log('\n[+] All Density Comparison Renders Completed Successfully!');
  });
}

runDensityComparisonRenders();

