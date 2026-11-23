const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 9005;
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

async function runForensicDiagnosticRenders() {
  server.listen(PORT, async () => {
    console.log(`[+] Forensic Render Server listening on http://127.0.0.1:${PORT}`);

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

    // Controlled Diagnostic Render Configurations
    const outDirV1 = path.join(BASE_DIR, 'outputs/indoor/stage2_texture_video1/diagnostic_renders');
    const outDirV2 = path.join(BASE_DIR, 'outputs/indoor/stage2_texture_video2/diagnostic_renders');
    fs.mkdirSync(outDirV1, { recursive: true });
    fs.mkdirSync(outDirV2, { recursive: true });

    // 1. Shaders for Room B 314
    const shaders = [
      { id: '1_flat_grey_flat_shading', shader: 'flat_grey_flat' },
      { id: '2_flat_grey_smooth_normals', shader: 'flat_grey_smooth' },
      { id: '3_uv_checkerboard', shader: 'uv_checkerboard' },
      { id: '4_actual_photographic_texture', shader: 'actual_texture' },
      { id: '5_normal_vector_visualization', shader: 'normal_vis' },
      { id: '6_wireframe_overlay', shader: 'wireframe_overlay' }
    ];

    const modelV1 = '/outputs/indoor/stage2_texture_video1/final_stage2_model.glb';
    const camV1 = { x: 0.0, y: 0.4, z: 2.2, tx: 0, ty: 0, tz: 0 };

    console.log('\n============================================================');
    console.log('[*] Capturing 6 Controlled Diagnostic Shaders for Room B 314');
    console.log('============================================================');

    for (const s of shaders) {
      const url = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelV1)}&shader=${s.shader}&camX=${camV1.x}&camY=${camV1.y}&camZ=${camV1.z}`;
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
      await page.waitForTimeout(1000);

      const savePath = path.join(outDirV1, `shader_${s.id}.png`);
      await page.screenshot({ path: savePath });
      console.log(`    [+] Captured: ${path.basename(savePath)}`);
    }

    // 2. Close-up Views: Furniture, Wall, Floor for Room B 314
    const closeupsV1 = [
      { name: 'furniture_closeup', camX: 0.35, camY: 0.15, camZ: 0.95, tx: 0.25, ty: 0.0, tz: 0.2 },
      { name: 'wall_closeup', camX: -0.65, camY: 0.35, camZ: 0.85, tx: -0.85, ty: 0.3, tz: 0.0 },
      { name: 'floor_closeup', camX: 0.1, camY: -0.2, camZ: 0.9, tx: 0.1, ty: -0.5, tz: 0.2 }
    ];

    console.log('\n============================================================');
    console.log('[*] Capturing Close-ups: Before (Stage 1) vs After (Stage 2)');
    console.log('============================================================');

    const modelV1_stage1 = '/outputs/indoor/texture_experiment_video1/room_photographic_model.glb';

    for (const cu of closeupsV1) {
      // Stage 1 (Before)
      const urlBefore = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelV1_stage1)}&shader=actual_texture&camX=${cu.camX}&camY=${cu.camY}&camZ=${cu.camZ}&targetX=${cu.tx}&targetY=${cu.ty}&targetZ=${cu.tz}`;
      await page.goto(urlBefore, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
      await page.waitForTimeout(1000);
      const saveBefore = path.join(outDirV1, `comparison_before_stage1_${cu.name}.png`);
      await page.screenshot({ path: saveBefore });
      console.log(`    [+] Stage 1 Before: ${path.basename(saveBefore)}`);

      // Stage 2 (After)
      const urlAfter = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelV1)}&shader=actual_texture&camX=${cu.camX}&camY=${cu.camY}&camZ=${cu.camZ}&targetX=${cu.tx}&targetY=${cu.ty}&targetZ=${cu.tz}`;
      await page.goto(urlAfter, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
      await page.waitForTimeout(1000);
      const saveAfter = path.join(outDirV1, `comparison_after_stage2_${cu.name}.png`);
      await page.screenshot({ path: saveAfter });
      console.log(`    [+] Stage 2 After:  ${path.basename(saveAfter)}`);
    }

    // 3. Diagnostic Shaders for Video 2 (Corridor)
    const modelV2 = '/outputs/indoor/stage2_texture_video2/final_stage2_model.glb';
    const camV2 = { x: 0.0, y: 0.3, z: 2.4, tx: 0, ty: 0, tz: 0 };

    console.log('\n============================================================');
    console.log('[*] Capturing 6 Controlled Diagnostic Shaders for Corridor Walkthrough');
    console.log('============================================================');

    for (const s of shaders) {
      const url = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(modelV2)}&shader=${s.shader}&camX=${camV2.x}&camY=${camV2.y}&camZ=${camV2.z}`;
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.RENDER_STATUS && window.RENDER_STATUS.ready === true, { timeout: 45000 });
      await page.waitForTimeout(1000);

      const savePath = path.join(outDirV2, `shader_${s.id}.png`);
      await page.screenshot({ path: savePath });
      console.log(`    [+] Captured: ${path.basename(savePath)}`);
    }

    await browser.close();
    server.close();
    console.log('\n[+] All Controlled Diagnostic Renders Successfully Generated!');
  });
}

runForensicDiagnosticRenders();

