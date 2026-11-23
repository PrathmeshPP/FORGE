const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 9001;
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const BASE_DIR = path.resolve(__dirname, '..');

// Simple multi-directory HTTP server
const server = http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  let filePath = '';

  if (urlPath === '/' || urlPath === '/test_viewer.html') {
    filePath = path.join(__dirname, 'stage2_viewer.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else {
    // Relative to workspace
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

async function runStage2ViewerVerification() {
  server.listen(PORT, async () => {
    console.log(`[+] Stage 2 Verification Server listening on http://127.0.0.1:${PORT}`);

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

    const targets = [
      {
        dataset: 'video_1',
        name: 'Student Hostel Room B 314',
        modelUrl: '/outputs/indoor/stage2_texture_video1/final_stage2_model.glb',
        camPos: [0.0, 0.4, 2.2],
        camTarget: [0, 0, 0],
        savePng: path.resolve(BASE_DIR, 'outputs/indoor/stage2_texture_video1/renders/H_viewer_verified_final_glb.png'),
        reportJson: path.resolve(BASE_DIR, 'outputs/indoor/stage2_texture_video1/viewer_verification_summary.json')
      },
      {
        dataset: 'video_2',
        name: 'Hostel Corridor Walkthrough',
        modelUrl: '/outputs/indoor/stage2_texture_video2/final_stage2_model.glb',
        camPos: [0.0, 0.3, 2.4],
        camTarget: [0, 0, 0],
        savePng: path.resolve(BASE_DIR, 'outputs/indoor/stage2_texture_video2/renders/H_viewer_verified_final_glb.png'),
        reportJson: path.resolve(BASE_DIR, 'outputs/indoor/stage2_texture_video2/viewer_verification_summary.json')
      }
    ];

    for (const t of targets) {
      console.log(`\n============================================================`);
      console.log(`[*] Verifying Three.js WebGL: ${t.name} (${t.dataset})`);
      console.log(`[*] Model: ${t.modelUrl}`);

      const pageUrl = `http://127.0.0.1:${PORT}/?model=${encodeURIComponent(t.modelUrl)}&camX=${t.camPos[0]}&camY=${t.camPos[1]}&camZ=${t.camPos[2]}`;
      await page.goto(pageUrl, { waitUntil: 'domcontentloaded' });

      // Wait for completion
      await page.waitForFunction(() => window.VERIFICATION_RESULTS && window.VERIFICATION_RESULTS.loaded === true, { timeout: 60000 });
      await page.waitForTimeout(2500);

      const results = await page.evaluate(() => window.VERIFICATION_RESULTS);
      console.log(`    Format:            ${results.format}`);
      console.log(`    Texture Bound:     ${results.hasTexture}`);
      console.log(`    Texture Dimensions: ${results.textureDimensions}`);
      console.log(`    Has UVs:           ${results.hasUV}`);
      console.log(`    Has Smooth Normals: ${results.hasNormals}`);
      console.log(`    Vertices:          ${results.vertexCount.toLocaleString()}`);
      console.log(`    Triangles:         ${results.triangleCount.toLocaleString()}`);
      console.log(`    Material Type:     ${results.materialType}`);

      await page.screenshot({ path: t.savePng });
      console.log(`    [+] Saved WebGL render: ${path.basename(t.savePng)}`);

      fs.writeFileSync(t.reportJson, JSON.stringify(results, null, 2));
    }

    await browser.close();
    server.close();
    console.log('\n[+] Stage 2 WebGL Viewer Verification Completed Successfully!');
  });
}

runStage2ViewerVerification();

