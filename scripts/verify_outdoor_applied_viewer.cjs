const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8998;
const WORKSPACE_ROOT = path.resolve(__dirname, '..');
const EXPERIMENT_DIR = path.join(WORKSPACE_ROOT, 'outputs/indoor/outdoor_pipeline_applied_video1');
const THREE_DIR = path.join(WORKSPACE_ROOT, 'viewer/node_modules/three');
const RENDERS_DIR = path.join(EXPERIMENT_DIR, 'previews');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

// HTTP Server
const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/test_viewer.html') {
    filePath = path.join(EXPERIMENT_DIR, 'test_viewer.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/outputs/')) {
    filePath = path.join(WORKSPACE_ROOT, urlPath.slice(1));
  } else {
    filePath = path.join(WORKSPACE_ROOT, urlPath.slice(1));
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

async function runViewerVerification() {
  server.listen(PORT, async () => {
    console.log(`[+] Verification server running on http://127.0.0.1:${PORT}`);

    let browser;
    try {
      browser = await chromium.launch({
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

      console.log(`[*] Loading reused outdoor architecture model in Three.js harness...`);
      await page.goto(`http://127.0.0.1:${PORT}/test_viewer.html`, { waitUntil: 'domcontentloaded' });

      // Wait for model to load (timeout 90s for 136MB model)
      console.log(`[*] Waiting for 136MB GLB model to parse and buffer into WebGL...`);
      await page.waitForFunction(() => window.VERIFICATION_RESULTS && window.VERIFICATION_RESULTS.loaded === true, {
        timeout: 90000
      });

      const results = await page.evaluate(() => window.VERIFICATION_RESULTS);
      console.log(`[+] Model verified successfully in Three.js:`);
      console.log(`    Loaded:           ${results.loaded}`);
      console.log(`    Texture Embedded: ${results.hasTexture}`);
      console.log(`    Texture Dims:     ${results.textureDimensions}`);
      console.log(`    Vertices:         ${results.vertexCount.toLocaleString()}`);
      console.log(`    Triangles:        ${results.triangleCount.toLocaleString()}`);
      console.log(`    Material Type:    ${results.materialType}`);
      console.log(`    Bounding Box:     ${JSON.stringify(results.boundingBox)}`);

      const views = [
        { id: 'full_room', filename: 'viewer_outdoor_applied_overview.png', label: 'Full Room View' },
        { id: 'top_plan', filename: 'viewer_outdoor_applied_top_plan.png', label: 'Top-Down Architectural Plan' },
        { id: 'front_view', filename: 'viewer_outdoor_applied_front.png', label: 'Front Elevation' },
        { id: 'side_view', filename: 'viewer_outdoor_applied_side.png', label: 'Side Elevation' },
        { id: 'isometric', filename: 'viewer_outdoor_applied_isometric.png', label: 'Isometric Overview' },
        { id: 'furniture_close', filename: 'viewer_outdoor_applied_furniture_close.png', label: 'Furniture Close-Up (Bed/Desk)' }
      ];

      const screenshotMap = {};
      for (const v of views) {
        console.log(`[*] Capturing perspective: ${v.label}...`);
        await page.evaluate((viewId) => window.setCameraView(viewId), v.id);
        await page.waitForTimeout(1500);
        const outP = path.join(RENDERS_DIR, v.filename);
        await page.screenshot({ path: outP });
        console.log(`    [+] Saved ${v.filename}`);
        screenshotMap[v.id] = v.filename;
      }

      // Save summary JSON
      const summaryOut = {
        timestamp: new Date().toISOString(),
        format: 'GLB',
        results: results,
        screenshots: screenshotMap
      };
      fs.writeFileSync(
        path.join(EXPERIMENT_DIR, 'diagnostics', 'viewer_verification_summary.json'),
        JSON.stringify(summaryOut, null, 2)
      );
      console.log(`[+] Viewer verification complete and summary saved.`);

      await browser.close();
      server.close();
      process.exit(0);
    } catch (err) {
      console.error('[-] Error during viewer verification:', err);
      if (browser) await browser.close();
      server.close();
      process.exit(1);
    }
  });
}

runViewerVerification();

