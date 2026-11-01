const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8997;
const EXPERIMENT_DIR = path.resolve(__dirname, '../outputs/indoor/texture_experiment_video_20261101_142811');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const RENDERS_DIR = path.join(EXPERIMENT_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

// Simple HTTP server
const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/test_viewer.html') {
    filePath = path.join(EXPERIMENT_DIR, 'test_viewer.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/model/')) {
    filePath = path.join(EXPERIMENT_DIR, urlPath.replace('/model/', ''));
  } else {
    filePath = path.join(EXPERIMENT_DIR, urlPath);
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
    console.log(`[+] Verification server running at http://127.0.0.1:${PORT}`);

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
      const text = msg.text();
      if (!text.startsWith('unhandled') && text.length < 200) {
        console.log('    [BROWSER ' + msg.type() + ']:', text);
      }
    });
    page.on('pageerror', err => console.error('    [BROWSER ERROR]:', err.message));

    const formats = ['glb', 'obj'];
    const summary = {};

    for (const fmt of formats) {
      console.log(`\n[*] Testing Three.js rendering for format: ${fmt.toUpperCase()}...`);
      const url = `http://127.0.0.1:${PORT}/?format=${fmt}`;
      await page.goto(url, { waitUntil: 'domcontentloaded' });

      await page.waitForFunction(() => window.VERIFICATION_RESULTS && window.VERIFICATION_RESULTS.loaded === true, { timeout: 45000 });
      await page.waitForTimeout(2000);

      const results = await page.evaluate(() => window.VERIFICATION_RESULTS);
      console.log(`    Format: ${results.format}`);
      console.log(`    Texture Present: ${results.hasTexture} (Dimensions: ${results.textureDimensions})`);
      console.log(`    UV Coordinates: ${results.hasUV}`);
      console.log(`    Geometry: ${results.vertexCount.toLocaleString()} vertices, ${results.triangleCount.toLocaleString()} triangles`);
      console.log(`    Material Type: ${results.materialType}`);

      const screenshotPath = path.join(RENDERS_DIR, `viewer_verified_${fmt}.png`);
      await page.screenshot({ path: screenshotPath });
      console.log(`    [+] Saved viewer screenshot: ${path.basename(screenshotPath)}`);

      summary[fmt] = {
        ...results,
        screenshot: screenshotPath
      };
    }

    fs.writeFileSync(
      path.join(EXPERIMENT_DIR, 'renders', 'viewer_verification_summary.json'),
      JSON.stringify(summary, null, 2)
    );
    console.log(`\n[+] Verification complete! Saved summary to renders/viewer_verification_summary.json`);

    await browser.close();
    server.close();
    process.exit(0);
  });
}

runViewerVerification().catch(err => {
  console.error('Fatal verification error:', err);
  process.exit(1);
});
