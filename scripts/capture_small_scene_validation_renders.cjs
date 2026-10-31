/**
 * F.O.R.G.E. — Automated Capture of Small-Scene Validation Renders
 * Renders both Photographic and Checkerboard models in Three.js across:
 * - Camera 0 View
 * - Camera 1 View
 * - Camera 2 View
 * - Oblique 3D Perspective View
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 9001;
const EXP_DIR = path.resolve(__dirname, '../outputs/indoor/controlled_three_frame_validation');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const MESH_DIR = path.join(EXP_DIR, 'step3_small_scene_reconstruction/meshes');
const RENDERS_DIR = path.join(EXP_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/viewer_small_scene.html') {
    filePath = path.join(EXP_DIR, 'viewer_small_scene.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/model/')) {
    filePath = path.join(MESH_DIR, urlPath.replace('/model/', ''));
  } else {
    filePath = path.join(EXP_DIR, urlPath);
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.json': 'application/json',
      '.png': 'image/png',
      '.glb': 'model/gltf-binary',
      '.obj': 'text/plain',
      '.mtl': 'text/plain'
    };
    const contentType = mimeTypes[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found: ' + urlPath);
  }
});

const VIEWS = ['cam0', 'cam1', 'cam2', 'iso_oblique'];

async function runCaptures() {
  server.listen(PORT, async () => {
    console.log(`[+] Three.js verification server running at http://127.0.0.1:${PORT}`);

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
      viewport: { width: 1280, height: 720 }
    });

    try {
      console.log(`[*] Loading Three.js verification scene...`);
      await page.goto(`http://127.0.0.1:${PORT}/viewer_small_scene.html`, { waitUntil: 'domcontentloaded' });

      // 1. Capture Photographic Model Views
      console.log(`\n======================================================`);
      console.log(`[*] Capturing Model A: Photographic Textured Model...`);
      console.log(`======================================================`);
      await page.waitForFunction(() => window.RENDER_READY === true, { timeout: 45000 });
      console.log(`[+] Photographic model ready in WebGL.`);

      for (const viewId of VIEWS) {
        await page.evaluate((vId) => window.setCameraView(vId), viewId);
        await page.waitForTimeout(300);
        const outPath = path.join(RENDERS_DIR, `photo_render_${viewId}.png`);
        await page.screenshot({ path: outPath });
        console.log(`    [+] Saved: ${path.basename(outPath)}`);
      }

      // 2. Load and Capture Checkerboard Model Views
      console.log(`\n======================================================`);
      console.log(`[*] Capturing Model B: UV Checkerboard Diagnostic Model...`);
      console.log(`======================================================`);
      await page.evaluate(() => window.loadModel('/model/small_scene_desk_window_checkerboard.glb'));
      await page.waitForFunction(() => window.RENDER_READY === true, { timeout: 45000 });
      console.log(`[+] Checkerboard model ready in WebGL.`);

      for (const viewId of VIEWS) {
        await page.evaluate((vId) => window.setCameraView(vId), viewId);
        await page.waitForTimeout(300);
        const outPath = path.join(RENDERS_DIR, `checker_render_${viewId}.png`);
        await page.screenshot({ path: outPath });
        console.log(`    [+] Saved: ${path.basename(outPath)}`);
      }

      console.log(`\n[+] Successfully captured all Photographic & Checkerboard validation renders in ${RENDERS_DIR}`);
    } catch (err) {
      console.error(`[-] Error during capture:`, err);
    } finally {
      await browser.close();
      server.close();
      process.exit(0);
    }
  });
}

runCaptures();
