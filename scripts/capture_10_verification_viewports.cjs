/**
 * F.O.R.G.E. — Automated Capture of the 10 Mandatory Verification Viewports.
 * Uses Playwright and Microsoft Edge with full WebGL hardware acceleration to render
 * Candidate H3 inside the Three.js production viewer across all 10 mandated views.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8999;
const RECOVERY_DIR = path.resolve(__dirname, '../outputs/indoor/final_room_reconstruction_recovery');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const MESH_DIR = path.join(RECOVERY_DIR, 'meshes');
const RENDERS_DIR = path.join(RECOVERY_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/viewer_10_views.html') {
    filePath = path.join(RECOVERY_DIR, 'viewer_10_views.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/model/')) {
    filePath = path.join(MESH_DIR, urlPath.replace('/model/', ''));
  } else {
    filePath = path.join(RECOVERY_DIR, urlPath);
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

const VIEW_IDS = [
  'view1_entrance',
  'view2_full_room',
  'view3_top_down',
  'view4_front_window',
  'view5_left_wardrobe',
  'view6_right_bed',
  'view7_bed_closeup',
  'view8_desk_chair',
  'view9_wardrobe_closeup',
  'view10_ceiling_corners'
];

async function captureAllViewports() {
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

    page.on('console', msg => {
      console.log('    [PAGE LOG]:', msg.text());
    });

    try {
      console.log(`[*] Loading Three.js verification scene...`);
      await page.goto(`http://127.0.0.1:${PORT}/viewer_10_views.html`, { waitUntil: 'domcontentloaded' });

      // Wait until model is loaded
      await page.waitForFunction(() => window.RENDER_READY === true, { timeout: 60000 });
      console.log(`[+] Candidate H3 Model loaded successfully in WebGL!`);

      for (let i = 0; i < VIEW_IDS.length; i++) {
        const viewId = VIEW_IDS[i];
        console.log(`[*] Capturing View ${i + 1}/10: ${viewId}...`);

        await page.evaluate((vId) => {
          window.setViewport(vId);
        }, viewId);

        // Allow 3 frames to settle
        await page.waitForTimeout(350);

        const outPath = path.join(RENDERS_DIR, `${viewId}_3d_reconstruction.png`);
        await page.screenshot({ path: outPath });
        console.log(`    [+] Saved: ${path.basename(outPath)}`);
      }

      console.log(`\n[+] Successfully captured all 10 Mandatory Viewports in ${RENDERS_DIR}`);
    } catch (err) {
      console.error(`[-] Error during capture:`, err);
    } finally {
      await browser.close();
      server.close();
      process.exit(0);
    }
  });
}

captureAllViewports();
