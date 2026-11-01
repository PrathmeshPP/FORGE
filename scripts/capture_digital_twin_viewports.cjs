/**
 * F.O.R.G.E. — Automated Capture of 13 Required Validation Viewports
 * Model: outputs/indoor/digital_twin_reconstruction_video_20261101_142811/meshes/room_digital_twin_photographic.glb
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 9026;
const EXP_DIR = path.resolve(__dirname, '../outputs/indoor/digital_twin_reconstruction_video_20261101_142811');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const MESH_DIR = path.join(EXP_DIR, 'meshes');
const RENDERS_DIR = path.join(EXP_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/viewer_digital_twin.html') {
    filePath = path.join(EXP_DIR, 'viewer_digital_twin.html');
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

const VIEWS = [
  {
    id: "01_entrance_view",
    title: "1. Complete Room from Entrance",
    eye: [0.0, 1.42, 2.7],
    target: [0.0, 1.42, -2.5],
    fov: 65
  },
  {
    id: "02_full_overview",
    title: "2. Full-Room Overview (All 4 Walls)",
    eye: [0.0, 2.60, 2.8],
    target: [0.0, 0.60, -0.8],
    fov: 82
  },
  {
    id: "03_bed_closeup",
    title: "3. Close-up of Bed 1 (Left)",
    eye: [0.65, 1.20, -0.05],
    target: [1.65, 0.35, -0.05],
    fov: 60
  },
  {
    id: "04_wardrobe_closeup",
    title: "4. Close-up of Wardrobe 1 (Left)",
    eye: [0.85, 1.45, 1.75],
    target: [2.05, 1.40, 1.75],
    fov: 60
  },
  {
    id: "05_desk_closeup",
    title: "5. Close-up of Study Desk 1 (Left)",
    eye: [0.75, 1.15, -2.05],
    target: [1.90, 0.70, -2.05],
    fov: 60
  },
  {
    id: "06_chair_closeup",
    title: "6. Close-up of Wooden Chair",
    eye: [0.75, 0.90, 2.32],
    target: [1.40, 0.50, 2.32],
    fov: 55
  },
  {
    id: "07_topdown_plan",
    title: "7. Top-Down Floor Plan View",
    eye: [0.0, 2.75, 0.0],
    target: [0.0, 0.0, 0.0],
    up: [0.0, 0.0, -1.0],
    fov: 90
  },
  {
    id: "08_front_wall_door",
    title: "8. Front Wall with Entrance Door",
    eye: [0.0, 1.42, 1.2],
    target: [0.0, 1.42, 3.25],
    fov: 65
  },
  {
    id: "09_back_wall_window",
    title: "9. Back Wall with Curtains & Window",
    eye: [0.0, 1.42, -1.2],
    target: [0.0, 1.42, -3.25],
    fov: 65
  },
  {
    id: "10_left_wall_furniture",
    title: "10. Left Wall with Wardrobe, Bed & Desk",
    eye: [-0.4, 1.42, 0.0],
    target: [2.65, 1.42, 0.0],
    fov: 75
  },
  {
    id: "11_right_wall_furniture",
    title: "11. Right Wall with Bed 2, Desk 2 & Wardrobes",
    eye: [0.4, 1.42, 0.0],
    target: [-2.55, 1.42, 0.0],
    fov: 75
  },
  {
    id: "12_ceiling_floor_alignment",
    title: "12. Floor and Ceiling Alignment",
    eye: [0.0, 1.42, 2.2],
    target: [0.0, 1.42, -2.5],
    fov: 85
  },
  {
    id: "13_walkthrough_perspective",
    title: "13. Realistic Walkthrough Perspective",
    eye: [0.15, 1.42, 0.5],
    target: [0.15, 1.42, -2.5],
    fov: 65
  }
];

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
        '--no-sandbox',
        '--disable-setuid-sandbox'
      ]
    });

    const page = await browser.newPage({
      viewport: { width: 1280, height: 720 },
      deviceScaleFactor: 1
    });

    console.log('[*] Navigating to Three.js Digital Twin viewer...');
    await page.goto(`http://127.0.0.1:${PORT}/viewer_digital_twin.html`, {
      waitUntil: 'networkidle',
      timeout: 45000
    });

    console.log('[*] Waiting for GLTF model to load in WebGL...');
    await page.waitForFunction(() => window.RENDER_READY === true, { timeout: 45000 });
    console.log('[+] GLTF Model loaded successfully in WebGL!');

    for (let i = 0; i < VIEWS.length; i++) {
      const v = VIEWS[i];
      console.log(`[*] Rendering view ${i + 1}/${VIEWS.length}: ${v.title}...`);

      await page.evaluate((viewConfig) => {
        window.setCameraView(viewConfig);
      }, v);

      await new Promise(r => setTimeout(r, 450));

      const outPath = path.join(RENDERS_DIR, `view_${v.id}.png`);
      const canvasHandle = await page.$('canvas');
      if (canvasHandle) {
        await canvasHandle.screenshot({ path: outPath });
      } else {
        await page.screenshot({ path: outPath });
      }
      console.log(`  [+] Saved: view_${v.id}.png`);
    }

    console.log('\n[+] All 13 validation viewports successfully rendered and captured!');
    await browser.close();
    server.close();
    process.exit(0);
  });
}

runCaptures().catch(err => {
  console.error('Fatal error during capture:', err);
  server.close();
  process.exit(1);
});
