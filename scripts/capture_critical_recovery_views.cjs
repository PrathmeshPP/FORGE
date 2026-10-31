// Automated 10-View Screen Capture for Candidates F & G
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8996;
const EXP_DIR = path.resolve(__dirname, '../outputs/indoor/critical_reconstruction_recovery_video1');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const RENDERS_DIR = path.join(EXP_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

// Minimal Three.js HTML Server for Candidate F and G
const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/viewer.html') {
    filePath = path.join(EXP_DIR, 'viewer_critical_recovery.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/model/')) {
    filePath = path.join(EXP_DIR, urlPath.replace('/model/', ''));
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
      '.glb': 'model/gltf-binary'
    };
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found: ' + urlPath);
  }
});

async function runCaptures() {
  server.listen(PORT, async () => {
    console.log(`[+] Server running at http://127.0.0.1:${PORT}`);

    const browser = await chromium.launch({
      headless: true,
      executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
      args: ['--enable-webgl', '--use-gl=angle', '--use-angle=default', '--ignore-gpu-blocklist', '--no-sandbox']
    });

    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

    const models = ['candidate_f', 'candidate_g'];
    const views = [
      'v01_room_overview',
      'v02_top_down_layout',
      'v03_room_entrance',
      'v04_rear_window_view',
      'v05_isometric_enclosure',
      'v06_bed_closeup',
      'v07_desk_chair_closeup',
      'v08_wardrobe_closeup',
      'v09_left_wall_window',
      'v10_floor_ceiling'
    ];

    for (const model of models) {
      console.log(`[*] Capturing ${model.toUpperCase()}...`);
      for (const view of views) {
        const url = `http://127.0.0.1:${PORT}/viewer.html?model=${model}&view=${view}`;
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        try {
          await page.waitForFunction(() => window.LOADED === true, { timeout: 30000 });
        } catch (e) {
          console.log(`    Timeout on ${model}/${view}`);
        }
        await page.waitForTimeout(1000);
        const out = path.join(RENDERS_DIR, `render_${model}_${view}.png`);
        await page.screenshot({ path: out });
        console.log(`    Saved: ${path.basename(out)}`);
      }
    }

    await browser.close();
    server.close();
    process.exit(0);
  });
}

runCaptures();
