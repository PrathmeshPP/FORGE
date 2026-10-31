const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8997;
const EXP_DIR = path.resolve(__dirname, '../outputs/indoor/geometry_quality_improvement_video1');
const PREV_DIR = path.resolve(__dirname, '../outputs/indoor/photographic_texture_quality_recovery_video1');
const THREE_DIR = path.resolve(__dirname, '../viewer/node_modules/three');
const RENDERS_DIR = path.join(EXP_DIR, 'renders');

fs.mkdirSync(RENDERS_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  let filePath = '';
  const urlPath = req.url.split('?')[0];

  if (urlPath === '/' || urlPath === '/test_viewer_geometry_comparison.html') {
    filePath = path.join(EXP_DIR, 'test_viewer_geometry_comparison.html');
  } else if (urlPath.startsWith('/three/')) {
    filePath = path.join(THREE_DIR, urlPath.replace('/three/', ''));
  } else if (urlPath.startsWith('/model/')) {
    filePath = path.join(EXP_DIR, urlPath.replace('/model/', ''));
  } else if (urlPath.startsWith('/model_b/')) {
    filePath = path.join(PREV_DIR, urlPath.replace('/model_b/', ''));
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

async function runViewerCaptures() {
  server.listen(PORT, async () => {
    console.log(`[+] Geometry comparison server running at http://127.0.0.1:${PORT}`);

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
      const text = msg.text();
      if (!text.startsWith('unhandled') && text.length < 200) {
        console.log('    [BROWSER ' + msg.type() + ']:', text);
      }
    });

    const models = ['candidate_b', 'candidate_d', 'candidate_e'];
    const views = [
      'v01_complete_room',
      'v02_top_down_layout',
      'v03_front_elevation',
      'v04_side_elevation',
      'v05_isometric',
      'v06_bed_closeup',
      'v07_desk_computer',
      'v08_wardrobe',
      'v09_wall_window',
      'v10_floor_ceiling'
    ];

    const summary = {};

    for (const model of models) {
      console.log(`\n======================================================`);
      console.log(`[*] Capturing Model: ${model.toUpperCase()}`);
      console.log(`======================================================`);
      summary[model] = {};

      for (const view of views) {
        const url = `http://127.0.0.1:${PORT}/?model=${model}&view=${view}`;
        console.log(`  -> Loading View: ${view}...`);

        await page.goto(url, { waitUntil: 'domcontentloaded' });
        try {
          await page.waitForFunction(
            () => window.VERIFICATION_STATE && window.VERIFICATION_STATE.loaded === true,
            { timeout: 60000 }
          );
        } catch (err) {
          console.log(`     [!] Timeout waiting for model to load, taking screenshot anyway`);
        }
        await page.waitForTimeout(2000);

        const outPath = path.join(RENDERS_DIR, `render_${model}_${view}.png`);
        await page.screenshot({ path: outPath });
        console.log(`     [+] Saved: ${path.basename(outPath)}`);
        summary[model][view] = outPath;
      }
    }

    // Save summary
    const summaryPath = path.join(RENDERS_DIR, 'capture_summary.json');
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
    console.log(`\n[+] All captures saved. Summary: ${summaryPath}`);

    await browser.close();
    server.close();
    process.exit(0);
  });
}

runViewerCaptures();

