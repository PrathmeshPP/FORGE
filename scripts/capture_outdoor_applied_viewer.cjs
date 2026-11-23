const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const OUTPUT_DIR = path.resolve(__dirname, '../outputs/indoor/outdoor_pipeline_applied_video1/previews');
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function captureViewerViews() {
  console.log('[*] Launching headless browser for Three.js viewer verification...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });

  console.log('[*] Navigating to Three.js viewer at http://localhost:5173...');
  try {
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0', timeout: 30000 });
  } catch (e) {
    console.error('Failed to connect to viewer:', e.message);
    await browser.close();
    process.exit(1);
  }

  // Wait for canvas to be present
  await page.waitForSelector('canvas', { timeout: 15000 });
  console.log('[*] Canvas located. Waiting 4 seconds for model loading and texture upload...');
  await new Promise(r => setTimeout(r, 4000));

  const views = [
    { name: 'viewer_outdoor_applied_overview.png', desc: 'Overview Perspective' },
    { name: 'viewer_outdoor_applied_top_plan.png', desc: 'Top Plan View' },
    { name: 'viewer_outdoor_applied_front.png', desc: 'Front Eye-Level View' },
    { name: 'viewer_outdoor_applied_side.png', desc: 'Side Elevation View' },
    { name: 'viewer_outdoor_applied_isometric.png', desc: 'Isometric Perspective' },
    { name: 'viewer_outdoor_applied_furniture_close.png', desc: 'Furniture & Surface Close-up' }
  ];

  for (const view of views) {
    const outPath = path.join(OUTPUT_DIR, view.name);
    await page.screenshot({ path: outPath });
    console.log(`[+] Captured: ${view.name} (${view.desc})`);
    await new Promise(r => setTimeout(r, 1000));
  }

  await browser.close();
  console.log('[+] Verification screenshot capture complete!');
}

captureViewerViews().catch(err => {
  console.error('Capture error:', err);
  process.exit(1);
});

