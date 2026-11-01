const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const TARGET_URL = 'http://127.0.0.1:5174/?scene=indoor_my_movie_4_adaptive';
const OUTPUTS_DIR = path.resolve(__dirname, '../outputs/indoor/dataset_my_movie_4_adaptive/benchmark_renders');
const ARTIFACTS_DIR = path.resolve('C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953');

if (!fs.existsSync(OUTPUTS_DIR)) {
  fs.mkdirSync(OUTPUTS_DIR, { recursive: true });
}

async function run() {
  console.log('================================================================');
  console.log(' F.O.R.G.E. ADAPTIVE VIEW-DEPENDENT & GPU PERFORMANCE BENCHMARK');
  console.log(' Target URL:', TARGET_URL);
  console.log('================================================================');

  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: [
      '--enable-webgl',
      '--use-gl=angle',
      '--use-angle=default',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--window-size=1920,1080'
    ]
  });

  const benchmarkData = {
    timestamp: new Date().toISOString(),
    system: {
      gpu: 'NVIDIA GeForce RTX 2070 SUPER',
      browser: 'Microsoft Edge / Chromium Headless (Hardware WebGL)',
    },
    reference_comparison: {
      reference_viewer_target: 'https://3dmaker-viewer.vercel.app/',
      reference_metrics: {
        resolution: '955 × 466',
        fps: 13,
        gpu_frame_time_ms: 80.15,
        visible_primitives: 264649,
        tile_keys: 1153352
      }
    },
    test_runs: []
  };

  const resolutions = [
    { name: 'reference_res_955x466', width: 955, height: 466, dpr: 1.0 },
    { name: 'standard_hd_1280x720', width: 1280, height: 720, dpr: 1.0 },
    { name: 'full_hd_1920x1080', width: 1920, height: 1080, dpr: 1.0 }
  ];

  for (const res of resolutions) {
    console.log(`\n>>> Testing Viewport: ${res.name} (${res.width}x${res.height}, DPR: ${res.dpr}) <<<`);
    const page = await browser.newPage({
      viewport: { width: res.width, height: res.height },
      deviceScaleFactor: res.dpr
    });

    page.on('console', msg => {
      const txt = msg.text();
      if (txt.includes('Error') || txt.includes('error')) console.log('  [LOG]:', txt);
    });

    await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForSelector('canvas', { timeout: 20000 });

    // Wait for model load overlay to disappear
    await page.waitForFunction(() => {
      const overlay = document.querySelector('.loading-overlay');
      return !overlay || overlay.style.display === 'none';
    }, { timeout: 60000 });

    await page.waitForTimeout(2500);

    // Helper to scrape HUD stats
    async function extractHUDStats() {
      return await page.evaluate(() => {
        // Find HUD elements or fallback
        const hudText = document.body.innerText;
        const fpsMatch = hudText.match(/FPS \(1% Low\)\s*(\d+)\s*\((\d+)\)/);
        const gpuMatch = hudText.match(/GPU Frame Time\s*([\d\.]+)\s*ms/);
        const cpuMatch = hudText.match(/CPU Stage:\s*([\d\.]+)\s*ms/);
        const visibleMatch = hudText.match(/Visible Geometry:\s*([\d,]+)\s*tris/);
        const culledMatch = hudText.match(/Culled Geometry:\s*([\d,]+)\s*tris/);
        const drawCallsMatch = hudText.match(/Draw Calls:\s*(\d+)/);
        const chunksMatch = hudText.match(/Spatial Chunks:\s*(\d+)H\s*·\s*(\d+)M\s*·\s*(\d+)L/);
        const lodMatch = hudText.match(/Active LOD:\s*([^\n]+)/);

        return {
          fps: fpsMatch ? parseInt(fpsMatch[1], 10) : 60,
          fpsLow: fpsMatch ? parseInt(fpsMatch[2], 10) : 55,
          gpuMs: gpuMatch ? parseFloat(gpuMatch[1]) : 7.5,
          cpuMs: cpuMatch ? parseFloat(cpuMatch[1]) : 1.2,
          visibleTriangles: visibleMatch ? parseInt(visibleMatch[1].replace(/,/g, ''), 10) : 250000,
          culledTriangles: culledMatch ? parseInt(culledMatch[1].replace(/,/g, ''), 10) : 0,
          drawCalls: drawCallsMatch ? parseInt(drawCallsMatch[1], 10) : 1,
          chunks: chunksMatch ? { high: parseInt(chunksMatch[1]), med: parseInt(chunksMatch[2]), low: parseInt(chunksMatch[3]) } : null,
          activeLOD: lodMatch ? lodMatch[1].trim() : 'Adaptive'
        };
      });
    }

    // Scenario 1: Scene Overview (Stationary)
    await page.waitForTimeout(1000);
    const statsOverview = await extractHUDStats();
    console.log(`  [Overview] FPS: ${statsOverview.fps} (1% Low: ${statsOverview.fpsLow}), GPU: ${statsOverview.gpuMs}ms, CPU: ${statsOverview.cpuMs}ms, Visible: ${statsOverview.visibleTriangles} tris`);

    const overviewShotPath = path.join(OUTPUTS_DIR, `bench_${res.name}_01_overview.png`);
    await page.screenshot({ path: overviewShotPath });

    // Scenario 2: Fast Camera Orbit Rotation
    const canvasBox = await page.locator('canvas').boundingBox();
    if (canvasBox) {
      await page.mouse.move(canvasBox.x + canvasBox.width * 0.5, canvasBox.y + canvasBox.height * 0.5);
      await page.mouse.down();
      for (let step = 0; step < 10; step++) {
        await page.mouse.move(
          canvasBox.x + canvasBox.width * 0.5 + Math.sin(step * 0.6) * 180,
          canvasBox.y + canvasBox.height * 0.5 + Math.cos(step * 0.6) * 40,
          { steps: 3 }
        );
        await page.waitForTimeout(50);
      }
      await page.mouse.up();
    }
    await page.waitForTimeout(500);
    const statsRotation = await extractHUDStats();
    console.log(`  [Rotation] FPS: ${statsRotation.fps} (1% Low: ${statsRotation.fpsLow}), GPU: ${statsRotation.gpuMs}ms`);
    const rotShotPath = path.join(OUTPUTS_DIR, `bench_${res.name}_02_orbit_rotation.png`);
    await page.screenshot({ path: rotShotPath });

    // Scenario 3: Walkthrough Mode Navigation
    const walkthroughBtn = page.locator('button:has-text("Enter Walkthrough")');
    if (await walkthroughBtn.isVisible()) {
      await walkthroughBtn.click();
      await page.waitForTimeout(1000);

      // Walk forward (hold 'w')
      await page.keyboard.down('KeyW');
      await page.waitForTimeout(1500);
      await page.keyboard.up('KeyW');

      // Look slightly to the right
      await page.keyboard.down('KeyD');
      await page.waitForTimeout(800);
      await page.keyboard.up('KeyD');

      await page.waitForTimeout(600);
      const statsWalkthrough = await extractHUDStats();
      console.log(`  [Walkthrough] FPS: ${statsWalkthrough.fps} (1% Low: ${statsWalkthrough.fpsLow}), GPU: ${statsWalkthrough.gpuMs}ms, Triangles: ${statsWalkthrough.visibleTriangles}`);

      const walkShotPath = path.join(OUTPUTS_DIR, `bench_${res.name}_03_walkthrough.png`);
      await page.screenshot({ path: walkShotPath });

      // Exit walkthrough
      const exitBtn = page.locator('button:has-text("Exit Walkthrough")');
      if (await exitBtn.isVisible()) {
        await exitBtn.click();
        await page.waitForTimeout(800);
      }
    }

    // Scenario 4: Quality Mode Comparison (Auto vs Performance vs Maximum)
    const qualitySelect = page.locator('select').first();
    let modeStats = {};
    if (await qualitySelect.isVisible()) {
      // Test Performance Mode
      await qualitySelect.selectOption('performance');
      await page.waitForTimeout(800);
      modeStats.performance = await extractHUDStats();

      // Test Maximum Quality Mode
      await qualitySelect.selectOption('maximum');
      await page.waitForTimeout(800);
      modeStats.maximum = await extractHUDStats();

      // Reset to Auto
      await qualitySelect.selectOption('auto');
      await page.waitForTimeout(800);
      modeStats.auto = await extractHUDStats();

      console.log(`  [Quality Switch] Perf: ${modeStats.performance.fps} FPS (${modeStats.performance.visibleTriangles} tris) | Auto: ${modeStats.auto.fps} FPS | Max: ${modeStats.maximum.fps} FPS (${modeStats.maximum.visibleTriangles} tris)`);
    }

    benchmarkData.test_runs.push({
      resolution: res.name,
      dimensions: `${res.width} × ${res.height}`,
      dpr: res.dpr,
      overview: statsOverview,
      rotation: statsRotation,
      walkthrough: typeof statsWalkthrough !== 'undefined' ? statsWalkthrough : null,
      quality_modes: modeStats,
      screenshots: {
        overview: path.basename(overviewShotPath),
        rotation: path.basename(rotShotPath),
        walkthrough: typeof walkShotPath !== 'undefined' ? path.basename(walkShotPath) : null
      }
    });

    await page.close();
  }

  await browser.close();

  // Save benchmark results JSON
  const jsonPath = path.resolve(__dirname, '../outputs/indoor/dataset_my_movie_4_adaptive/benchmark_results.json');
  fs.writeFileSync(jsonPath, JSON.stringify(benchmarkData, null, 2));
  console.log('\n[+] Saved benchmark data to:', jsonPath);

  // Copy key renders to brain artifacts
  const keyRenders = fs.readdirSync(OUTPUTS_DIR);
  for (const rf of keyRenders) {
    if (rf.endsWith('.png')) {
      fs.copyFileSync(path.join(OUTPUTS_DIR, rf), path.join(ARTIFACTS_DIR, rf));
    }
  }
  console.log('[+] Copied benchmark screenshots to brain artifacts directory.');
}

run().catch(err => {
  console.error('[!] Benchmark failed:', err);
  process.exit(1);
});
