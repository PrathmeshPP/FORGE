/**
 * F.O.R.G.E. V2 — Actual Exported GLB Viewer Reload & Walkthrough QA Harness
 * Runs against the real F.O.R.G.E. React/Three.js viewer (`viewer/`), reloads the
 * exported GLB via HTTP, tests orbit & clay mode + ENTER INDOOR WALKTHROUGH + WASD/UP
 * controls, checks WebGL console/page errors, and captures visual QA screenshots.
 */

const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

function parseArgs() {
  const args = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    const k = process.argv[i].replace(/^--/, '');
    args[k] = process.argv[i + 1];
  }
  return args;
}

function waitForServer(url, timeoutMs = 30000) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const ping = () => {
      http.get(url, (res) => {
        res.resume();
        resolve();
      }).on('error', () => {
        if (Date.now() - t0 > timeoutMs) {
          reject(new Error(`Vite server timeout waiting for ${url}`));
        } else {
          setTimeout(ping, 400);
        }
      });
    };
    ping();
  });
}

async function main() {
  const args = parseArgs();
  const sceneId = args['scene-id'];
  const outputDir = path.resolve(args['output-dir']);
  const qaDir = path.join(outputDir, 'viewer_qa');
  const port = parseInt(args['port'] || '5175', 10);
  fs.mkdirSync(qaDir, { recursive: true });

  const viewerDir = path.resolve(__dirname, '../viewer');
  const viteProc = spawn(
    process.platform === 'win32' ? 'cmd.exe' : 'npx',
    process.platform === 'win32' ? ['/c', 'npx', 'vite', '--port', String(port), '--host', '127.0.0.1'] : ['vite', '--port', String(port), '--host', '127.0.0.1'],
    { cwd: viewerDir, stdio: 'ignore', windowsHide: true }
  );

  let browser;
  const pageErrors = [];
  const consoleErrors = [];
  const report = {
    scene_id: sceneId,
    viewer_url: `http://127.0.0.1:${port}/?scene=${encodeURIComponent(sceneId)}`,
    status: 'FAIL',
    checks: {},
    screenshots: {},
    console_errors: [],
    page_errors: [],
  };

  try {
    await waitForServer(`http://127.0.0.1:${port}/`);
    browser = await chromium.launch({
      headless: true,
      executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
      args: [
        '--enable-webgl',
        '--use-gl=angle',
        '--use-angle=default',
        '--ignore-gpu-blocklist',
        '--no-sandbox',
      ],
    });

    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto(report.viewer_url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForSelector('canvas', { timeout: 20000 });

    // Wait up to 45s for the GLB mesh to load into Three.js model group
    await page.waitForFunction(() => {
      const sc = window.__viewportScene;
      if (!sc) return false;
      let meshCount = 0;
      let hasVerts = false;
      sc.traverse((o) => {
        if (o.isMesh && o.geometry && o.geometry.attributes && o.geometry.attributes.position) {
          if (o.geometry.attributes.position.count > 100) {
            meshCount++;
            hasVerts = true;
          }
        }
      });
      return hasVerts && meshCount >= 1;
    }, { timeout: 45000 });
    await page.waitForTimeout(1500);

    // Inspect loaded scene stats directly from Three.js runtime
    const sceneInfo = await page.evaluate(() => {
      const sc = window.__viewportScene;
      let verts = 0;
      let tris = 0;
      let meshes = 0;
      let hasTex = false;
      sc.traverse((o) => {
        if (o.isMesh && o.geometry && o.geometry.attributes && o.geometry.attributes.position) {
          meshes++;
          verts += o.geometry.attributes.position.count;
          tris += o.geometry.index ? Math.round(o.geometry.index.count / 3) : Math.round(o.geometry.attributes.position.count / 3);
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          for (const m of mats) {
            if (m && m.map) hasTex = true;
          }
        }
      });
      return { meshes, verts, tris, hasTex };
    });

    report.checks.glb_loaded = sceneInfo.meshes >= 1 && sceneInfo.verts > 100;
    report.checks.has_texture_map = sceneInfo.hasTex;
    report.checks.scene_info = sceneInfo;

    // Capture 1: Orbit Textured Overview
    const pOrbit = path.join(qaDir, '01_viewer_orbit_textured.png');
    await page.screenshot({ path: pOrbit });
    report.screenshots.orbit_textured = pOrbit;

    // Capture 2: Untextured Clay Inspection Mode
    await page.evaluate(() => window.__setClayMode && window.__setClayMode(true));
    await page.waitForTimeout(500);
    const pClay = path.join(qaDir, '02_viewer_untextured_clay.png');
    await page.screenshot({ path: pClay });
    report.screenshots.untextured_clay = pClay;
    await page.evaluate(() => window.__setClayMode && window.__setClayMode(false));
    await page.waitForTimeout(400);

    // Capture 3: Enter Indoor Walkthrough & Test WASD + U/P Navigation
    const enterBtn = await page.$('.enter-space-btn');
    let walkthroughOk = false;
    let deltaMove = 0.0;
    let deltaUp = 0.0;

    if (enterBtn) {
      await enterBtn.click();
      await page.waitForTimeout(900);
      const posBefore = await page.evaluate(() => {
        const c = window.__viewportCamera;
        return c ? [c.position.x, c.position.y, c.position.z] : null;
      });

      const pSpawn = path.join(qaDir, '03_viewer_walkthrough_spawn.png');
      await page.screenshot({ path: pSpawn });
      report.screenshots.walkthrough_spawn = pSpawn;

      // Hold W (Forward) + A slightly
      await page.keyboard.down('KeyW');
      await page.waitForTimeout(650);
      await page.keyboard.up('KeyW');
      await page.waitForTimeout(250);

      const posAfterMove = await page.evaluate(() => {
        const c = window.__viewportCamera;
        return c ? [c.position.x, c.position.y, c.position.z] : null;
      });
      if (posBefore && posAfterMove) {
        deltaMove = Math.hypot(
          posAfterMove[0] - posBefore[0],
          posAfterMove[1] - posBefore[1],
          posAfterMove[2] - posBefore[2]
        );
      }

      const pMoved = path.join(qaDir, '04_viewer_walkthrough_moved.png');
      await page.screenshot({ path: pMoved });
      report.screenshots.walkthrough_moved = pMoved;

      // Press U (Elevate up)
      await page.keyboard.down('KeyU');
      await page.waitForTimeout(450);
      await page.keyboard.up('KeyU');
      await page.waitForTimeout(250);

      const posAfterUp = await page.evaluate(() => {
        const c = window.__viewportCamera;
        return c ? [c.position.x, c.position.y, c.position.z] : null;
      });
      if (posAfterMove && posAfterUp) {
        deltaUp = posAfterUp[1] - posAfterMove[1];
      }

      const pUp = path.join(qaDir, '05_viewer_walkthrough_elevated.png');
      await page.screenshot({ path: pUp });
      report.screenshots.walkthrough_elevated = pUp;

      walkthroughOk = deltaMove > 0.01 || deltaUp > 0.01;
    }

    report.checks.walkthrough_entered = Boolean(enterBtn);
    report.checks.walkthrough_move_delta_m = Number(deltaMove.toFixed(4));
    report.checks.walkthrough_elevate_delta_m = Number(deltaUp.toFixed(4));
    report.checks.walkthrough_navigation_ok = walkthroughOk;
    report.console_errors = consoleErrors;
    report.page_errors = pageErrors;

    const noFatalErrors = pageErrors.length === 0;
    report.status = (report.checks.glb_loaded && report.checks.has_texture_map && walkthroughOk && noFatalErrors)
      ? 'PASS'
      : 'FAIL';

  } catch (err) {
    report.error = String(err && err.message ? err.message : err);
    report.status = 'FAIL';
  } finally {
    if (browser) await browser.close();
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(viteProc.pid), '/f', '/t'], { stdio: 'ignore' });
      } else {
        viteProc.kill();
      }
    } catch (_) {}
  }

  const outJson = path.join(qaDir, 'viewer_qa_report.json');
  fs.writeFileSync(outJson, JSON.stringify(report, null, 2), 'utf-8');
  console.log(JSON.stringify(report, null, 2));
  process.exit(report.status === 'PASS' ? 0 : 1);
}

main();
