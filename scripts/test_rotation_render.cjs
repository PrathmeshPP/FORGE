const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('C:/Users/25beevdt025/AppData/Local/OpenAI/Codex/runtimes/cua_node/78fb63cda1a9e2b2/bin/node_modules/playwright');

const PORT = 8996;
const ROOT_DIR = path.resolve(__dirname, '..');

const server = http.createServer((req, res) => {
  const urlPath = req.url.split('?')[0];
  if (urlPath === '/' || urlPath === '/test.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
      <!DOCTYPE html>
      <html><head>
      <script type="importmap">
      {
        "imports": {
          "three": "/viewer/node_modules/three/build/three.module.js",
          "three/addons/": "/viewer/node_modules/three/examples/jsm/"
        }
      }
      </script>
      </head><body style='margin:0;overflow:hidden;background:#181a1f;'>
      <script type='module'>
        import * as THREE from 'three';
        import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x181a1f);
        const camera = new THREE.PerspectiveCamera(65, window.innerWidth/window.innerHeight, 0.05, 50);
        
        // With mesh.rotation.x = Math.PI, Three.js standard camera works:
        // camera.up is (0, 1, 0)
        camera.position.set(0, 1.42, -2.0);
        camera.lookAt(0, 1.30, 1.0);

        const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        document.body.appendChild(renderer.domElement);

        const ambLight = new THREE.AmbientLight(0xffffff, 1.2);
        scene.add(ambLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
        dirLight.position.set(0, 3, 0);
        scene.add(dirLight);

        const loader = new PLYLoader();
        loader.load('/outputs/indoor/texture_experiment_video_20261101_142811/meshes/poisson_mesh.ply', (geom) => {
          if (geom.attributes.position && geom.attributes.position.array instanceof Float64Array) {
            geom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(geom.attributes.position.array), 3));
          }
          if (geom.attributes.normal && geom.attributes.normal.array instanceof Float64Array) {
            geom.setAttribute('normal', new THREE.Float32BufferAttribute(new Float32Array(geom.attributes.normal.array), 3));
          } else if (!geom.attributes.normal) {
            geom.computeVertexNormals();
          }
          const hasColors = geom.hasAttribute('color');
          const mat = new THREE.MeshStandardMaterial({
            vertexColors: hasColors,
            roughness: 0.85,
            metalness: 0.05,
            side: THREE.DoubleSide
          });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.rotation.x = Math.PI; // Coordinate orientation adaptation: Y up (+Y), Z forward (+Z)
          scene.add(mesh);

          renderer.render(scene, camera);
          window.__RENDERED__ = true;
          window.__STATS__ = {
            verts: geom.attributes.position.count,
            faces: geom.index ? geom.index.count / 3 : 0,
            hasColors: hasColors,
            hasNormals: geom.hasAttribute('normal')
          };
        });
      </script>
      </body></html>
    `);
    return;
  }
  
  const filePath = path.join(ROOT_DIR, urlPath);
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    const mime = ext === '.ply' ? 'application/octet-stream' : ext === '.js' ? 'application/javascript' : 'text/plain';
    res.writeHead(200, { 'Content-Type': mime });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(PORT, async () => {
  console.log('Server running on', PORT);
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: ['--enable-webgl', '--use-gl=angle', '--use-angle=default', '--ignore-gpu-blocklist', '--no-sandbox']
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', err => console.log('ERR:', err.message));

  await page.goto('http://127.0.0.1:' + PORT + '/test.html');
  await page.waitForFunction(() => window.__RENDERED__ === true, { timeout: 30000 });
  const stats = await page.evaluate(() => window.__STATS__);
  console.log('Rendered stats:', stats);
  await page.screenshot({ path: 'scratch_rotation_view.png' });
  console.log('Saved scratch_rotation_view.png');
  await browser.close();
  server.close();
  process.exit(0);
});

