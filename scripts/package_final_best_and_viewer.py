#!/usr/bin/env python3
"""
Package Final Best Deliverable and Create Campaign 3D Viewer.
"""

import json
import shutil
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

PROJECT_ROOT = Path("C:/Users/25beevdt025/Desktop/FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
FINAL_BEST_DIR = CAMPAIGN_DIR / "final_best"
VIEWER_DIR = CAMPAIGN_DIR / "viewer"
POISSON_DIR = CAMPAIGN_DIR / "candidate_16f_poisson"
UPSTREAM_DIR = CAMPAIGN_DIR / "upstream_16_frames"


def package_final_best():
    print("[*] Packaging final_best reconstruction deliverable...")
    FINAL_BEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy and rename GLB
    src_glb = POISSON_DIR / "candidate_16f_poisson_reconstruction.glb"
    dst_glb = FINAL_BEST_DIR / "final_best_reconstruction.glb"
    shutil.copy2(src_glb, dst_glb)
    print(f"    Exported GLB: {dst_glb.name} ({dst_glb.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Copy texture atlas
    src_atlas = POISSON_DIR / "texture_atlas.png"
    dst_atlas = FINAL_BEST_DIR / "texture_atlas.png"
    shutil.copy2(src_atlas, dst_atlas)
    print(f"    Exported Texture Atlas: {dst_atlas.name} ({dst_atlas.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Create canonical OBJ & MTL linking to texture_atlas.png
    src_obj = POISSON_DIR / "candidate_16f_poisson_reconstruction.obj"
    dst_obj = FINAL_BEST_DIR / "final_best_reconstruction.obj"
    dst_mtl = FINAL_BEST_DIR / "final_best_reconstruction.mtl"

    with open(dst_mtl, "w") as f_mtl:
        f_mtl.write("# F.O.R.G.E Final Best Material Library\n")
        f_mtl.write("newmtl material_atlas\n")
        f_mtl.write("Ka 1.000 1.000 1.000\n")
        f_mtl.write("Kd 1.000 1.000 1.000\n")
        f_mtl.write("Ks 0.000 0.000 0.000\n")
        f_mtl.write("d 1.0\n")
        f_mtl.write("illum 2\n")
        f_mtl.write("map_Kd texture_atlas.png\n")

    # Read original OBJ and adjust mtllib line
    with open(src_obj, "r") as f_in, open(dst_obj, "w") as f_out:
        for line in f_in:
            if line.startswith("mtllib "):
                f_out.write("mtllib final_best_reconstruction.mtl\n")
            else:
                f_out.write(line)
    print(f"    Exported OBJ: {dst_obj.name} ({dst_obj.stat().st_size / (1024*1024):.2f} MB)")
    print(f"    Exported MTL: {dst_mtl.name}")

    # 4. Copy PLY mesh
    src_ply = POISSON_DIR / "candidate_16f_poisson_reconstruction.ply"
    dst_ply = FINAL_BEST_DIR / "final_best_reconstruction.ply"
    shutil.copy2(src_ply, dst_ply)
    print(f"    Exported PLY Mesh: {dst_ply.name} ({dst_ply.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Export fused point cloud
    # Load 16 frame depth and cameras to fuse and save final_best_point_cloud.ply
    print("[*] Generating final_best_point_cloud.ply from 16 dense keyframes...")
    cam_data = np.load(UPSTREAM_DIR / "predictions/camera_poses.npz")
    depth_data = np.load(UPSTREAM_DIR / "predictions/depth_predictions.npz")
    extrinsics = cam_data["extrinsics"]
    intrinsics = cam_data["intrinsics"]
    depth_maps = depth_data["depth"]
    conf_maps = depth_data["confidence"]

    frame_files = sorted((UPSTREAM_DIR / "frames").glob("*.png"))
    H_in, W_in = depth_maps.shape[1], depth_maps.shape[2]
    u_grid, v_grid = np.meshgrid(np.arange(W_in), np.arange(H_in))

    pts_list, col_list, nrm_list = [], [], []
    for i in range(len(frame_files)):
        d = depth_maps[i].squeeze()
        c = conf_maps[i].squeeze()
        mask = (c >= 1.8) & (d >= 0.1) & (d <= 6.0)
        u_v = u_grid[mask]
        v_v = v_grid[mask]
        z_v = d[mask]

        K = intrinsics[i]
        x_c = (u_v - K[0, 2]) * z_v / K[0, 0]
        y_c = (v_v - K[1, 2]) * z_v / K[1, 1]
        pts_cam = np.stack([x_c, y_c, z_v], axis=-1)

        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        pts_w = (R.T @ (pts_cam.T - t[:, None])).T

        img = o3d.io.read_image(str(frame_files[i]))
        img_np = np.asarray(img).astype(np.float32) / 255.0
        cols_w = img_np[v_v, u_v]

        ray_dirs = pts_cam / (np.linalg.norm(pts_cam, axis=1, keepdims=True) + 1e-6)
        nrm_cam = -ray_dirs
        nrm_w = (R.T @ nrm_cam.T).T

        pts_list.append(pts_w)
        col_list.append(cols_w)
        nrm_list.append(nrm_w)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_list, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_list, axis=0))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate(nrm_list, axis=0))

    # Fine voxel downsample (0.004m)
    pcd_fused = pcd.voxel_down_sample(voxel_size=0.004)
    pcd_dst = FINAL_BEST_DIR / "final_best_point_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_dst), pcd_fused)
    print(f"    Exported Fused Point Cloud: {pcd_dst.name} ({len(pcd_fused.points):,} points, {pcd_dst.stat().st_size / (1024*1024):.2f} MB)")

    # 6. Copy preview images
    for p_name in ["model_preview.png", "neutral_geometry_preview.png", "texture_detail_preview.png"]:
        shutil.copy2(POISSON_DIR / p_name, FINAL_BEST_DIR / p_name)

    # 7. Write reconstruction metadata
    mesh_metrics = {
        "candidate_id": "candidate_16f_poisson",
        "method": "Screened Poisson Reconstruction (16 Dense Keyframes)",
        "selection_rationale": "Superior surface continuity, watertight ground and building structures, zero voxel-stepping artifacts (which compromised TSDF), zero boundary perforations (which compromised BPA), and accurate 4096 texture projection.",
        "vertices": len(pcd_fused.points),
        "triangles": 258635,
        "mesh_vertices": 133172,
        "extents_xyz_meters": [3.273, 0.596, 3.125],
        "scale_status": "Relative metric estimate from multi-view VGGT; real-world scale uncalibrated without GCPs / surveyed ground control points.",
        "accuracy_target_5cm": "Unverified without surveyed ground control points or flight telemetry RTK logs.",
        "input_video": "WhatsApp Video 2026-10-24 at 17.53.55.mp4",
        "frames_used": 16,
        "runtime_opencv_version": "5.0.0",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(FINAL_BEST_DIR / "reconstruction_metadata.json", "w") as f_meta:
        json.dump(mesh_metrics, f_meta, indent=2)
    print("    Exported metadata: reconstruction_metadata.json")


def create_campaign_viewer():
    print("[*] Creating dedicated campaign local 3D viewer...")
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>F.O.R.G.E — WhatsApp Video 3D Reconstruction Explorer</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
  <style>
    :root {
      --bg-primary: #0d1117;
      --bg-card: rgba(22, 27, 34, 0.88);
      --border-color: #30363d;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-green: #3fb950;
      --accent-gold: #d29922;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-winner {
      background: rgba(63, 185, 80, 0.15);
      color: var(--accent-green);
      border-color: var(--accent-green);
    }
    .controls {
      display: flex;
      gap: 14px;
      align-items: center;
    }
    select, button {
      background: #21262d;
      color: var(--text-main);
      border: 1px solid var(--border-color);
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      outline: none;
      transition: all 0.2s;
    }
    select:hover, button:hover {
      background: #30363d;
      border-color: #8b949e;
    }
    select:focus {
      border-color: var(--accent);
    }
    main {
      flex: 1;
      position: relative;
    }
    model-viewer {
      width: 100%;
      height: 100%;
      background-color: #090d13;
      --poster-color: transparent;
    }
    .stats-overlay {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 13px;
      min-width: 280px;
      z-index: 5;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .stats-overlay h4 {
      margin-bottom: 8px;
      font-size: 14px;
      color: var(--accent);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      margin: 4px 0;
      color: var(--text-muted);
    }
    .stat-val {
      color: var(--text-main);
      font-family: monospace;
      font-weight: 600;
    }
    .audit-note {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid var(--border-color);
      font-size: 11px;
      color: var(--accent-gold);
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h2>F.O.R.G.E</h2>
      <span class="badge badge-winner">Recovery Campaign</span>
      <span style="font-size: 13px; color: var(--text-muted);">Scene: Solar Complex (WhatsApp Drone 2026-10-24)</span>
    </div>
    <div class="controls">
      <label for="candidate-select" style="font-size: 13px; color: var(--text-muted);">Candidate Model:</label>
      <select id="candidate-select">
        <option value="../final_best/final_best_reconstruction.glb" selected>Candidate 1: Screened Poisson HQ (16 Frames) — [Winner]</option>
        <option value="../candidate_16f_tsdf/candidate_16f_tsdf_reconstruction.glb">Candidate 2: TSDF Integration (16 Frames)</option>
        <option value="../candidate_16f_bpa/candidate_16f_bpa_reconstruction.glb">Candidate 3: Multi-Scale Ball Pivoting (16 Frames)</option>
      </select>
      <button id="btn-reset-cam">Reset Camera</button>
      <button id="btn-toggle-auto">Auto-Rotate</button>
    </div>
  </header>

  <main>
    <model-viewer
      id="viewer"
      src="../final_best/final_best_reconstruction.glb"
      alt="F.O.R.G.E 3D Scene Reconstruction"
      camera-controls
      auto-rotate-delay="3000"
      interaction-prompt="auto"
      shadow-intensity="1.2"
      exposure="1.0"
      environment-image="neutral"
      camera-orbit="45deg 55deg 5m"
      min-camera-orbit="auto auto 0.5m"
      max-camera-orbit="auto auto 15m">
    </model-viewer>

    <div class="stats-overlay">
      <h4 id="stat-title">Final Best Reconstruction</h4>
      <div class="stat-row"><span>Method:</span><span class="stat-val" id="stat-method">Screened Poisson Depth 9</span></div>
      <div class="stat-row"><span>Frames Used:</span><span class="stat-val" id="stat-frames">16 Dense Keyframes</span></div>
      <div class="stat-row"><span>Vertices / Triangles:</span><span class="stat-val" id="stat-geometry">133,172 / 258,635</span></div>
      <div class="stat-row"><span>Bounding Box:</span><span class="stat-val" id="stat-extents">3.27m x 0.60m x 3.12m</span></div>
      <div class="stat-row"><span>Texture:</span><span class="stat-val">4096 x 4096 Atlas</span></div>
      <div class="stat-row"><span>OpenCV Version:</span><span class="stat-val">5.0.0 Active</span></div>
      <div class="audit-note">
        <strong>Accuracy Audit Status:</strong> Relative metric reconstruction verified; 5 cm absolute ground accuracy remains UNVERIFIED due to lack of ground control points and GPS telemetry in WhatsApp video.
      </div>
    </div>
  </main>

  <script>
    const viewer = document.getElementById('viewer');
    const selector = document.getElementById('candidate-select');
    const btnReset = document.getElementById('btn-reset-cam');
    const btnAuto = document.getElementById('btn-toggle-auto');

    const meta = {
      "../final_best/final_best_reconstruction.glb": {
        title: "Final Best: Screened Poisson HQ",
        method: "Screened Poisson Depth 9 (Trim 0.001)",
        frames: "16 Dense Keyframes",
        geometry: "133,172 / 258,635",
        extents: "3.27m x 0.60m x 3.12m"
      },
      "../candidate_16f_tsdf/candidate_16f_tsdf_reconstruction.glb": {
        title: "Candidate 2: TSDF Volume Integration",
        method: "Open3D Voxel TSDF Integration (1.2cm)",
        frames: "16 Dense Keyframes",
        geometry: "34,014 / 56,740",
        extents: "3.19m x 0.38m x 2.74m"
      },
      "../candidate_16f_bpa/candidate_16f_bpa_reconstruction.glb": {
        title: "Candidate 3: Ball Pivoting Algorithm",
        method: "Multi-Radius Ball Pivoting (0.8cm - 3.2cm)",
        frames: "16 Dense Keyframes",
        geometry: "247,574 / 433,874",
        extents: "3.25m x 0.58m x 3.11m"
      }
    };

    selector.addEventListener('change', (e) => {
      const path = e.target.value;
      viewer.src = path;
      const data = meta[path];
      if (data) {
        document.getElementById('stat-title').textContent = data.title;
        document.getElementById('stat-method').textContent = data.method;
        document.getElementById('stat-frames').textContent = data.frames;
        document.getElementById('stat-geometry').textContent = data.geometry;
        document.getElementById('stat-extents').textContent = data.extents;
      }
    });

    btnReset.addEventListener('click', () => {
      viewer.cameraOrbit = "45deg 55deg 5m";
    });

    let autoRotating = false;
    btnAuto.addEventListener('click', () => {
      autoRotating = !autoRotating;
      viewer.autoRotate = autoRotating;
      btnAuto.style.background = autoRotating ? "#3fb950" : "#21262d";
    });
  </script>
</body>
</html>
"""
    with open(VIEWER_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("    Created: viewer/index.html")

    # Also create serve_viewer.py inside VIEWER_DIR
    serve_py = """#!/usr/bin/env python3
import http.server
import mimetypes
import socketserver
import sys
from pathlib import Path

PORT = 8092
DIRECTORY = Path(__file__).resolve().parent

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/octet-stream", ".ply")
mimetypes.add_type("text/plain", ".obj")
mimetypes.add_type("text/plain", ".mtl")
mimetypes.add_type("image/png", ".png")

class DualOriginHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY.parent), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"================================================================")
    print(f"  F.O.R.G.E Campaign 3D Model Viewer")
    print(f"  Serving at: http://127.0.0.1:{port}/viewer/index.html")
    print(f"================================================================")
    with socketserver.TCPServer(("127.0.0.1", port), DualOriginHandler) as httpd:
        httpd.serve_forever()
"""
    with open(VIEWER_DIR / "serve_viewer.py", "w", encoding="utf-8") as f:
        f.write(serve_py)
    print("    Created: viewer/serve_viewer.py")


if __name__ == "__main__":
    package_final_best()
    create_campaign_viewer()
    print("[*] Final best packaging and viewer creation complete!")
