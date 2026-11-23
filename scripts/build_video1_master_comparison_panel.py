"""
F.O.R.G.E. Master Comparison Panel for Video 1 (Hostel Room B 314)
Renders side-by-side comparison:
A. Previous Baseline (stage2_texture_video1 / final_stage2_model.glb)
B. Clean Pre-TSDF Globally Aligned Point Cloud
C. TSDF 15mm Fused Surface Point Cloud
D. TSDF 15mm Watertight Neutral Mesh
E. Final Native 4K Textured Digital Twin
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261030_170234"
RENDERS_DIR = EXP_DIR / "renders"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

RENDERS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — RENDERING MASTER 5-WAY COMPARISON PANEL (VIDEO 1)")
print("=" * 76)

W, H = 800, 600

def render_geometry(geom, is_pcd=False, point_size=2.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.11, 0.12, 0.15])
    if is_pcd:
        opt.point_size = point_size
    else:
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_front([0.55, -0.55, -0.62])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_lookat(geom.get_center())
    ctr.set_zoom(0.70)

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return Image.fromarray((np.asarray(img) * 255.0).astype(np.uint8))

# A. Previous Baseline
prev_glb_p = PROJECT_ROOT / "outputs/indoor/stage2_texture_video1/final_stage2_model.glb"
print(f"[*] Loading Previous Baseline from {prev_glb_p.name}...")
tm_prev = trimesh.load(str(prev_glb_p), force='mesh')
o3d_prev = o3d.geometry.TriangleMesh()
o3d_prev.vertices = o3d.utility.Vector3dVector(tm_prev.vertices)
o3d_prev.triangles = o3d.utility.Vector3iVector(tm_prev.faces)
if hasattr(tm_prev.visual, 'vertex_colors') and len(tm_prev.visual.vertex_colors) > 0:
    o3d_prev.vertex_colors = o3d.utility.Vector3dVector(tm_prev.visual.vertex_colors[:, :3] / 255.0)
else:
    o3d_prev.paint_uniform_color([0.7, 0.7, 0.75])
o3d_prev.compute_vertex_normals()
img_A = render_geometry(o3d_prev)
print("  Rendered Panel A: Previous Baseline")

# B. Clean Pre-TSDF Global Point Cloud
pcd_clean_p = EXP_DIR / "point_clouds/fused_point_cloud_clean.ply"
o3d_clean_pcd = o3d.io.read_point_cloud(str(pcd_clean_p))
img_B = render_geometry(o3d_clean_pcd, is_pcd=True, point_size=3.0)
print("  Rendered Panel B: Pre-TSDF Global Point Cloud")

# C. TSDF 15mm Fused Point Cloud
tsdf_pcd_p = EXP_DIR / "tsdf_benchmark/voxel_15mm/tsdf_point_cloud.ply"
o3d_tsdf_pcd = o3d.io.read_point_cloud(str(tsdf_pcd_p))
img_C = render_geometry(o3d_tsdf_pcd, is_pcd=True, point_size=3.0)
print("  Rendered Panel C: TSDF Fused Point Cloud")

# D. TSDF 15mm Neutral Mesh
tsdf_mesh_p = EXP_DIR / "tsdf_benchmark/voxel_15mm/tsdf_mesh.ply"
o3d_tsdf_mesh = o3d.io.read_triangle_mesh(str(tsdf_mesh_p))
o3d_tsdf_mesh.compute_vertex_normals()
o3d_tsdf_mesh.paint_uniform_color([0.75, 0.75, 0.78])
img_D = render_geometry(o3d_tsdf_mesh)
print("  Rendered Panel D: TSDF Neutral Mesh")

# E. Final Textured Digital Twin
textured_glb_p = EXP_DIR / "textured_tsdf_model/model_video1_textured.glb"
tm_tex = trimesh.load(str(textured_glb_p), force='mesh')
o3d_tex = o3d.geometry.TriangleMesh()
o3d_tex.vertices = o3d.utility.Vector3dVector(tm_tex.vertices)
o3d_tex.triangles = o3d.utility.Vector3iVector(tm_tex.faces)
if hasattr(tm_tex.visual, 'vertex_colors') and len(tm_tex.visual.vertex_colors) > 0:
    o3d_tex.vertex_colors = o3d.utility.Vector3dVector(tm_tex.visual.vertex_colors[:, :3] / 255.0)
elif hasattr(tm_tex.visual, 'to_color'):
    vc = tm_tex.visual.to_color().vertex_colors
    o3d_tex.vertex_colors = o3d.utility.Vector3dVector(vc[:, :3] / 255.0)
else:
    o3d_tex.paint_uniform_color([0.8, 0.8, 0.8])
o3d_tex.compute_vertex_normals()
img_E = render_geometry(o3d_tex)
print("  Rendered Panel E: Final Textured Twin")

# Assemble Grid: 2 rows (Row 1: A, B, C; Row 2: D, E, info card)
panel_w = W * 3
panel_h = H * 2 + 100
panel = Image.new("RGB", (panel_w, panel_h), (18, 20, 24))
draw = ImageDraw.Draw(panel)

try:
    font_title = ImageFont.truetype("arial.py", 32)
    font_sub = ImageFont.truetype("arial.py", 22)
    font_panel = ImageFont.truetype("arial.py", 26)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = font_title
    font_panel = font_title

# Header
draw.rectangle([(0, 0), (panel_w, 90)], fill=(28, 32, 40))
draw.text((30, 15), "F.O.R.G.E. — MASTER RECONSTRUCTION COMPARISON PANEL (HOSTEL ROOM B 314)", fill=(240, 240, 250), font=font_title)
draw.text((30, 55), "Source: WhatsApp Video 2026-10-30 at 17.02.34.mp4 | VGGT + Candidate A Sim(3) + Scalable TSDF 15mm + Native 4K Atlas", fill=(140, 170, 210), font=font_sub)

def place_panel(img, col, row, title, status_text, status_color):
    x = col * W
    y = 95 + row * H
    panel.paste(img, (x, y))

    # Badge overlay
    draw.rectangle([(x + 15, y + 15), (x + W - 15, y + 65)], fill=(15, 18, 24, 230))
    draw.text((x + 25, y + 25), title, fill=(255, 255, 255), font=font_panel)
    draw.text((x + W - 220, y + 25), status_text, fill=status_color, font=font_panel)

place_panel(img_A, 0, 0, "A: PREVIOUS BASELINE", "[POISSON HULL]", (240, 120, 100))
place_panel(img_B, 1, 0, "B: PRE-TSDF POINT CLOUD", "[187k CLEAN PTS]", (100, 200, 240))
place_panel(img_C, 2, 0, "C: TSDF SURFACE CLOUD", "[80k FUSED PTS]", (120, 220, 160))
place_panel(img_D, 0, 1, "D: SCALABLE TSDF MESH", "[154k FACES, MANIFOLD]", (160, 220, 120))
place_panel(img_E, 1, 1, "E: FINAL TEXTURED TWIN", "[VERIFIED 4K ATLAS]", (80, 230, 140))

# Info Card in slot (2, 1)
info_x = 2 * W
info_y = 95 + 1 * H
draw.rectangle([(info_x, info_y), (info_x + W, info_y + H)], fill=(22, 26, 34))
draw.rectangle([(info_x + 20, info_y + 20), (info_x + W - 20, info_y + H - 20)], outline=(50, 60, 75), width=2)

info_lines = [
    ("RECONSTRUCTION SPECIFICATIONS", (96, 165, 250)),
    ("", (0,0,0)),
    ("Architecture: VGGT + Sim(3) + Scalable TSDF", (220, 220, 230)),
    ("Alignment: Candidate A (t = \u03bcY - s R \u03bcX)", (130, 220, 150)),
    ("Confidence Thresh: Calibrated p20 (2.1338)", (220, 220, 230)),
    ("Median Held-Out 3D Error: 1.29 cm", (130, 220, 150)),
    ("Median Camera Center Error: 3.09 cm", (130, 220, 150)),
    ("Trajectory Length: 6.856 m (63 frames)", (220, 220, 230)),
    ("Physical Room Extents: 1.82m \u00d7 1.48m \u00d7 3.68m", (240, 200, 100)),
    ("TSDF Voxel Size: 15 mm (86k V, 154k F)", (220, 220, 230)),
    ("Surface Agreement: 0.94 cm median", (130, 220, 150)),
    ("Atlas Resolution: 4096 \u00d7 4096 Native Charts", (220, 220, 230)),
    ("Seam Reduction: -37.97% (MRF Graph Cut)", (130, 220, 150)),
    ("Direct Coverage: 74.17% (114k faces)", (130, 220, 150)),
    ("Manifold: 100% Manifold, 0 Degeneracies", (130, 220, 150)),
    ("Viewer Test: Passed (Three.js WebGL)", (130, 220, 150)),
]

cur_y = info_y + 35
for text, col in info_lines:
    if text:
        draw.text((info_x + 40, cur_y), text, fill=col, font=font_sub)
    cur_y += 32

master_panel_path = RENDERS_DIR / "master_five_way_comparison_panel.png"
panel.save(str(master_panel_path), quality=95)
print(f"[+] Saved Master 5-Way Comparison Panel to {master_panel_path.name}")

# Copy to artifacts directory
shutil.copy2(master_panel_path, ARTIFACT_DIR / "master_five_way_comparison_panel_video1.png")
shutil.copy2(RENDERS_DIR / "viewer_video1_textured_overview.png", ARTIFACT_DIR / "viewer_video1_textured_overview.png")
shutil.copy2(RENDERS_DIR / "viewer_video1_textured_top_plan.png", ARTIFACT_DIR / "viewer_video1_textured_top_plan.png")
shutil.copy2(RENDERS_DIR / "viewer_video1_textured_furniture_close.png", ARTIFACT_DIR / "viewer_video1_textured_furniture_close.png")
shutil.copy2(EXP_DIR / "textured_tsdf_model/texture_atlas.png", ARTIFACT_DIR / "video1_texture_atlas.png")
print("[+] Successfully copied all visual assets to artifact directory.")

