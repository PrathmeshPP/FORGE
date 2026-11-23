"""
Build Master 5-Way Architectural Comparison Panel:
A. Original Full-Duration Defective Model (with ghost walls & translation drift)
B. Full-Duration Candidate A Pre-TSDF Point Cloud (256k clean pts)
C. Full-Duration Candidate A TSDF Point Cloud (15mm zero-crossing surface)
D. Full-Duration Candidate A TSDF Mesh (15mm surface mesh)
E. Final Photographic-Textured Digital Twin (4K aspect-preserving atlas + crease normals)
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
GATE_D_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/gate_d_full_duration"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

# Models to render
# A. Defective mesh
orig_mesh_p = BASE_DIR / "tsdf_fusion/voxel_15mm/tsdf_mesh.ply"
orig_mesh = o3d.io.read_triangle_mesh(str(orig_mesh_p))
orig_mesh.compute_vertex_normals()

# B. Pre-TSDF cloud
clean_pcd = o3d.io.read_point_cloud(str(GATE_D_DIR / "full_pre_tsdf_point_cloud_clean.ply"))

# C. TSDF point cloud
tsdf_pcd = o3d.io.read_point_cloud(str(GATE_D_DIR / "tsdf_benchmark/tsdf_pcd_voxel_15mm.ply"))

# D. TSDF 15mm mesh
tsdf_mesh = o3d.io.read_triangle_mesh(str(GATE_D_DIR / "tsdf_benchmark/tsdf_mesh_voxel_15mm.ply"))
tsdf_mesh.compute_vertex_normals()

# E. Textured mesh
tex_mesh = o3d.io.read_triangle_mesh(str(GATE_D_DIR / "textured_tsdf_model/model_gate_d_textured.ply"))
tex_mesh.compute_vertex_normals()

items = [
    {"id": "A_defective", "title": "A. ORIGINAL DEFECTIVE BASELINE", "tag": "Unrecomputed t | Parallel Ghost Walls", "geom": orig_mesh, "is_pcd": False, "badge": (239, 68, 68)},
    {"id": "B_pre_tsdf", "title": "B. CANDIDATE A PRE-TSDF CLOUD", "tag": "256k Clean Pts | Zero Ghost Walls", "geom": clean_pcd, "is_pcd": True, "badge": (56, 189, 248)},
    {"id": "C_tsdf_pcd", "title": "C. CANDIDATE A TSDF CLOUD", "tag": "81k Zero-Crossing Pts | 15mm Voxels", "geom": tsdf_pcd, "is_pcd": True, "badge": (147, 51, 234)},
    {"id": "D_tsdf_mesh", "title": "D. CANDIDATE A TSDF MESH", "tag": "86k V / 152k F | Watertight Surface", "geom": tsdf_mesh, "is_pcd": False, "badge": (245, 158, 11)},
    {"id": "E_textured", "title": "E. FINAL TEXTURED DIGITAL TWIN", "tag": "Embedded 4K Atlas | Crease Normals", "geom": tex_mesh, "is_pcd": False, "badge": (34, 197, 94)}
]

def render_view(geom, out_path, offset, view_up=[0, 1, 0], is_pcd=False):
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    ext = bbox.get_extent()
    diag = float(np.linalg.norm(ext))
    if diag < 1e-4: diag = 1.0

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=540)
    vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.08, 0.09, 0.12])
    opt.light_on = True
    if is_pcd:
        opt.point_size = 2.4
        opt.light_on = False

    ctr = vis.get_view_control()
    cam_pos = center + offset * (diag * 0.72)
    front = center - cam_pos
    front /= (np.linalg.norm(front) + 1e-8)
    ctr.set_front(front)
    ctr.set_up(view_up)
    ctr.set_lookat(center)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()

print("[*] Rendering 5 comparative models from canonical angles...")
iso_offset = np.array([-1.0, 0.85, 1.2])
top_offset = np.array([0.0, 2.8, 0.0])

for itm in items:
    p_iso = GATE_D_DIR / f"five_way_{itm['id']}_iso.png"
    p_top = GATE_D_DIR / f"five_way_{itm['id']}_top.png"
    render_view(itm["geom"], p_iso, iso_offset, is_pcd=itm["is_pcd"])
    render_view(itm["geom"], p_top, top_offset, view_up=[0, 0, -1], is_pcd=itm["is_pcd"])
    itm["p_iso"] = p_iso
    itm["p_top"] = p_top
    print(f"  Rendered {itm['id']}")

# Assemble 5-Column Master Panel
cell_w, cell_h = 360, 260
header_h = 75
sub_h = 36
panel_w = 10 + len(items) * (cell_w + 8)
panel_h = header_h + sub_h + cell_h * 2 + 15

panel = Image.new("RGB", (panel_w, panel_h), color=(15, 23, 42))
draw = ImageDraw.Draw(panel)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 20)
    font_sub = ImageFont.truetype("arial.ttf", 12)
    font_col = ImageFont.truetype("arialbd.ttf", 11)
    font_row = ImageFont.truetype("arialbd.ttf", 10)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_col = ImageFont.load_default()
    font_row = ImageFont.load_default()

# Header
draw.rectangle([0, 0, panel_w, header_h], fill=(24, 32, 47))
draw.text((25, 16), "F.O.R.G.E. — GATE D: COMPLETE 5-STAGE DIGITAL TWIN COMPARISON (136s, 179 FRAMES)", fill=(56, 189, 248), font=font_title)
draw.text((25, 44), "Full Video Evaluation: Original Defective vs Corrected Pre-TSDF vs TSDF Cloud vs TSDF Mesh vs Final Textured Twin", fill=(148, 163, 184), font=font_sub)

for idx, itm in enumerate(items):
    x = 10 + idx * (cell_w + 8)

    # Column Subheader
    draw.rectangle([x, header_h, x + cell_w, header_h + sub_h], fill=(30, 41, 59))
    draw.rectangle([x + 6, header_h + 6, x + 12, header_h + 28], fill=itm["badge"])
    draw.text((x + 18, header_h + 6), itm["title"], fill=(241, 245, 249), font=font_col)
    draw.text((x + 18, header_h + 20), itm["tag"], fill=(148, 163, 184), font=font_row)

    # Row 1: Isometric View
    y1 = header_h + sub_h + 5
    if itm["p_iso"].exists():
        im1 = Image.open(itm["p_iso"]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        panel.paste(im1, (x, y1))
    draw.rectangle([x, y1, x + 130, y1 + 18], fill=(15, 23, 42, 200))
    draw.text((x + 6, y1 + 3), "ISOMETRIC OVERVIEW", fill=(226, 232, 240), font=font_row)

    # Row 2: Top-down Floorplan
    y2 = y1 + cell_h + 5
    if itm["p_top"].exists():
        im2 = Image.open(itm["p_top"]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        panel.paste(im2, (x, y2))
    draw.rectangle([x, y2, x + 130, y2 + 18], fill=(15, 23, 42, 200))
    draw.text((x + 6, y2 + 3), "TOP-DOWN FLOORPLAN", fill=(226, 232, 240), font=font_row)

    draw.rectangle([x, y1, x + cell_w, y1 + cell_h], outline=(51, 65, 85), width=1)
    draw.rectangle([x, y2, x + cell_w, y2 + cell_h], outline=(51, 65, 85), width=1)

panel_out = GATE_D_DIR / "master_five_way_comparison_panel.png"
panel.save(panel_out, quality=95)
print(f"[+] Saved Master 5-Way Comparison Panel: {panel_out.name}")

if ARTIFACT_DIR.is_dir():
    import shutil
    shutil.copy(panel_out, ARTIFACT_DIR / "master_five_way_comparison_panel.png")
    print(f"[+] Copied to artifacts directory.")
