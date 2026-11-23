#!/usr/bin/env python3
"""
V4 Stage 2: Visual Comparison Audit — Point Cloud vs V3 Mesh
Renders identical camera views of:
1. Point Cloud (colored)
2. Neutral V3 Mesh
3. Textured V3 Model
4. Point Cloud error map (colored by distance to V3 mesh)

All on pure white (#FFFFFF) background with exact matching camera targets and FOVs.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V3_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v3"
V4_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v4"
DIAG_DIR = V4_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def render_geometry_view(geom, lookat, front, up, zoom, w=1000, h=750, is_point_cloud=False, point_size=2.0, neutral_color=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    
    if neutral_color is not None and isinstance(geom, o3d.geometry.TriangleMesh):
        g_render = o3d.geometry.TriangleMesh(geom)
        g_render.compute_vertex_normals()
        g_render.paint_uniform_color(neutral_color)
    else:
        g_render = geom
        
    vis.add_geometry(g_render)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])  # Pure white #FFFFFF
    
    if is_point_cloud:
        opt.point_size = point_size
        opt.light_on = False
    elif neutral_color is not None:
        opt.light_on = True
    else:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        opt.light_on = False

    ctrl = vis.get_view_control()
    ctrl.set_lookat(lookat)
    ctrl.set_front(front)
    ctrl.set_up(up)
    ctrl.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    buf = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    return Image.fromarray((np.asarray(buf) * 255).astype(np.uint8))

def main():
    print("=" * 80)
    print("V4 STAGE 2: VISUAL AUDIT — POINT CLOUD VS V3 MESH")
    print("=" * 80)

    # 1. Load V3 point cloud
    pc_path = V3_DIR / "final_best_canonical_v3_point_cloud.ply"
    print(f"Loading point cloud: {pc_path}")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    bbox_pcd = pcd.get_axis_aligned_bounding_box()
    ctr = bbox_pcd.get_center()
    print(f"  Point count: {len(pcd.points):,}")
    print(f"  Center: {ctr.round(3)}")

    # 2. Load V3 mesh
    mesh_path = V3_DIR / "final_best_canonical_v3_mesh.ply"
    print(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    # 3. Create Distance-to-Mesh Heatmap Point Cloud
    print("Computing Point Cloud distance-to-mesh heatmap...")
    mesh_tree = o3d.geometry.KDTreeFlann(mesh)
    pts = np.asarray(pcd.points)
    dists = np.zeros(len(pts), dtype=np.float32)
    for i in range(len(pts)):
        _, _, d2 = mesh_tree.search_knn_vector_3d(pts[i], 1)
        dists[i] = np.sqrt(d2[0])
    
    # Pure numpy colormap: 0.0=blue (<=5mm), 0.5=yellow (15mm), 1.0=bright red (>=30mm)
    norm_d = np.clip(dists / 0.03, 0.0, 1.0)
    heatmap_colors = np.zeros((len(pts), 3), dtype=np.float64)
    # R: rises from 0 at val=0.3 to 1 at val=0.8
    heatmap_colors[:, 0] = np.clip((norm_d - 0.2) / 0.5, 0.0, 1.0)
    # G: peaks at 0.5
    heatmap_colors[:, 1] = np.clip(1.0 - 2.0 * np.abs(norm_d - 0.4), 0.0, 1.0)
    # B: falls from 1 at val=0 to 0 at val=0.5
    heatmap_colors[:, 2] = np.clip(1.0 - norm_d / 0.4, 0.0, 1.0)
    
    pcd_heatmap = o3d.geometry.PointCloud()
    pcd_heatmap.points = o3d.utility.Vector3dVector(pts)
    pcd_heatmap.colors = o3d.utility.Vector3dVector(heatmap_colors)

    # Views to render
    # Canonical coordinates: +Y is Up (sky), -Y is Down (ground), X-Z is ground plane
    views = [
        {
            "name": "Top-Down Nadir",
            "lookat": ctr,
            "front": [0.0, 1.0, 0.001],
            "up": [0.0, 0.0, -1.0],
            "zoom": 0.75,
        },
        {
            "name": "Oblique Aerial Corridor",
            "lookat": ctr,
            "front": [0.35, 0.55, 0.76],
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.70,
        },
        {
            "name": "Rooftop Fixtures & Solar Focus",
            "lookat": [ctr[0], ctr[1] + 0.05, ctr[2]],
            "front": [0.20, 0.45, 0.87],
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.40,
        },
        {
            "name": "Side Facade Profile",
            "lookat": ctr,
            "front": [0.95, 0.15, 0.25],
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.65,
        }
    ]

    # Generate 4-panel comparison for each view:
    # [Point Cloud (RGB)] [Point Cloud (Error Heatmap)] [V3 Neutral Mesh] [V3 Textured Mesh]
    w, h = 800, 600
    for v_idx, v in enumerate(views):
        v_name = v["name"]
        print(f"\nRendering View {v_idx+1}: {v_name}...")

        img_pc_rgb = render_geometry_view(pcd, v["lookat"], v["front"], v["up"], v["zoom"], w=w, h=h, is_point_cloud=True, point_size=2.0)
        img_pc_heat = render_geometry_view(pcd_heatmap, v["lookat"], v["front"], v["up"], v["zoom"], w=w, h=h, is_point_cloud=True, point_size=2.0)
        img_mesh_neutral = render_geometry_view(mesh, v["lookat"], v["front"], v["up"], v["zoom"], w=w, h=h, neutral_color=[0.78, 0.80, 0.84])
        img_mesh_tex = render_geometry_view(mesh, v["lookat"], v["front"], v["up"], v["zoom"], w=w, h=h)

        # Composite into 2x2 grid or 1x4 strip
        strip = Image.new("RGB", (w * 2, h * 2 + 80), (255, 255, 255))
        draw = ImageDraw.Draw(strip)

        # Header banner
        draw.rectangle([(0, 0), (w * 2, 70)], fill=(245, 247, 250))
        draw.line([(0, 70), (w * 2, 70)], fill=(220, 224, 230), width=2)
        draw.text((30, 22), f"F.O.R.G.E V4 Visual Audit — {v_name} — Point Cloud vs V3 Mesh Geometry", fill=(30, 41, 59))

        panels = [
            (img_pc_rgb, 0, 75, "1. Dense Point Cloud (Photographic RGB) — 963,153 Points"),
            (img_pc_heat, w, 75, "2. Point Cloud Coverage Heatmap (Blue <= 5mm, Red > 20mm missing)"),
            (img_mesh_neutral, 0, h + 80, "3. V3 Neutral Mesh (31,245 Verts, 873 Disconnected Fragments)"),
            (img_mesh_tex, w, h + 80, "4. V3 Textured Mesh (TSDF 12mm Voxelization)")
        ]

        for p_img, px, py, p_title in panels:
            strip.paste(p_img, (px, py))
            # Subtitle banner
            draw.rectangle([(px, py), (px + w, py + 30)], fill=(240, 243, 246))
            draw.text((px + 15, py + 7), p_title, fill=(50, 60, 75))
            draw.rectangle([(px, py), (px + w, py + h)], outline=(225, 230, 236), width=1)

        slug = v_name.lower().replace(" ", "_").replace("-", "_").replace("&", "and")
        out_card = DIAG_DIR / f"audit_comparison_{v_idx+1}_{slug}.png"
        strip.save(out_card, format="PNG")
        print(f"  Saved comparison card: {out_card}")

    print("\nVisual audit comparison sheets generated successfully!")

if __name__ == "__main__":
    main()
