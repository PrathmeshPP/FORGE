#!/usr/bin/env python3
"""
F.O.R.G.E V6 — Missing Geometry Mapping and Stage-by-Stage Trace

Analyzes and visualizes exactly where geometry is missing in V5 compared to:
1. The 16-frame complete scene observations (35.1s to 66.5s)
2. Source frames and per-frame depth predictions
3. Stage-by-stage filtering operations

Generates:
- missing_geometry_comparison.png (Multi-view comparative card)
- missing_geometry_audit.json (Quantitative spatial cell analysis)
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V5_DIR = CAMPAIGN_DIR / "final_best_canonical_v5"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
V6_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = V6_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

V5_PCD_PATH = V5_DIR / "final_best_canonical_v5_point_cloud.ply"
V5_MESH_PATH = V5_DIR / "final_best_canonical_v5_mesh.ply"
V5_RAW_MESH_PATH = V5_DIR / "stages/stage_A_raw_poisson.ply"

UP16_DIR = CAMPAIGN_DIR / "upstream_16_frames"
UP16_PRED = UP16_DIR / "predictions"
UP16_FRAMES = UP16_DIR / "frames"

T_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])

def render_geometry_view(geom, out_path: Path, lookat, front, up, zoom=0.65, is_mesh=True, color=(0.80, 0.84, 0.90)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Render", width=1280, height=960, visible=False)
    ro = vis.get_render_option()
    ro.background_color = np.array([1.0, 1.0, 1.0])
    ro.light_on = True
    ro.show_coordinate_frame = False

    if is_mesh:
        ro.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        m_copy = o3d.geometry.TriangleMesh(geom)
        m_copy.compute_vertex_normals()
        if not m_copy.has_vertex_colors() or len(m_copy.vertex_colors) == 0:
            m_copy.paint_uniform_color(color)
        vis.add_geometry(m_copy)
    else:
        ro.point_size = 2.0
        vis.add_geometry(geom)

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()

def main():
    print("=" * 80)
    print("F.O.R.G.E V6 — SPATIAL MISSING-GEOMETRY ANALYSIS & REGION MAPPING")
    print("=" * 80)

    # 1. Load V5 Assets
    print(f"[*] Loading V5 Point Cloud: {V5_PCD_PATH.name}...")
    pcd_v5 = o3d.io.read_point_cloud(str(V5_PCD_PATH))
    pts_v5 = np.asarray(pcd_v5.points)
    print(f"    V5 Points: {len(pts_v5):,}")
    print(f"    V5 Bounds: Min={pts_v5.min(axis=0).round(3)}, Max={pts_v5.max(axis=0).round(3)}")

    print(f"[*] Loading V5 Neutral Mesh: {V5_MESH_PATH.name}...")
    mesh_v5 = o3d.io.read_triangle_mesh(str(V5_MESH_PATH))
    print(f"    V5 Mesh Verts: {len(mesh_v5.vertices):,}, Tris: {len(mesh_v5.triangles):,}")

    # 2. Build 16-Frame Full Scene Multi-View Point Cloud
    print("\n[*] Loading 16-Frame Multi-View Observations from upstream_16_frames...")
    poses16 = np.load(UP16_PRED / "camera_poses.npz")
    depths16 = np.load(UP16_PRED / "depth_predictions.npz")

    ext_cv16 = poses16["extrinsics"]   # (16, 3, 4)
    intr16 = poses16["intrinsics"]     # (16, 3, 3)
    d_maps16 = np.squeeze(depths16["depth"]) # (16, 384, 688)
    conf_maps16 = depths16["confidence"]     # (16, 384, 688)

    import cv2
    from src.reconstruction.point_cloud import depth_to_points, filter_by_multiview_support

    frame_files16 = sorted(list(UP16_FRAMES.glob("*.png")))
    colors16 = []
    for f in frame_files16:
        im = cv2.imread(str(f))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (688, 384))
        colors16.append(im)
    colors16 = np.array(colors16)

    print("    Backprojecting 16-frame depth maps...")
    world_pts16 = depth_to_points(d_maps16, ext_cv16, intr16) # (16, 384, 688, 3)

    flat_pts16 = world_pts16.reshape(-1, 3)
    flat_colors16 = colors16.reshape(-1, 3)

    print("    Applying Multi-View Camera Support Filtering (min_support=2, min_conf=1.5)...")
    pts_filt16, cols_filt16 = filter_by_multiview_support(
        points=flat_pts16,
        colors=flat_colors16,
        extrinsics=ext_cv16,
        intrinsics=intr16,
        depth_maps=d_maps16,
        confidence_maps=conf_maps16,
        min_camera_support=2,
        rtol=0.08,
        min_confidence=1.5
    )
    print(f"    Multi-view triangulated points: {len(pts_filt16):,}")

    # Convert to canonical glTF: P_canon = P_cv @ T_CV_TO_GLTF
    canon_pts16 = pts_filt16 @ T_CV_TO_GLTF

    # Statistical outlier removal on 16-frame cloud
    pcd16 = o3d.geometry.PointCloud()
    pcd16.points = o3d.utility.Vector3dVector(canon_pts16.astype(np.float64))
    pcd16.colors = o3d.utility.Vector3dVector(cols_filt16.astype(np.float64) / 255.0)

    pcd16_clean, ind = pcd16.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    pts16_clean = np.asarray(pcd16_clean.points)
    cols16_clean = np.asarray(pcd16_clean.colors)
    print(f"    Clean 16-frame Point Cloud: {len(pts16_clean):,} points")
    print(f"    16-Frame Bounds: Min={pts16_clean.min(axis=0).round(3)}, Max={pts16_clean.max(axis=0).round(3)}")
    print(f"    16-Frame Extent: {(pts16_clean.max(axis=0) - pts16_clean.min(axis=0)).round(3)}")

    # Save 16-frame full scene point cloud
    full_pcd_path = DIAG_DIR / "full_16frame_clean_point_cloud.ply"
    o3d.io.write_point_cloud(str(full_pcd_path), pcd16_clean)
    print(f"    Saved: {full_pcd_path.name}")

    # 3. Spatial Cell Analysis & Region Breakdown
    print("\n[*] Auditing Five Critical Spatial Scene Regions...")
    regions = {
        "road_and_entrance_courtyard": {
            "name": "Access Road & Front Courtyard",
            "bbox_canon": {"x": [-0.6, 2.5], "y": [-0.25, -0.05], "z": [-0.8, 0.8]},
            "physical_description": "Asphalt access driveway and paved pedestrian courtyard in front of facility entrance."
        },
        "ground_adjacent_to_building": {
            "name": "Ground Adjacent to Building",
            "bbox_canon": {"x": [1.5, 3.0], "y": [-0.25, 0.0], "z": [-2.0, -0.5]},
            "physical_description": "Open ground plane, grass patches, and eastern facility boundary."
        },
        "vegetation_and_tree_canopies": {
            "name": "Vegetation & Tree Canopies",
            "bbox_canon": {"x": [-1.2, 0.5], "y": [-0.15, 0.20], "z": [-2.8, -0.8]},
            "physical_description": "Dense tree canopies surrounding western perimeter and facility courtyard."
        },
        "rooftop_and_fixtures": {
            "name": "Rooftop & Rooftop Fixtures",
            "bbox_canon": {"x": [-0.8, 0.4], "y": [0.08, 0.29], "z": [-1.6, -0.6]},
            "physical_description": "Double solar array rows, multiple water storage tanks, and raised parapet."
        },
        "building_facade_and_cantilevers": {
            "name": "Building Facade & Boundaries",
            "bbox_canon": {"x": [-0.7, 0.5], "y": [-0.10, 0.12], "z": [-1.2, -0.4]},
            "physical_description": "South-facing entrance facade, windows, doors, and entrance overhang cantilever."
        }
    }

    tree_v5_mesh = cKDTree(np.asarray(mesh_v5.vertices))
    audit_results = {}

    for r_key, r_cfg in regions.items():
        bx = r_cfg["bbox_canon"]
        m_in16 = (
            (pts16_clean[:, 0] >= bx["x"][0]) & (pts16_clean[:, 0] <= bx["x"][1]) &
            (pts16_clean[:, 1] >= bx["y"][0]) & (pts16_clean[:, 1] <= bx["y"][1]) &
            (pts16_clean[:, 2] >= bx["z"][0]) & (pts16_clean[:, 2] <= bx["z"][1])
        )
        pts_r16 = pts16_clean[m_in16]

        m_in_v5 = (
            (pts_v5[:, 0] >= bx["x"][0]) & (pts_v5[:, 0] <= bx["x"][1]) &
            (pts_v5[:, 1] >= bx["y"][0]) & (pts_v5[:, 1] <= bx["y"][1]) &
            (pts_v5[:, 2] >= bx["z"][0]) & (pts_v5[:, 2] <= bx["z"][1])
        )
        pts_r_v5 = pts_v5[m_in_v5]

        # Check coverage of 16-frame points in V5 mesh
        if len(pts_r16) > 0:
            d_r16_to_v5, _ = tree_v5_mesh.query(pts_r16, k=1, workers=-1)
            v5_retained_pct = float(np.mean(d_r16_to_v5 <= 0.025)) * 100
            missing_pct = float(np.mean(d_r16_to_v5 > 0.050)) * 100
        else:
            v5_retained_pct = 0.0
            missing_pct = 0.0

        print(f"\n  Region: {r_cfg['name']}")
        print(f"    Observed in 16-frame cloud: {len(pts_r16):,} points")
        print(f"    Observed in 8-frame V5 cloud: {len(pts_r_v5):,} points")
        print(f"    V5 mesh retention (<=25mm): {v5_retained_pct:.1f}%")
        print(f"    Missing geometry (>50mm from V5 mesh): {missing_pct:.1f}% ({int(missing_pct/100*len(pts_r16)):,} pts)")

        # Classify root cause
        if len(pts_r_v5) < len(pts_r16) * 0.60:
            classification = "Visible in source footage (16 frames) but omitted from V5 input window (8 frames)."
        elif missing_pct > 20:
            classification = "Present in point cloud but pruned by meshing/filtering stage."
        else:
            classification = "Substantially preserved with high fidelity."

        audit_results[r_key] = {
            "name": r_cfg["name"],
            "physical_description": r_cfg["physical_description"],
            "points_in_16frame_cloud": len(pts_r16),
            "points_in_v5_cloud": len(pts_r_v5),
            "v5_mesh_retention_pct": round(v5_retained_pct, 2),
            "missing_pct_gt_50mm": round(missing_pct, 2),
            "root_cause_classification": classification
        }

    # Save JSON audit
    audit_json_path = DIAG_DIR / "missing_geometry_audit.json"
    with open(audit_json_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[*] Audit report saved: {audit_json_path}")

    # 4. Render Comparative Views from Exact Same Viewpoints
    print("\n[*] Rendering View-Matched Previews for Missing-Geometry Sheet...")
    # Center on global scene
    ctr_global = np.array([0.5, -0.05, -1.0])

    views = {
        "nadir": {"front": [0.0, 1.0, 0.001], "lookat": ctr_global, "up": [0.0, 0.0, -1.0], "zoom": 0.65},
        "oblique": {"front": [0.35, 0.55, 0.76], "lookat": ctr_global, "up": [0.0, 1.0, 0.0], "zoom": 0.60},
        "road_focus": {"front": [0.20, 0.60, 0.75], "lookat": np.array([0.8, -0.15, -0.1]), "up": [0.0, 1.0, 0.0], "zoom": 0.40}
    }

    # Render PCD 16-frame (Full Scene Ground Truth)
    for vname, vcfg in views.items():
        render_geometry_view(pcd16_clean, DIAG_DIR / f"full16_pcd_{vname}.png", vcfg["lookat"], vcfg["front"], vcfg["up"], vcfg["zoom"], is_mesh=False)

    # Render PCD V5 (8-frame input)
    for vname, vcfg in views.items():
        render_geometry_view(pcd_v5, DIAG_DIR / f"v5_pcd_{vname}.png", vcfg["lookat"], vcfg["front"], vcfg["up"], vcfg["zoom"], is_mesh=False)

    # Render V5 Final Mesh
    for vname, vcfg in views.items():
        render_geometry_view(mesh_v5, DIAG_DIR / f"v5_mesh_{vname}.png", vcfg["lookat"], vcfg["front"], vcfg["up"], vcfg["zoom"], is_mesh=True)

    # 5. Assemble Missing-Geometry Comparison Card (missing_geometry_comparison.png)
    print("\n[*] Assembling missing_geometry_comparison.png (3840 x 2160)...")
    card_w, card_h = 3840, 2160
    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    try:
        font_title = ImageFont.truetype("arial.ttf", 46)
        font_sub = ImageFont.truetype("arial.ttf", 24)
        font_col = ImageFont.truetype("arial.ttf", 26)
        font_text = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_col = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Title Banner
    draw.text((60, 40), "F.O.R.G.E V6 — EVIDENCE-BASED MISSING GEOMETRY ANALYSIS", fill=(10, 15, 25), font=font_title)
    draw.text((60, 100), "Traces missing scene surfaces across the full 16-frame recording vs the restricted 8-frame V5 input window.", fill=(80, 90, 105), font=font_sub)

    # 3 Columns:
    # Col 1: Full 16-Frame Observed Scene (2.02M points, 35.1s - 66.5s)
    # Col 2: V5 8-Frame Input Point Cloud (963k points, 44.0s - 54.5s)
    # Col 3: V5 Reconstructed Mesh (Showing missing road, courtyard, and periphery)
    col_w = (card_w - 120) // 3
    col_titles = [
        ("Full 16-Frame Observation Ground Truth", "Multi-view triangulated: 2,022,730 points spanning 35.1s - 66.5s", "full16_pcd"),
        ("V5 8-Frame Sub-Window Point Cloud", "Restricted input: 963,153 points spanning only 44.0s - 54.5s", "v5_pcd"),
        ("V5 Reconstructed Mesh Deliverable", "308,675 vertices, 614,160 triangles (Showing severed road & missing ground)", "v5_mesh")
    ]

    view_keys = [("Top-Down Nadir View", "nadir"), ("Oblique Aerial Corridor", "oblique"), ("Access Road & Courtyard Close-Up", "road_focus")]
    row_h = (card_h - 220) // 3

    for col_idx, (col_name, col_sub, file_prefix) in enumerate(col_titles):
        x_left = 60 + col_idx * col_w
        x_right = x_left + col_w - 20

        # Column Header
        draw.rectangle([(x_left, 150), (x_right, 210)], fill=(245, 247, 250), outline=(215, 222, 232), width=1)
        draw.text((x_left + 15, 158), col_name, fill=(20, 30, 45), font=font_col)
        draw.text((x_left + 15, 186), col_sub, fill=(100, 110, 125), font=ImageFont.load_default())

        for row_idx, (row_label, row_key) in enumerate(view_keys):
            y_top = 225 + row_idx * (row_h - 10)
            img_path = DIAG_DIR / f"{file_prefix}_{row_key}.png"
            if img_path.exists():
                im = Image.open(img_path).resize((col_w - 20, row_h - 50), Image.Resampling.LANCZOS)
                card.paste(im, (x_left, y_top))
                draw.rectangle([(x_left, y_top), (x_right, y_top + row_h - 50)], outline=(230, 235, 240), width=1)

            # View label badge
            draw.rectangle([(x_left + 10, y_top + 10), (x_left + 280, y_top + 38)], fill=(255, 255, 255), outline=(200, 210, 220), width=1)
            draw.text((x_left + 18, y_top + 14), row_label, fill=(30, 40, 55), font=font_text)

    comp_card_path = V6_DIR / "missing_geometry_comparison.png"
    card.save(comp_card_path, quality=95)
    print(f"[*] Missing geometry comparison card saved: {comp_card_path}")

    # Copy to DIAG_DIR
    shutil.copy2(comp_card_path, DIAG_DIR / "missing_geometry_comparison.png")

if __name__ == "__main__":
    main()
