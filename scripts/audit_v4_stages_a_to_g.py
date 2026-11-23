#!/usr/bin/env python3
"""
F.O.R.G.E V5 — Stage-by-Stage Reconstruction Audit (Stages A through G)

Audits each operation in the V4 reconstruction pipeline to pinpoint exactly where
ground, vegetation, and scene geometry disappear.

Stages:
- Stage A: Raw Poisson output (Depth 10, scale 1.1) before any trimming or filtering.
- Stage B: After Poisson density trimming only (lowest 2% density removed).
- Stage C: After vertex-to-point-cloud support-distance filtering only (dists > 22 mm removed).
- Stage D: After long-triangle filtering only (max_e > 50 mm removed).
- Stage E: After non-manifold topology cleanup only.
- Stage F: After small connected-component removal only (clusters < 60 faces removed).
- Stage G: Final V4 result on disk for reference.
"""

from __future__ import annotations
import os
import sys
import time
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V3_DIR = CAMPAIGN_DIR / "final_best_canonical_v3"
V4_DIR = CAMPAIGN_DIR / "final_best_canonical_v4"
V5_DIR = CAMPAIGN_DIR / "final_best_canonical_v5"

V5_DIR.mkdir(parents=True, exist_ok=True)
STAGES_DIR = V5_DIR / "stages"
DIAG_DIR = V5_DIR / "diagnostics"
STAGES_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PCD_PATH = V3_DIR / "final_best_canonical_v3_point_cloud.ply"
V4_MESH_PATH = V4_DIR / "final_best_canonical_v4_mesh.ply"

def render_mesh_views(mesh: o3d.geometry.TriangleMesh, prefix: str, out_dir: Path) -> dict[str, Path]:
    """Renders 4 canonical viewpoints on pure #FFFFFF white background."""
    bbox = mesh.get_axis_aligned_bounding_box()
    ctr = bbox.get_center()
    extent = bbox.get_max_extent()

    views = {
        "nadir": {
            "front": [0.0, 1.0, 0.001],
            "lookat": ctr,
            "up": [0.0, 0.0, -1.0],
            "zoom": 0.70
        },
        "oblique": {
            "front": [0.35, 0.55, 0.76],
            "lookat": ctr,
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.65
        },
        "side_facade": {
            "front": [1.0, 0.05, 0.0],
            "lookat": ctr,
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.70
        },
        "closeup": {
            "front": [0.20, 0.45, 0.87],
            "lookat": ctr + np.array([0.0, 0.05, -0.2]),
            "up": [0.0, 1.0, 0.0],
            "zoom": 0.35
        }
    }

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Render_{prefix}", width=1280, height=960, visible=False)
    vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
    vis.get_render_option().light_on = True
    vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    vis.get_render_option().show_coordinate_frame = False

    # Ensure mesh has normals and neutral silver/light blue shading
    m_render = copy.deepcopy(mesh)
    m_render.compute_vertex_normals()
    if not m_render.has_vertex_colors():
        m_render.paint_uniform_color([0.78, 0.82, 0.88])
    else:
        # Mix 50% vertex color with neutral shading for visibility
        colors = np.asarray(m_render.vertex_colors)
        if len(colors) == 0:
            m_render.paint_uniform_color([0.78, 0.82, 0.88])

    vis.add_geometry(m_render)
    ctr_ctrl = vis.get_view_control()

    out_paths = {}
    for vname, vcfg in views.items():
        ctr_ctrl.set_lookat(vcfg["lookat"])
        ctr_ctrl.set_front(vcfg["front"])
        ctr_ctrl.set_up(vcfg["up"])
        ctr_ctrl.set_zoom(vcfg["zoom"])
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
        img_path = out_dir / f"{prefix}_{vname}.png"
        vis.capture_screen_image(str(img_path), do_render=True)
        out_paths[vname] = img_path

    vis.destroy_window()
    return out_paths

def evaluate_mesh_stage(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    pts: np.ndarray,
    colors: np.ndarray,
    tree_pc: cKDTree,
    sample_pc_idx: np.ndarray,
    green_idx: np.ndarray,
    lower_idx: np.ndarray,
    roof_idx: np.ndarray
) -> dict:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    n_v = len(verts)
    n_t = len(tris)
    if n_v == 0 or n_t == 0:
        return {"name": name, "num_vertices": n_v, "num_triangles": n_t, "error": "Empty mesh"}

    # Topology
    nme_true = len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    nme_all = len(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    boundary = nme_all - nme_true

    # Components
    clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
    cluster_counts = np.asarray(cluster_counts)
    num_comp = len(cluster_counts)
    comp_sorted = sorted(cluster_counts.tolist(), reverse=True)
    largest_comp = comp_sorted[0] if comp_sorted else 0
    largest_pct = round(largest_comp / n_t * 100, 2)
    small_comp = int(np.sum(cluster_counts < 60))

    # Bounding Box
    bbox_min = verts.min(axis=0).tolist()
    bbox_max = verts.max(axis=0).tolist()
    extent = (verts.max(axis=0) - verts.min(axis=0)).tolist()

    # Elevation percentiles
    y_perc = [float(np.percentile(verts[:, 1], p)) for p in [0, 5, 25, 50, 75, 95, 100]]

    # Mesh KDTree for Point Cloud -> Mesh Distance
    tree_mesh = cKDTree(verts)

    # 1. Overall PC -> Mesh distance
    sample_pts = pts[sample_pc_idx]
    d_all, _ = tree_mesh.query(sample_pts, k=1, workers=-1)
    pc_med = float(np.median(d_all)) * 1000
    pc_mean = float(np.mean(d_all)) * 1000
    pc_p95 = float(np.percentile(d_all, 95)) * 1000
    pc_w10mm = float(np.mean(d_all <= 0.010)) * 100
    pc_w20mm = float(np.mean(d_all <= 0.020)) * 100
    pc_b20mm = float(np.mean(d_all > 0.020)) * 100

    # 2. Regional Subsets
    d_green, _ = tree_mesh.query(pts[green_idx], k=1, workers=-1)
    green_med = float(np.median(d_green)) * 1000
    green_w10mm = float(np.mean(d_green <= 0.010)) * 100
    green_b20mm = float(np.mean(d_green > 0.020)) * 100

    d_lower, _ = tree_mesh.query(pts[lower_idx], k=1, workers=-1)
    lower_med = float(np.median(d_lower)) * 1000
    lower_w10mm = float(np.mean(d_lower <= 0.010)) * 100
    lower_b20mm = float(np.mean(d_lower > 0.020)) * 100

    d_roof, _ = tree_mesh.query(pts[roof_idx], k=1, workers=-1)
    roof_med = float(np.median(d_roof)) * 1000
    roof_w10mm = float(np.mean(d_roof <= 0.010)) * 100
    roof_b20mm = float(np.mean(d_roof > 0.020)) * 100

    # 3. Mesh Surface -> Point Cloud Distance (sampled over triangle surfaces)
    # Uniformly sample 50k points across triangle surfaces
    pcd_mesh_surf = mesh.sample_points_uniformly(number_of_points=min(50000, n_t * 3))
    surf_pts = np.asarray(pcd_mesh_surf.points)
    d_surf2pc, _ = tree_pc.query(surf_pts, k=1, workers=-1)
    surf_med = float(np.median(d_surf2pc)) * 1000
    surf_p95 = float(np.percentile(d_surf2pc, 95)) * 1000
    surf_b20mm = float(np.mean(d_surf2pc > 0.020)) * 100
    surf_b50mm = float(np.mean(d_surf2pc > 0.050)) * 100

    # Triangle area distribution
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_area = float(np.sum(tri_areas))

    # Long edges count (>50mm)
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e1, e2), e3)
    long_tri_count = int(np.sum(max_e > 0.050))

    return {
        "name": name,
        "num_vertices": n_v,
        "num_triangles": n_t,
        "total_surface_area_m2": round(total_area, 3),
        "components_count": num_comp,
        "largest_component_triangles": largest_comp,
        "largest_component_pct": largest_pct,
        "small_components_under_60_faces": small_comp,
        "true_non_manifold_edges": nme_true,
        "boundary_edges": boundary,
        "long_triangles_gt_50mm": long_tri_count,
        "bounding_box_min": [round(x, 4) for x in bbox_min],
        "bounding_box_max": [round(x, 4) for x in bbox_max],
        "extent_xyz": [round(x, 4) for x in extent],
        "y_elevation_percentiles": [round(y, 4) for y in y_perc],
        "metrics_overall_pc2mesh": {
            "median_mm": round(pc_med, 2),
            "mean_mm": round(pc_mean, 2),
            "p95_mm": round(pc_p95, 2),
            "pct_within_10mm": round(pc_w10mm, 2),
            "pct_within_20mm": round(pc_w20mm, 2),
            "pct_exceeding_20mm": round(pc_b20mm, 2)
        },
        "metrics_regional_pc2mesh": {
            "green_vegetation": {
                "median_mm": round(green_med, 2),
                "pct_within_10mm": round(green_w10mm, 2),
                "pct_exceeding_20mm": round(green_b20mm, 2)
            },
            "ground_lower_terrain": {
                "median_mm": round(lower_med, 2),
                "pct_within_10mm": round(lower_w10mm, 2),
                "pct_exceeding_20mm": round(lower_b20mm, 2)
            },
            "rooftop_fixtures": {
                "median_mm": round(roof_med, 2),
                "pct_within_10mm": round(roof_w10mm, 2),
                "pct_exceeding_20mm": round(roof_b20mm, 2)
            }
        },
        "metrics_mesh_surface2pc": {
            "median_mm": round(surf_med, 2),
            "p95_mm": round(surf_p95, 2),
            "pct_unsupported_gt_20mm": round(surf_b20mm, 2),
            "pct_unsupported_gt_50mm": round(surf_b50mm, 2)
        }
    }

def main():
    print("=" * 80)
    print("F.O.R.G.E V5 — STAGE-BY-STAGE RECONSTRUCTION AUDIT (STAGES A -> G)")
    print("=" * 80)

    # 1. Load Point Cloud
    print(f"\n[*] Loading Point Cloud: {PCD_PATH}")
    pcd = o3d.io.read_point_cloud(str(PCD_PATH))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"    Loaded {len(pts):,} points.")

    tree_pc = cKDTree(pts)

    # Pre-compute category masks for regional evaluation
    rng = np.random.default_rng(42)
    sample_pc_idx = rng.choice(len(pts), min(100000, len(pts)), replace=False)

    is_green = (colors[:, 1] > colors[:, 0] * 1.05) & (colors[:, 1] > colors[:, 2] * 1.05)
    green_idx = np.where(is_green)[0]
    green_sub_idx = rng.choice(green_idx, min(25000, len(green_idx)), replace=False)

    is_lower = pts[:, 1] < -0.10
    lower_idx = np.where(is_lower)[0]
    lower_sub_idx = rng.choice(lower_idx, min(25000, len(lower_idx)), replace=False)

    is_roof = pts[:, 1] > 0.0
    roof_idx = np.where(is_roof)[0]
    roof_sub_idx = rng.choice(roof_idx, min(25000, len(roof_idx)), replace=False)

    print(f"    Evaluation sets: Sample={len(sample_pc_idx):,}, Green={len(green_sub_idx):,}, Ground={len(lower_sub_idx):,}, Roof={len(roof_sub_idx):,}")

    # Check if raw Poisson already exists in STAGES_DIR
    raw_mesh_path = STAGES_DIR / "stage_A_raw_poisson.ply"
    raw_dens_path = STAGES_DIR / "stage_A_densities.npy"

    if raw_mesh_path.exists() and raw_dens_path.exists():
        print(f"\n[*] Loading existing Raw Poisson Mesh from: {raw_mesh_path.name}")
        mesh_A = o3d.io.read_triangle_mesh(str(raw_mesh_path))
        dens = np.load(raw_dens_path)
    else:
        print("\n[*] Running Screened Poisson Surface Reconstruction (Depth=10, Scale=1.1)...")
        cam_mean = np.array([-0.25, 0.02, -0.05])
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.035, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=cam_mean)

        t0 = time.perf_counter()
        mesh_A, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, scale=1.1, linear_fit=False
        )
        dens = np.asarray(densities)
        print(f"    Raw Poisson completed in {time.perf_counter() - t0:.2f}s")
        o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh_A)
        np.save(raw_dens_path, dens)
        print(f"    Saved: {raw_mesh_path.name}")

    stages_results = {}
    renders_map = {}

    # -------------------------------------------------------------------------
    # STAGE A: Raw Poisson
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage A: Raw Poisson Output ---")
    stages_results["Stage_A"] = evaluate_mesh_stage(
        "Stage A (Raw Poisson)", mesh_A, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_A"] = render_mesh_views(mesh_A, "stage_A_raw", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE B: Density Trimmed Only
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage B: After Poisson Density Trimming Only (q=0.020) ---")
    mesh_B = copy.deepcopy(mesh_A)
    trim_mask = dens < np.quantile(dens, 0.020)
    mesh_B.remove_vertices_by_mask(trim_mask)
    mesh_B.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(STAGES_DIR / "stage_B_density_trimmed.ply"), mesh_B)

    stages_results["Stage_B"] = evaluate_mesh_stage(
        "Stage B (Density Trimmed)", mesh_B, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_B"] = render_mesh_views(mesh_B, "stage_B_density", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE C: Vertex Support-Distance Filtered Only (< 22 mm)
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage C: After Vertex Support-Distance Filtering (d > 22mm removed) ---")
    mesh_C = copy.deepcopy(mesh_B)
    mv_C = np.asarray(mesh_C.vertices)
    dists_C, _ = tree_pc.query(mv_C, k=1, workers=-1)
    mesh_C.remove_vertices_by_mask(dists_C > 0.022)
    mesh_C.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(STAGES_DIR / "stage_C_support_filtered.ply"), mesh_C)

    stages_results["Stage_C"] = evaluate_mesh_stage(
        "Stage C (Support Distance Filtered)", mesh_C, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_C"] = render_mesh_views(mesh_C, "stage_C_support", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE D: Long Triangles Filtered Only (max_e > 50 mm)
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage D: After Long-Triangle Filtering (max_e > 50mm removed) ---")
    mesh_D = copy.deepcopy(mesh_C)
    mt_D = np.asarray(mesh_D.triangles)
    mv_D = np.asarray(mesh_D.vertices)
    e1 = np.linalg.norm(mv_D[mt_D[:, 1]] - mv_D[mt_D[:, 0]], axis=1)
    e2 = np.linalg.norm(mv_D[mt_D[:, 2]] - mv_D[mt_D[:, 1]], axis=1)
    e3 = np.linalg.norm(mv_D[mt_D[:, 0]] - mv_D[mt_D[:, 2]], axis=1)
    max_e = np.maximum(np.maximum(e1, e2), e3)
    mesh_D.remove_triangles_by_mask(max_e > 0.050)
    mesh_D.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(STAGES_DIR / "stage_D_long_triangles_removed.ply"), mesh_D)

    stages_results["Stage_D"] = evaluate_mesh_stage(
        "Stage D (Long Triangles Removed)", mesh_D, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_D"] = render_mesh_views(mesh_D, "stage_D_long_tris", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE E: Non-Manifold Topology Cleanup Only
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage E: After Non-Manifold Cleanup Only ---")
    mesh_E = copy.deepcopy(mesh_D)
    mesh_E.remove_non_manifold_edges()
    mesh_E.remove_duplicated_vertices()
    mesh_E.remove_duplicated_triangles()
    mesh_E.remove_degenerate_triangles()
    mesh_E.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(STAGES_DIR / "stage_E_non_manifold_cleaned.ply"), mesh_E)

    stages_results["Stage_E"] = evaluate_mesh_stage(
        "Stage E (Non-Manifold Cleaned)", mesh_E, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_E"] = render_mesh_views(mesh_E, "stage_E_topology", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE F: Connected Component Removal Only (< 60 faces)
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage F: After Connected Component Removal (< 60 faces) ---")
    mesh_F = copy.deepcopy(mesh_E)
    clusters, num_faces, _ = mesh_F.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    num_faces = np.asarray(num_faces)
    keep_mask = np.isin(clusters, np.where(num_faces >= 60)[0])
    mesh_F.remove_triangles_by_mask(~keep_mask)
    mesh_F.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(STAGES_DIR / "stage_F_components_filtered.ply"), mesh_F)

    stages_results["Stage_F"] = evaluate_mesh_stage(
        "Stage F (Small Components Removed)", mesh_F, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_F"] = render_mesh_views(mesh_F, "stage_F_components", DIAG_DIR)

    # -------------------------------------------------------------------------
    # STAGE G: Final V4 Mesh on Disk
    # -------------------------------------------------------------------------
    print("\n--- Auditing Stage G: Final V4 Result on Disk for Reference ---")
    mesh_G = o3d.io.read_triangle_mesh(str(V4_MESH_PATH))
    stages_results["Stage_G"] = evaluate_mesh_stage(
        "Stage G (Final V4 Reference)", mesh_G, pts, colors, tree_pc,
        sample_pc_idx, green_sub_idx, lower_sub_idx, roof_sub_idx
    )
    renders_map["Stage_G"] = render_mesh_views(mesh_G, "stage_G_v4_ref", DIAG_DIR)

    # Save JSON Report
    audit_json_path = DIAG_DIR / "stage_by_stage_audit_report.json"
    with open(audit_json_path, "w") as f:
        json.dump(stages_results, f, indent=2)
    print(f"\n[*] Audit report saved: {audit_json_path}")

    # Build Multi-Panel Comparison Card: geometry_recovery_stage_comparison.png
    print("\n[*] Assembling Comprehensive Stage Comparison Card...")
    stages = ["Stage_A", "Stage_B", "Stage_C", "Stage_D", "Stage_E", "Stage_F", "Stage_G"]
    stage_titles = [
        "Stage A: Raw Poisson (D=10)",
        "Stage B: Density Trim (q=0.02)",
        "Stage C: Support Filter (<22mm)",
        "Stage D: Long Edge Filter (<50mm)",
        "Stage E: Non-Manifold Clean",
        "Stage F: Comp Filter (<60f)",
        "Stage G: V4 Canonical Reference"
    ]

    card_w, card_h = 3840, 2160
    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    try:
        font_title = ImageFont.truetype("arial.ttf", 48)
        font_sub = ImageFont.truetype("arial.ttf", 26)
        font_col = ImageFont.truetype("arial.ttf", 24)
        font_metric = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_col = ImageFont.load_default()
        font_metric = ImageFont.load_default()

    # Header
    draw.text((60, 40), "F.O.R.G.E V5 — STAGE-BY-STAGE RECONSTRUCTION AUDIT", fill=(10, 15, 25), font=font_title)
    draw.text((60, 100), "Decomposition of V4 geometry loss across Poisson density, support-distance pruning, edge length, and topology filtering.", fill=(80, 90, 105), font=font_sub)

    # 7 columns for the 7 stages, 2 rows of renders (Top: Nadir, Bottom: Oblique)
    col_w = int((card_w - 120) / 7)
    y_top_render = 160
    render_h = 750

    for col_idx, (st_key, st_title) in enumerate(zip(stages, stage_titles)):
        x_left = 60 + col_idx * col_w
        x_right = x_left + col_w - 15

        # Stage Header
        draw.rectangle([(x_left, y_top_render), (x_right, y_top_render + 60)], fill=(245, 247, 250), outline=(215, 222, 232), width=1)
        draw.text((x_left + 10, y_top_render + 10), st_title.split(":")[0], fill=(20, 30, 45), font=font_col)
        draw.text((x_left + 10, y_top_render + 35), st_title.split(":")[1].strip(), fill=(100, 110, 125), font=font_metric)

        # Nadir Render
        nadir_img_path = renders_map[st_key]["nadir"]
        if nadir_img_path.exists():
            img_n = Image.open(nadir_img_path).resize((col_w - 15, int(render_h * 0.48)), Image.Resampling.LANCZOS)
            card.paste(img_n, (x_left, y_top_render + 65))
            draw.rectangle([(x_left, y_top_render + 65), (x_right, y_top_render + 65 + int(render_h * 0.48))], outline=(230, 235, 240), width=1)

        # Oblique Render
        oblique_img_path = renders_map[st_key]["oblique"]
        y_ob = y_top_render + 70 + int(render_h * 0.48)
        if oblique_img_path.exists():
            img_ob = Image.open(oblique_img_path).resize((col_w - 15, int(render_h * 0.48)), Image.Resampling.LANCZOS)
            card.paste(img_ob, (x_left, y_ob))
            draw.rectangle([(x_left, y_ob), (x_right, y_ob + int(render_h * 0.48))], outline=(230, 235, 240), width=1)

        # Metrics Card below renders
        st_data = stages_results[st_key]
        y_box = y_ob + int(render_h * 0.48) + 15
        box_h = card_h - y_box - 40
        draw.rectangle([(x_left, y_box), (x_right, y_box + box_h)], fill=(250, 252, 254), outline=(215, 222, 232), width=1)

        lines = [
            f"Verts: {st_data['num_vertices']:,}",
            f"Tris: {st_data['num_triangles']:,}",
            f"Area: {st_data['total_surface_area_m2']} m²",
            f"Comps: {st_data['components_count']} (top: {st_data['largest_component_pct']}%)",
            f"Bound Edges: {st_data['boundary_edges']:,}",
            f"Non-Manifold: {st_data['true_non_manifold_edges']}",
            f"Long Tris (>50mm): {st_data['long_triangles_gt_50mm']:,}",
            "---",
            f"Extent Y (height): {st_data['extent_xyz'][1]} m",
            f"PC Med: {st_data['metrics_overall_pc2mesh']['median_mm']} mm",
            f"PC <=10mm: {st_data['metrics_overall_pc2mesh']['pct_within_10mm']}%",
            f"Ground Med: {st_data['metrics_regional_pc2mesh']['ground_lower_terrain']['median_mm']} mm",
            f"Green Med: {st_data['metrics_regional_pc2mesh']['green_vegetation']['median_mm']} mm",
            f"Roof Med: {st_data['metrics_regional_pc2mesh']['rooftop_fixtures']['median_mm']} mm",
            "---",
            f"Surf>20mm: {st_data['metrics_mesh_surface2pc']['pct_unsupported_gt_20mm']}%",
            f"Surf>50mm: {st_data['metrics_mesh_surface2pc']['pct_unsupported_gt_50mm']}%",
        ]
        y_text = y_box + 12
        for l in lines:
            if l == "---":
                draw.line([(x_left + 10, y_text + 4), (x_right - 10, y_text + 4)], fill=(210, 218, 228), width=1)
                y_text += 10
            else:
                draw.text((x_left + 10, y_text), l, fill=(40, 50, 65), font=font_metric)
                y_text += 22

    comparison_card_path = V5_DIR / "geometry_recovery_stage_comparison.png"
    card.save(comparison_card_path, quality=95)
    print(f"[*] Comparison card saved: {comparison_card_path}")

    # Copy to DIAG_DIR as well
    diag_card_path = DIAG_DIR / "geometry_recovery_stage_comparison.png"
    card.save(diag_card_path, quality=95)

    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE AUDIT SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Stage':<18} | {'Verts':<9} | {'Tris':<9} | {'Area(m²)':<8} | {'Comps':<6} | {'Boundary':<8} | {'Y Ext(m)':<8} | {'Surf>20mm':<9}")
    print("-" * 88)
    for st_key in stages:
        d = stages_results[st_key]
        print(f"{d['name']:<18} | {d['num_vertices']:<9,d} | {d['num_triangles']:<9,d} | {d['total_surface_area_m2']:<8.2f} | {d['components_count']:<6d} | {d['boundary_edges']:<8,d} | {d['extent_xyz'][1]:<8.3f} | {d['metrics_mesh_surface2pc']['pct_unsupported_gt_20mm']:<8.1f}%")

if __name__ == "__main__":
    main()
