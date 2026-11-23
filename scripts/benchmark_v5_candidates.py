#!/usr/bin/env python3
"""
F.O.R.G.E V5 — Candidate Benchmarking & Evaluation

Benchmarks:
- Candidate 1: Corrected Poisson (Depth 10) with multi-view camera ray normals,
  gentle density trimming (q=0.010), support-aware face centroid filtering (<38mm),
  and conservative component preservation (>=20 faces).
- Candidate 2: 3D Alpha Shape (alpha=0.020) on multi-view consistent points,
  with non-manifold cleanup.
- Baseline V4: V4 canonical reference for direct comparison.

Evaluates:
- Point-cloud-to-mesh surface distances (overall, ground, vegetation, roof).
- Mesh-surface-to-point-cloud distances (sampled over triangle surfaces).
- Surface area and spatial coverage.
- Boundary edges and non-manifold edges.
- Number of components and component retention.
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
CAND_DIR = V5_DIR / "candidates"
DIAG_DIR = V5_DIR / "diagnostics"
CAND_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PCD_PATH = V3_DIR / "final_best_canonical_v3_point_cloud.ply"
V4_MESH_PATH = V4_DIR / "final_best_canonical_v4_mesh.ply"
POSES_PATH = PROJECT_ROOT / "outputs/test_4k_dense_building_window/predictions/camera_poses.npz"

def render_mesh_preview(mesh: o3d.geometry.TriangleMesh, out_path: Path, view_type: str = "oblique"):
    bbox = mesh.get_axis_aligned_bounding_box()
    ctr = bbox.get_center()

    views = {
        "nadir": {"front": [0.0, 1.0, 0.001], "up": [0.0, 0.0, -1.0], "zoom": 0.70},
        "oblique": {"front": [0.35, 0.55, 0.76], "up": [0.0, 1.0, 0.0], "zoom": 0.65},
        "roof": {"front": [0.20, 0.45, 0.87], "up": [0.0, 1.0, 0.0], "zoom": 0.35}
    }
    vcfg = views.get(view_type, views["oblique"])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Render_{out_path.stem}", width=1280, height=960, visible=False)
    vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
    vis.get_render_option().light_on = True
    vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    vis.get_render_option().show_coordinate_frame = False

    m_render = copy.deepcopy(mesh)
    m_render.compute_vertex_normals()
    if not m_render.has_vertex_colors() or len(m_render.vertex_colors) == 0:
        m_render.paint_uniform_color([0.78, 0.82, 0.88])

    vis.add_geometry(m_render)
    ctr_ctrl = vis.get_view_control()
    ctr_ctrl.set_lookat(ctr)
    ctr_ctrl.set_front(vcfg["front"])
    ctr_ctrl.set_up(vcfg["up"])
    ctr_ctrl.set_zoom(vcfg["zoom"])
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()

def evaluate_mesh(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    pts: np.ndarray,
    tree_pc: cKDTree,
    sample_pc_idx: np.ndarray,
    green_idx: np.ndarray,
    lower_idx: np.ndarray,
    roof_idx: np.ndarray
) -> dict:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n_v, n_t = len(verts), len(tris)

    if n_v == 0 or n_t == 0:
        return {"name": name, "error": "Empty mesh"}

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

    # Surface area
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_area = float(np.sum(tri_areas))

    # PC -> Mesh Distance
    tree_mesh = cKDTree(verts)
    d_all, _ = tree_mesh.query(pts[sample_pc_idx], k=1, workers=-1)
    d_green, _ = tree_mesh.query(pts[green_idx], k=1, workers=-1)
    d_lower, _ = tree_mesh.query(pts[lower_idx], k=1, workers=-1)
    d_roof, _ = tree_mesh.query(pts[roof_idx], k=1, workers=-1)

    # Mesh Surface -> PC Distance (uniform surface sampling)
    pcd_surf = mesh.sample_points_uniformly(number_of_points=min(50000, n_t * 3))
    surf_pts = np.asarray(pcd_surf.points)
    d_surf2pc, _ = tree_pc.query(surf_pts, k=1, workers=-1)

    return {
        "name": name,
        "num_vertices": n_v,
        "num_triangles": n_t,
        "total_surface_area_m2": round(total_area, 3),
        "components_count": num_comp,
        "largest_component_pct": largest_pct,
        "true_non_manifold_edges": nme_true,
        "boundary_edges": boundary,
        "extent_xyz": [round(float(x), 4) for x in (verts.max(axis=0) - verts.min(axis=0))],
        "pc2mesh_distance_mm": {
            "overall_median": round(float(np.median(d_all)) * 1000, 2),
            "overall_p95": round(float(np.percentile(d_all, 95)) * 1000, 2),
            "overall_pct_within_10mm": round(float(np.mean(d_all <= 0.010)) * 100, 2),
            "overall_pct_within_20mm": round(float(np.mean(d_all <= 0.020)) * 100, 2),
            "ground_median": round(float(np.median(d_lower)) * 1000, 2),
            "ground_pct_within_10mm": round(float(np.mean(d_lower <= 0.010)) * 100, 2),
            "green_median": round(float(np.median(d_green)) * 1000, 2),
            "green_pct_within_10mm": round(float(np.mean(d_green <= 0.010)) * 100, 2),
            "roof_median": round(float(np.median(d_roof)) * 1000, 2),
            "roof_pct_within_10mm": round(float(np.mean(d_roof <= 0.010)) * 100, 2)
        },
        "mesh_surface2pc_distance_mm": {
            "median": round(float(np.median(d_surf2pc)) * 1000, 2),
            "p95": round(float(np.percentile(d_surf2pc, 95)) * 1000, 2),
            "pct_unsupported_gt_20mm": round(float(np.mean(d_surf2pc > 0.020)) * 100, 2),
            "pct_unsupported_gt_50mm": round(float(np.mean(d_surf2pc > 0.050)) * 100, 2)
        }
    }

def main():
    print("=" * 80)
    print("F.O.R.G.E V5 — CANDIDATE BENCHMARKING (CANDIDATE 1 vs CANDIDATE 2 vs V4)")
    print("=" * 80)

    # 1. Load PCD
    print(f"[*] Loading point cloud: {PCD_PATH}")
    pcd = o3d.io.read_point_cloud(str(PCD_PATH))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    tree_pc = cKDTree(pts)

    # Sampling indices
    rng = np.random.default_rng(42)
    sample_pc_idx = rng.choice(len(pts), min(100000, len(pts)), replace=False)
    is_green = (colors[:, 1] > colors[:, 0] * 1.05) & (colors[:, 1] > colors[:, 2] * 1.05)
    green_idx = rng.choice(np.where(is_green)[0], min(25000, np.sum(is_green)), replace=False)
    is_lower = pts[:, 1] < -0.10
    lower_idx = rng.choice(np.where(is_lower)[0], min(25000, np.sum(is_lower)), replace=False)
    is_roof = pts[:, 1] > 0.0
    roof_idx = rng.choice(np.where(is_roof)[0], min(25000, np.sum(is_roof)), replace=False)

    # Load Camera Centers for Ray Orientation
    poses = np.load(POSES_PATH)
    ext_cv = poses['extrinsics']
    T_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])
    cam_centers = np.array([T_CV_TO_GLTF @ (-ext_cv[i, :3, :3].T @ ext_cv[i, :3, 3]) for i in range(len(ext_cv))])

    # -------------------------------------------------------------------------
    # BUILD CANDIDATE 1: Corrected Poisson with Support-Aware Face Pruning
    # -------------------------------------------------------------------------
    print("\n[+] Building Candidate 1: Corrected Screened Poisson (D=10, Support-Aware Pruning)...")
    pcd_cand1 = copy.deepcopy(pcd)
    pcd_cand1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.035, max_nn=30))
    n_c1 = np.asarray(pcd_cand1.normals)

    # Orient towards nearest camera center
    tree_cams = cKDTree(cam_centers)
    _, nearest_cam_idx = tree_cams.query(pts, k=1)
    ray_to_cam = cam_centers[nearest_cam_idx] - pts
    ray_to_cam /= np.maximum(np.linalg.norm(ray_to_cam, axis=1, keepdims=True), 1e-6)
    dots = np.sum(n_c1 * ray_to_cam, axis=1)
    n_c1[dots < 0] *= -1

    # Gravity-sky consistency for horizontal ground/roof surfaces (|Ny| > 0.4)
    is_horiz = np.abs(n_c1[:, 1]) > 0.4
    n_c1[is_horiz & (n_c1[:, 1] < 0)] *= -1
    pcd_cand1.normals = o3d.utility.Vector3dVector(n_c1)

    t0 = time.perf_counter()
    mesh_c1, dens_c1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_cand1, depth=10, scale=1.1, linear_fit=False
    )
    print(f"    Poisson solved in {time.perf_counter() - t0:.2f}s")

    # 1. Calibrated Density Trim (q = 0.010 removes sky balloon while preserving ground)
    dens = np.asarray(dens_c1)
    mesh_c1.remove_vertices_by_mask(dens < np.quantile(dens, 0.010))
    mesh_c1.remove_unreferenced_vertices()

    # 2. Support-Aware Face Filtering:
    # Instead of deleting single vertices > 22mm, evaluate each face centroid:
    # Keep face if centroid is within 36mm of point cloud (adaptive to sparse ground/vegetation)
    mv_c1 = np.asarray(mesh_c1.vertices)
    mt_c1 = np.asarray(mesh_c1.triangles)
    centroids = (mv_c1[mt_c1[:, 0]] + mv_c1[mt_c1[:, 1]] + mv_c1[mt_c1[:, 2]]) / 3.0
    c_dists, _ = tree_pc.query(centroids, k=1, workers=-1)

    # Face test: keep if centroid <= 0.036m (36mm)
    # Also clamp max edge to 0.075m (75mm) if centroid > 0.025m
    v0 = mv_c1[mt_c1[:, 0]]
    v1 = mv_c1[mt_c1[:, 1]]
    v2 = mv_c1[mt_c1[:, 2]]
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e1, e2), e3)

    keep_face_mask = (c_dists <= 0.036) & ~((max_e > 0.075) & (c_dists > 0.025))
    mesh_c1.remove_triangles_by_mask(~keep_face_mask)
    mesh_c1.remove_unreferenced_vertices()

    # 3. Non-manifold cleanup
    mesh_c1.remove_non_manifold_edges()
    mesh_c1.remove_duplicated_vertices()
    mesh_c1.remove_duplicated_triangles()
    mesh_c1.remove_degenerate_triangles()
    mesh_c1.remove_unreferenced_vertices()

    # 4. Conservative Component Preservation:
    # Filter only tiny floating specks (< 20 faces), preserving genuine disconnected vegetation / structures
    clusters, num_faces, _ = mesh_c1.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    num_faces = np.asarray(num_faces)
    keep_comp_mask = np.isin(clusters, np.where(num_faces >= 20)[0])
    mesh_c1.remove_triangles_by_mask(~keep_comp_mask)
    mesh_c1.remove_unreferenced_vertices()
    mesh_c1.compute_vertex_normals()

    cand1_path = CAND_DIR / "candidate1_corrected_poisson_support_aware.ply"
    o3d.io.write_triangle_mesh(str(cand1_path), mesh_c1)
    print(f"    Saved Candidate 1: {cand1_path.name} (Verts={len(mesh_c1.vertices):,}, Tris={len(mesh_c1.triangles):,})")

    # -------------------------------------------------------------------------
    # BUILD CANDIDATE 2: 3D Alpha Shape (alpha=0.025)
    # -------------------------------------------------------------------------
    print("\n[+] Building Candidate 2: 3D Alpha Shape (alpha=0.025)...")
    pcd_cand2 = pcd.voxel_down_sample(voxel_size=0.003)
    t0 = time.perf_counter()
    mesh_c2 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_cand2, alpha=0.025)
    print(f"    Alpha Shape built in {time.perf_counter() - t0:.2f}s")

    mesh_c2.remove_non_manifold_edges()
    mesh_c2.remove_duplicated_vertices()
    mesh_c2.remove_duplicated_triangles()
    mesh_c2.remove_degenerate_triangles()
    mesh_c2.remove_unreferenced_vertices()

    # Prune tiny specks
    clusters2, num_faces2, _ = mesh_c2.cluster_connected_triangles()
    clusters2 = np.asarray(clusters2)
    num_faces2 = np.asarray(num_faces2)
    keep_comp2 = np.isin(clusters2, np.where(num_faces2 >= 20)[0])
    mesh_c2.remove_triangles_by_mask(~keep_comp2)
    mesh_c2.remove_unreferenced_vertices()
    mesh_c2.compute_vertex_normals()

    cand2_path = CAND_DIR / "candidate2_alpha_shape.ply"
    o3d.io.write_triangle_mesh(str(cand2_path), mesh_c2)
    print(f"    Saved Candidate 2: {cand2_path.name} (Verts={len(mesh_c2.vertices):,}, Tris={len(mesh_c2.triangles):,})")

    # -------------------------------------------------------------------------
    # LOAD BASELINE V4 REFERENCE
    # -------------------------------------------------------------------------
    print("\n[+] Loading Baseline V4 Reference Mesh...")
    mesh_v4 = o3d.io.read_triangle_mesh(str(V4_MESH_PATH))

    # -------------------------------------------------------------------------
    # EVALUATE ALL CANDIDATES
    # -------------------------------------------------------------------------
    print("\n[*] Rigorously Evaluating Candidates...")
    res_c1 = evaluate_mesh("Candidate 1 (Corrected Poisson Support-Aware)", mesh_c1, pts, tree_pc, sample_pc_idx, green_idx, lower_idx, roof_idx)
    res_c2 = evaluate_mesh("Candidate 2 (Alpha Shape 0.025)", mesh_c2, pts, tree_pc, sample_pc_idx, green_idx, lower_idx, roof_idx)
    res_v4 = evaluate_mesh("Baseline V4 Reference", mesh_v4, pts, tree_pc, sample_pc_idx, green_idx, lower_idx, roof_idx)

    cand_results = {
        "candidate_1_corrected_poisson": res_c1,
        "candidate_2_alpha_shape": res_c2,
        "baseline_v4": res_v4
    }

    eval_json_path = DIAG_DIR / "candidates_rigorous_evaluation.json"
    with open(eval_json_path, "w") as f:
        json.dump(cand_results, f, indent=2)
    print(f"    Evaluation saved: {eval_json_path}")

    # Render Previews for Comparison
    render_mesh_preview(mesh_c1, DIAG_DIR / "cand1_oblique.png", "oblique")
    render_mesh_preview(mesh_c1, DIAG_DIR / "cand1_nadir.png", "nadir")
    render_mesh_preview(mesh_c2, DIAG_DIR / "cand2_oblique.png", "oblique")
    render_mesh_preview(mesh_c2, DIAG_DIR / "cand2_nadir.png", "nadir")
    render_mesh_preview(mesh_v4, DIAG_DIR / "v4_oblique.png", "oblique")
    render_mesh_preview(mesh_v4, DIAG_DIR / "v4_nadir.png", "nadir")

    print("\n" + "=" * 80)
    print("CANDIDATE COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Candidate':<38} | {'Verts':<9} | {'Tris':<9} | {'Area(m²)':<8} | {'Boundary':<8} | {'PC Med(mm)':<10} | {'Surf>20mm':<9}")
    print("-" * 105)
    for c_key, c_val in [("Candidate 1 (Corrected Poisson)", res_c1), ("Candidate 2 (Alpha Shape)", res_c2), ("Baseline V4", res_v4)]:
        print(f"{c_key:<38} | {c_val['num_vertices']:<9,d} | {c_val['num_triangles']:<9,d} | {c_val['total_surface_area_m2']:<8.2f} | {c_val['boundary_edges']:<8,d} | {c_val['pc2mesh_distance_mm']['overall_median']:<10.2f} | {c_val['mesh_surface2pc_distance_mm']['pct_unsupported_gt_20mm']:<8.2f}%")

if __name__ == "__main__":
    main()
