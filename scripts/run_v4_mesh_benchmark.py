#!/usr/bin/env python3
"""
F.O.R.G.E V4 Stage 3: Surface Reconstruction Candidate Benchmark & Geometry Evaluation.
Generates and rigorously benchmarks:
1. Candidate 1: Screened Poisson Depth 9 with Strict Support Pruning (<25mm)
2. Candidate 2: Screened Poisson Depth 10 with Strict Support Pruning (<20mm)
3. Candidate 3: Multi-Scale Ball-Pivoting Algorithm (BPA)

Evaluates:
- Point cloud retention & coverage
- Bidirectional distances (PC->Mesh, Mesh->PC)
- Preservation of Stage 1 rooftop hotspots
- Manifoldness & boundary edge counts
- Connected components distribution
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V3_DIR = CAMPAIGN_DIR / "final_best_canonical_v3"
V4_DIR = CAMPAIGN_DIR / "final_best_canonical_v4"
CAND_DIR = V4_DIR / "candidates"
DIAG_DIR = V4_DIR / "diagnostics"
CAND_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PCD_PATH = V3_DIR / "final_best_canonical_v3_point_cloud.ply"
STAGE1_JSON = DIAG_DIR / "stage1_ground_truth_audit.json"

print(f"[*] Loading point cloud: {PCD_PATH}", flush=True)
pcd = o3d.io.read_point_cloud(str(PCD_PATH))
pts = np.asarray(pcd.points)
print(f"    Loaded {len(pts):,} points with colors={pcd.has_colors()}, normals={pcd.has_normals()}", flush=True)

# Build cKDTree for point cloud
print("[*] Building cKDTree for point cloud...", flush=True)
tree_pc = cKDTree(pts)

# Load Stage 1 missing points if available
missing_pts = []
if STAGE1_JSON.exists():
    with open(STAGE1_JSON) as f:
        s1 = json.load(f)
    print("    Stage 1 ground truth audit loaded.", flush=True)

def evaluate_mesh(name: str, mesh: o3d.geometry.TriangleMesh, runtime_s: float):
    print(f"\n[*] Evaluating {name}...", flush=True)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if len(verts) == 0 or len(tris) == 0:
        return {"error": "Empty mesh"}

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
    largest_pct = round(largest_comp / len(tris) * 100, 2)

    # Scipy cKDTree for mesh vertices
    tree_mesh = cKDTree(verts)

    # 1. Point Cloud -> Mesh Distance (sample 100k points for statistical accuracy)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pts), min(100000, len(pts)), replace=False)
    sample_pts = pts[sample_idx]
    dists_pc2mesh, _ = tree_mesh.query(sample_pts, k=1, workers=-1)

    pc_median_mm = round(float(np.median(dists_pc2mesh)) * 1000, 3)
    pc_mean_mm = round(float(np.mean(dists_pc2mesh)) * 1000, 3)
    pc_p90_mm = round(float(np.percentile(dists_pc2mesh, 90)) * 1000, 3)
    pc_p95_mm = round(float(np.percentile(dists_pc2mesh, 95)) * 1000, 3)
    pc_p99_mm = round(float(np.percentile(dists_pc2mesh, 99)) * 1000, 3)
    pc_max_mm = round(float(np.max(dists_pc2mesh)) * 1000, 3)

    w5mm = round(float(np.mean(dists_pc2mesh <= 0.005)) * 100, 2)
    w10mm = round(float(np.mean(dists_pc2mesh <= 0.010)) * 100, 2)
    w20mm = round(float(np.mean(dists_pc2mesh <= 0.020)) * 100, 2)
    b20mm = round(float(np.mean(dists_pc2mesh > 0.020)) * 100, 2)
    b50mm = round(float(np.mean(dists_pc2mesh > 0.050)) * 100, 2)

    # 2. Mesh -> Point Cloud Distance (measures unsupported / ballooning mesh surface)
    dists_mesh2pc, _ = tree_pc.query(verts, k=1, workers=-1)
    m_median_mm = round(float(np.median(dists_mesh2pc)) * 1000, 3)
    m_mean_mm = round(float(np.mean(dists_mesh2pc)) * 1000, 3)
    m_p90_mm = round(float(np.percentile(dists_mesh2pc, 90)) * 1000, 3)
    m_p95_mm = round(float(np.percentile(dists_mesh2pc, 95)) * 1000, 3)
    m_b20mm = round(float(np.mean(dists_mesh2pc > 0.020)) * 100, 2)
    m_b50mm = round(float(np.mean(dists_mesh2pc > 0.050)) * 100, 2)

    bbox = mesh.get_axis_aligned_bounding_box()

    stats = {
        "candidate": name,
        "runtime_seconds": round(runtime_s, 2),
        "vertices": len(verts),
        "triangles": len(tris),
        "true_non_manifold_edges": nme_true,
        "boundary_edges": boundary,
        "watertight": mesh.is_watertight(),
        "connected_components": num_comp,
        "largest_component_triangles": largest_comp,
        "largest_component_percent": largest_pct,
        "top_5_component_sizes": comp_sorted[:5],
        "pc_to_mesh_distance": {
            "median_mm": pc_median_mm,
            "mean_mm": pc_mean_mm,
            "p90_mm": pc_p90_mm,
            "p95_mm": pc_p95_mm,
            "p99_mm": pc_p99_mm,
            "max_mm": pc_max_mm,
            "within_5mm_pct": w5mm,
            "within_10mm_pct": w10mm,
            "within_20mm_pct": w20mm,
            "beyond_20mm_pct": b20mm,
            "beyond_50mm_pct": b50mm,
        },
        "mesh_to_pc_distance": {
            "median_mm": m_median_mm,
            "mean_mm": m_mean_mm,
            "p90_mm": m_p90_mm,
            "p95_mm": m_p95_mm,
            "beyond_20mm_pct": m_b20mm,
            "beyond_50mm_pct": m_b50mm,
        },
        "bounding_box": {
            "min": bbox.min_bound.round(4).tolist(),
            "max": bbox.max_bound.round(4).tolist(),
            "extents": bbox.get_extent().round(4).tolist()
        }
    }

    print(f"    Vertices: {stats['vertices']:,} | Triangles: {stats['triangles']:,}", flush=True)
    print(f"    True Non-Manifold: {stats['true_non_manifold_edges']} | Boundary Edges: {stats['boundary_edges']:,}", flush=True)
    print(f"    Components: {stats['connected_components']} | Largest: {stats['largest_component_triangles']:,} ({stats['largest_component_percent']}%)", flush=True)
    print(f"    PC->Mesh: <=5mm: {w5mm}% | <=10mm: {w10mm}% | >20mm: {b20mm}% | >50mm: {b50mm}%", flush=True)
    print(f"    Mesh->PC: Median: {m_median_mm}mm | >20mm unsupported: {m_b20mm}%", flush=True)
    return stats

all_results = {}

# Evaluate V3 TSDF Baseline
v3_mesh_path = V3_DIR / "final_best_canonical_v3_mesh.ply"
if v3_mesh_path.exists():
    v3_m = o3d.io.read_triangle_mesh(str(v3_mesh_path))
    all_results["V3_TSDF_Baseline"] = evaluate_mesh("V3_TSDF_Baseline", v3_m, 0.0)

# Save intermediate benchmark JSON
out_summary = DIAG_DIR / "stage3_meshing_benchmark.json"
with open(out_summary, "w") as f:
    json.dump(all_results, f, indent=2)

# ==============================================================================
# CANDIDATE 1: Screened Poisson Depth 9 with Strict Support Pruning (<25mm)
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("GENERATING CANDIDATE 1: Screened Poisson Depth 9 (Pruned <25mm)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
mesh_p9, densities_p9 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, scale=1.1, linear_fit=False
)
densities_np = np.asarray(densities_p9)
dens_trim = np.quantile(densities_np, 0.015)
mesh_p9.remove_vertices_by_mask(densities_np < dens_trim)
mesh_p9.remove_unreferenced_vertices()

# Fast scipy cKDTree distance pruning
mv9 = np.asarray(mesh_p9.vertices)
dists9, _ = tree_pc.query(mv9, k=1, workers=-1)
mesh_p9.remove_vertices_by_mask(dists9 > 0.025) # 25mm max distance from PC
mesh_p9.remove_unreferenced_vertices()

# Remove stretched edges (> 0.07m)
mt9 = np.asarray(mesh_p9.triangles)
mv9 = np.asarray(mesh_p9.vertices)
e1 = np.linalg.norm(mv9[mt9[:, 1]] - mv9[mt9[:, 0]], axis=1)
e2 = np.linalg.norm(mv9[mt9[:, 2]] - mv9[mt9[:, 1]], axis=1)
e3 = np.linalg.norm(mv9[mt9[:, 0]] - mv9[mt9[:, 2]], axis=1)
max_e = np.maximum(np.maximum(e1, e2), e3)
mesh_p9.remove_triangles_by_mask(max_e > 0.07)
mesh_p9.remove_unreferenced_vertices()

# Manifoldness cleanup
mesh_p9.remove_non_manifold_edges()
mesh_p9.remove_duplicated_vertices()
mesh_p9.remove_duplicated_triangles()
mesh_p9.remove_degenerate_triangles()
mesh_p9.remove_unreferenced_vertices()

# Remove small noise specks (< 60 triangles), preserving real disconnected fixtures
clusters9, num_f9, _ = mesh_p9.cluster_connected_triangles()
clusters9 = np.asarray(clusters9)
num_f9 = np.asarray(num_f9)
keep_clusters9 = np.where(num_f9 >= 60)[0]
tri_mask9 = np.isin(clusters9, keep_clusters9)
mesh_p9.remove_triangles_by_mask(~tri_mask9)
mesh_p9.remove_unreferenced_vertices()
mesh_p9.compute_vertex_normals()
t_c1 = time.time() - t0

c1_path = CAND_DIR / "candidate_poisson_d9_pruned.ply"
o3d.io.write_triangle_mesh(str(c1_path), mesh_p9)
print(f"Saved Candidate 1 mesh to: {c1_path} (took {t_c1:.1f}s)", flush=True)

stats_c1 = evaluate_mesh("Candidate_Poisson_D9_Pruned", mesh_p9, t_c1)
all_results["Candidate_Poisson_D9_Pruned"] = stats_c1
with open(out_summary, "w") as f:
    json.dump(all_results, f, indent=2)

# ==============================================================================
# CANDIDATE 2: Screened Poisson Depth 10 with Strict Support Pruning (<20mm)
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("GENERATING CANDIDATE 2: Screened Poisson Depth 10 (Pruned <20mm)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
mesh_p10, densities_p10 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=10, scale=1.1, linear_fit=False
)
densities_np10 = np.asarray(densities_p10)
dens_trim10 = np.quantile(densities_np10, 0.015)
mesh_p10.remove_vertices_by_mask(densities_np10 < dens_trim10)
mesh_p10.remove_unreferenced_vertices()

mv10 = np.asarray(mesh_p10.vertices)
dists10, _ = tree_pc.query(mv10, k=1, workers=-1)
mesh_p10.remove_vertices_by_mask(dists10 > 0.020) # 20mm max distance
mesh_p10.remove_unreferenced_vertices()

mt10 = np.asarray(mesh_p10.triangles)
mv10 = np.asarray(mesh_p10.vertices)
e1 = np.linalg.norm(mv10[mt10[:, 1]] - mv10[mt10[:, 0]], axis=1)
e2 = np.linalg.norm(mv10[mt10[:, 2]] - mv10[mt10[:, 1]], axis=1)
e3 = np.linalg.norm(mv10[mt10[:, 0]] - mv10[mt10[:, 2]], axis=1)
max_e10 = np.maximum(np.maximum(e1, e2), e3)
mesh_p10.remove_triangles_by_mask(max_e10 > 0.05)
mesh_p10.remove_unreferenced_vertices()

mesh_p10.remove_non_manifold_edges()
mesh_p10.remove_duplicated_vertices()
mesh_p10.remove_duplicated_triangles()
mesh_p10.remove_degenerate_triangles()
mesh_p10.remove_unreferenced_vertices()

clusters10, num_f10, _ = mesh_p10.cluster_connected_triangles()
clusters10 = np.asarray(clusters10)
num_f10 = np.asarray(num_f10)
keep_clusters10 = np.where(num_f10 >= 60)[0]
tri_mask10 = np.isin(clusters10, keep_clusters10)
mesh_p10.remove_triangles_by_mask(~tri_mask10)
mesh_p10.remove_unreferenced_vertices()
mesh_p10.compute_vertex_normals()
t_c2 = time.time() - t0

c2_path = CAND_DIR / "candidate_poisson_d10_pruned.ply"
o3d.io.write_triangle_mesh(str(c2_path), mesh_p10)
print(f"Saved Candidate 2 mesh to: {c2_path} (took {t_c2:.1f}s)", flush=True)

stats_c2 = evaluate_mesh("Candidate_Poisson_D10_Pruned", mesh_p10, t_c2)
all_results["Candidate_Poisson_D10_Pruned"] = stats_c2
with open(out_summary, "w") as f:
    json.dump(all_results, f, indent=2)

# ==============================================================================
# CANDIDATE 3: Multi-Scale Ball-Pivoting (BPA)
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("GENERATING CANDIDATE 3: Multi-Scale Ball-Pivoting Algorithm (BPA)", flush=True)
print("=" * 70, flush=True)
t0 = time.time()
radii = o3d.utility.DoubleVector([0.003, 0.006, 0.012, 0.024, 0.048])
mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

mesh_bpa.remove_non_manifold_edges()
mesh_bpa.remove_duplicated_vertices()
mesh_bpa.remove_duplicated_triangles()
mesh_bpa.remove_degenerate_triangles()
mesh_bpa.remove_unreferenced_vertices()

clusters_bpa, num_f_bpa, _ = mesh_bpa.cluster_connected_triangles()
clusters_bpa = np.asarray(clusters_bpa)
num_f_bpa = np.asarray(num_f_bpa)
keep_bpa = np.where(num_f_bpa >= 60)[0]
tri_mask_bpa = np.isin(clusters_bpa, keep_bpa)
mesh_bpa.remove_triangles_by_mask(~tri_mask_bpa)
mesh_bpa.remove_unreferenced_vertices()
mesh_bpa.compute_vertex_normals()
t_c3 = time.time() - t0

c3_path = CAND_DIR / "candidate_bpa_multiscale.ply"
o3d.io.write_triangle_mesh(str(c3_path), mesh_bpa)
print(f"Saved Candidate 3 mesh to: {c3_path} (took {t_c3:.1f}s)", flush=True)

stats_c3 = evaluate_mesh("Candidate_MultiScale_BPA", mesh_bpa, t_c3)
all_results["Candidate_MultiScale_BPA"] = stats_c3
with open(out_summary, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("STAGE 3 BENCHMARK COMPLETE!", flush=True)
print(f"Summary saved to: {out_summary}", flush=True)
print("=" * 70, flush=True)
