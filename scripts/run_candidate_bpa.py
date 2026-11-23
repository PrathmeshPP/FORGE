#!/usr/bin/env python3
import time, json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V3_DIR = CAMPAIGN_DIR / "final_best_canonical_v3"
V4_DIR = CAMPAIGN_DIR / "final_best_canonical_v4"
CAND_DIR = V4_DIR / "candidates"
DIAG_DIR = V4_DIR / "diagnostics"
PCD_PATH = V3_DIR / "final_best_canonical_v3_point_cloud.ply"
OUT_SUMMARY = DIAG_DIR / "stage3_meshing_benchmark.json"

print("Loading point cloud for BPA...", flush=True)
pcd = o3d.io.read_point_cloud(str(PCD_PATH))
pts = np.asarray(pcd.points)
tree_pc = cKDTree(pts)

print("Running Multi-Scale BPA with calibrated radii [2mm, 4mm, 8mm, 16mm]...", flush=True)
t0 = time.time()
radii = o3d.utility.DoubleVector([0.002, 0.004, 0.008, 0.016])
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
t_bpa = time.time() - t0
print(f"BPA meshing complete in {t_bpa:.1f}s", flush=True)

c3_path = CAND_DIR / "candidate_bpa_multiscale.ply"
o3d.io.write_triangle_mesh(str(c3_path), mesh_bpa)
print(f"Saved BPA mesh to: {c3_path}", flush=True)

# Evaluate
verts = np.asarray(mesh_bpa.vertices)
tris = np.asarray(mesh_bpa.triangles)
nme_true = len(mesh_bpa.get_non_manifold_edges(allow_boundary_edges=True))
nme_all = len(mesh_bpa.get_non_manifold_edges(allow_boundary_edges=False))
boundary = nme_all - nme_true

clusters, cluster_counts, _ = mesh_bpa.cluster_connected_triangles()
cluster_counts = np.asarray(cluster_counts)
num_comp = len(cluster_counts)
comp_sorted = sorted(cluster_counts.tolist(), reverse=True)
largest_comp = comp_sorted[0] if comp_sorted else 0
largest_pct = round(largest_comp / len(tris) * 100, 2)

tree_mesh = cKDTree(verts)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(pts), min(100000, len(pts)), replace=False)
sample_pts = pts[sample_idx]
dists_pc2mesh, _ = tree_mesh.query(sample_pts, k=1, workers=-1)

pc_median_mm = round(float(np.median(dists_pc2mesh)) * 1000, 3)
pc_mean_mm = round(float(np.mean(dists_pc2mesh)) * 1000, 3)
w5mm = round(float(np.mean(dists_pc2mesh <= 0.005)) * 100, 2)
w10mm = round(float(np.mean(dists_pc2mesh <= 0.010)) * 100, 2)
w20mm = round(float(np.mean(dists_pc2mesh <= 0.020)) * 100, 2)
b20mm = round(float(np.mean(dists_pc2mesh > 0.020)) * 100, 2)
b50mm = round(float(np.mean(dists_pc2mesh > 0.050)) * 100, 2)

dists_mesh2pc, _ = tree_pc.query(verts, k=1, workers=-1)
m_median_mm = round(float(np.median(dists_mesh2pc)) * 1000, 3)
m_b20mm = round(float(np.mean(dists_mesh2pc > 0.020)) * 100, 2)

bbox = mesh_bpa.get_axis_aligned_bounding_box()

stats_bpa = {
    "candidate": "Candidate_MultiScale_BPA",
    "runtime_seconds": round(t_bpa, 2),
    "vertices": len(verts),
    "triangles": len(tris),
    "true_non_manifold_edges": nme_true,
    "boundary_edges": boundary,
    "watertight": mesh_bpa.is_watertight(),
    "connected_components": num_comp,
    "largest_component_triangles": largest_comp,
    "largest_component_percent": largest_pct,
    "top_5_component_sizes": comp_sorted[:5],
    "pc_to_mesh_distance": {
        "median_mm": pc_median_mm,
        "mean_mm": pc_mean_mm,
        "within_5mm_pct": w5mm,
        "within_10mm_pct": w10mm,
        "within_20mm_pct": w20mm,
        "beyond_20mm_pct": b20mm,
        "beyond_50mm_pct": b50mm,
    },
    "mesh_to_pc_distance": {
        "median_mm": m_median_mm,
        "beyond_20mm_pct": m_b20mm,
    },
    "bounding_box": {
        "min": bbox.min_bound.round(4).tolist(),
        "max": bbox.max_bound.round(4).tolist(),
        "extents": bbox.get_extent().round(4).tolist()
    }
}

print(f"BPA Results: Verts={len(verts):,}, Tris={len(tris):,}, Boundary={boundary:,}, Comp={num_comp}, PC<=5mm={w5mm}%, PC<=10mm={w10mm}%, >20mm={b20mm}%", flush=True)

with open(OUT_SUMMARY) as f:
    summary = json.load(f)
summary["Candidate_MultiScale_BPA"] = stats_bpa
with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)
print("Updated stage3_meshing_benchmark.json with BPA results!", flush=True)
