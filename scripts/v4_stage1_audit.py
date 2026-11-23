"""
V4 Comprehensive Point-Cloud vs Mesh Audit
==========================================
Stage 1: Ground truth verification of existing V3 assets and source files.
"""
import sys, os, json, time
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = r"C:\Users\25beevdt025\Desktop\FORGE"
sys.path.insert(0, PROJECT_ROOT)

V3_DIR = os.path.join(PROJECT_ROOT, "outputs", "new_video_reconstructions",
                      "whatsapp_video_2026_10_24", "best_reconstruction_campaign",
                      "final_best_canonical_v3")
V4_DIR = os.path.join(PROJECT_ROOT, "outputs", "new_video_reconstructions",
                      "whatsapp_video_2026_10_24", "best_reconstruction_campaign",
                      "final_best_canonical_v4")
DIAG_DIR = os.path.join(V4_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

results = {"stage": "ground_truth_verification", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

# ─── 1. Verify source videos ─────────────────────────────────────────────────
import cv2

whatsapp_path = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-24 at 17.53.55.mp4"
dji_path = r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4"

video_info = {}
for label, path in [("whatsapp", whatsapp_path), ("dji_4k_master", dji_path)]:
    exists = os.path.isfile(path)
    info = {"exists": exists, "path": path}
    if exists:
        cap = cv2.VideoCapture(path)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["duration_s"] = round(info["frame_count"] / info["fps"], 2) if info["fps"] > 0 else 0
        info["file_size_MB"] = round(os.path.getsize(path) / 1e6, 2)
        cap.release()
    video_info[label] = info
    print(f"  [{label}] exists={exists}" + (f"  {info.get('width','?')}x{info.get('height','?')} {info.get('frame_count','?')}F {info.get('fps','?')}fps {info.get('file_size_MB','?')}MB" if exists else ""))

results["source_videos"] = video_info

# ─── 2. Load and measure existing V3 point cloud ─────────────────────────────
import open3d as o3d

pc_path = os.path.join(V3_DIR, "final_best_canonical_v3_point_cloud.ply")
mesh_path = os.path.join(V3_DIR, "final_best_canonical_v3_mesh.ply")
glb_path = os.path.join(V3_DIR, "final_best_canonical_v3_reconstruction.glb")

print("\n=== Point Cloud Verification ===")
pcd = o3d.io.read_point_cloud(pc_path)
pc_points = np.asarray(pcd.points)
pc_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
pc_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

pc_info = {
    "file_path": pc_path,
    "file_size_MB": round(os.path.getsize(pc_path) / 1e6, 2),
    "point_count": len(pc_points),
    "has_colors": pc_colors is not None and len(pc_colors) > 0,
    "has_normals": pc_normals is not None and len(pc_normals) > 0,
    "bbox_min": pc_points.min(axis=0).tolist(),
    "bbox_max": pc_points.max(axis=0).tolist(),
    "bbox_extents": (pc_points.max(axis=0) - pc_points.min(axis=0)).tolist(),
}
print(f"  Points: {pc_info['point_count']}")
print(f"  Has colors: {pc_info['has_colors']}")
print(f"  Has normals: {pc_info['has_normals']}")
print(f"  BBox min: {pc_info['bbox_min']}")
print(f"  BBox max: {pc_info['bbox_max']}")
print(f"  BBox extents: {pc_info['bbox_extents']}")

# Point density analysis
print("\n=== Point Density Analysis ===")
# Compute nearest-neighbor distances for a subsample
rng = np.random.default_rng(42)
if len(pc_points) > 50000:
    idx_sample = rng.choice(len(pc_points), 50000, replace=False)
    pcd_sample = pcd.select_by_index(idx_sample.tolist())
else:
    pcd_sample = pcd

nn_tree = o3d.geometry.KDTreeFlann(pcd)
nn_dists = []
sample_pts = np.asarray(pcd_sample.points)
for i in range(len(sample_pts)):
    [_, _, dist_sq] = nn_tree.search_knn_vector_3d(sample_pts[i], 2)
    nn_dists.append(np.sqrt(dist_sq[1]))  # nearest neighbor (excluding self)

nn_dists = np.array(nn_dists)
pc_density = {
    "sample_size": len(nn_dists),
    "nn_dist_median_mm": round(float(np.median(nn_dists)) * 1000, 3),
    "nn_dist_mean_mm": round(float(np.mean(nn_dists)) * 1000, 3),
    "nn_dist_p5_mm": round(float(np.percentile(nn_dists, 5)) * 1000, 3),
    "nn_dist_p25_mm": round(float(np.percentile(nn_dists, 25)) * 1000, 3),
    "nn_dist_p75_mm": round(float(np.percentile(nn_dists, 75)) * 1000, 3),
    "nn_dist_p95_mm": round(float(np.percentile(nn_dists, 95)) * 1000, 3),
    "nn_dist_p99_mm": round(float(np.percentile(nn_dists, 99)) * 1000, 3),
    "nn_dist_max_mm": round(float(np.max(nn_dists)) * 1000, 3),
}
print(f"  Median NN distance: {pc_density['nn_dist_median_mm']} mm")
print(f"  Mean NN distance: {pc_density['nn_dist_mean_mm']} mm")
print(f"  P5-P95 range: {pc_density['nn_dist_p5_mm']} - {pc_density['nn_dist_p95_mm']} mm")

# ─── 3. Load and measure existing V3 mesh ─────────────────────────────────────
print("\n=== Mesh Verification ===")
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh_verts = np.asarray(mesh.vertices)
mesh_tris = np.asarray(mesh.triangles)

mesh.compute_vertex_normals()

mesh_info = {
    "file_path": mesh_path,
    "file_size_MB": round(os.path.getsize(mesh_path) / 1e6, 2),
    "vertex_count": len(mesh_verts),
    "triangle_count": len(mesh_tris),
    "has_vertex_colors": mesh.has_vertex_colors(),
    "has_vertex_normals": mesh.has_vertex_normals(),
    "has_triangle_normals": mesh.has_triangle_normals(),
    "bbox_min": mesh_verts.min(axis=0).tolist(),
    "bbox_max": mesh_verts.max(axis=0).tolist(),
    "bbox_extents": (mesh_verts.max(axis=0) - mesh_verts.min(axis=0)).tolist(),
}

# Topology analysis
edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
vertex_manifold = mesh.is_vertex_manifold()
watertight = mesh.is_watertight()

# Get boundary and non-manifold edges
edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
boundary_edges_set = set()
# Count boundary edges by checking half-edge structure
tri_arr = np.asarray(mesh.triangles)
from collections import Counter
edge_count = Counter()
for tri in tri_arr:
    for i in range(3):
        e = tuple(sorted([tri[i], tri[(i+1)%3]]))
        edge_count[e] += 1
boundary_edge_count = sum(1 for e, c in edge_count.items() if c == 1)
non_manifold_edge_count = sum(1 for e, c in edge_count.items() if c > 2)

# Connected components
cluster_ids, cluster_counts, _ = mesh.cluster_connected_triangles()
cluster_ids = np.asarray(cluster_ids)
cluster_counts = np.asarray(cluster_counts)
num_components = len(cluster_counts)
component_sizes = sorted(cluster_counts.tolist(), reverse=True)

mesh_topology = {
    "edge_manifold": edge_manifold,
    "vertex_manifold": vertex_manifold,
    "watertight": watertight,
    "boundary_edges": boundary_edge_count,
    "non_manifold_edges": non_manifold_edge_count,
    "connected_components": num_components,
    "component_sizes_top10": component_sizes[:10],
    "smallest_component_tris": component_sizes[-1] if component_sizes else 0,
}

print(f"  Vertices: {mesh_info['vertex_count']}")
print(f"  Triangles: {mesh_info['triangle_count']}")
print(f"  BBox min: {mesh_info['bbox_min']}")
print(f"  BBox max: {mesh_info['bbox_max']}")
print(f"  BBox extents: {mesh_info['bbox_extents']}")
print(f"  Edge manifold: {edge_manifold}")
print(f"  Vertex manifold: {vertex_manifold}")
print(f"  Watertight: {watertight}")
print(f"  Boundary edges: {boundary_edge_count}")
print(f"  Non-manifold edges: {non_manifold_edge_count}")
print(f"  Connected components: {num_components}")
print(f"  Component sizes (top 10): {component_sizes[:10]}")

# ─── 4. Point-cloud-to-mesh distance analysis ────────────────────────────────
print("\n=== Coverage Analysis: Point Cloud -> Mesh ===")

# For each point in the point cloud, find distance to nearest mesh vertex
mesh_tree = o3d.geometry.KDTreeFlann(mesh)
pc2mesh_dists = []
# Sample to keep compute reasonable
if len(pc_points) > 100000:
    idx_coverage = rng.choice(len(pc_points), 100000, replace=False)
else:
    idx_coverage = np.arange(len(pc_points))

for i in idx_coverage:
    [_, _, dist_sq] = mesh_tree.search_knn_vector_3d(pc_points[i], 1)
    pc2mesh_dists.append(np.sqrt(dist_sq[0]))

pc2mesh_dists = np.array(pc2mesh_dists)

coverage_pc2mesh = {
    "sample_size": len(pc2mesh_dists),
    "median_mm": round(float(np.median(pc2mesh_dists)) * 1000, 3),
    "mean_mm": round(float(np.mean(pc2mesh_dists)) * 1000, 3),
    "p50_mm": round(float(np.percentile(pc2mesh_dists, 50)) * 1000, 3),
    "p75_mm": round(float(np.percentile(pc2mesh_dists, 75)) * 1000, 3),
    "p90_mm": round(float(np.percentile(pc2mesh_dists, 90)) * 1000, 3),
    "p95_mm": round(float(np.percentile(pc2mesh_dists, 95)) * 1000, 3),
    "p99_mm": round(float(np.percentile(pc2mesh_dists, 99)) * 1000, 3),
    "max_mm": round(float(np.max(pc2mesh_dists)) * 1000, 3),
    "within_5mm_pct": round(float(np.mean(pc2mesh_dists <= 0.005)) * 100, 2),
    "within_10mm_pct": round(float(np.mean(pc2mesh_dists <= 0.010)) * 100, 2),
    "within_20mm_pct": round(float(np.mean(pc2mesh_dists <= 0.020)) * 100, 2),
    "within_50mm_pct": round(float(np.mean(pc2mesh_dists <= 0.050)) * 100, 2),
    "beyond_50mm_pct": round(float(np.mean(pc2mesh_dists > 0.050)) * 100, 2),
    "beyond_100mm_pct": round(float(np.mean(pc2mesh_dists > 0.100)) * 100, 2),
}

print(f"  Median PC->Mesh distance: {coverage_pc2mesh['median_mm']} mm")
print(f"  Points within 5mm of mesh: {coverage_pc2mesh['within_5mm_pct']}%")
print(f"  Points within 10mm of mesh: {coverage_pc2mesh['within_10mm_pct']}%")
print(f"  Points within 20mm of mesh: {coverage_pc2mesh['within_20mm_pct']}%")
print(f"  Points within 50mm of mesh: {coverage_pc2mesh['within_50mm_pct']}%")
print(f"  Points BEYOND 50mm of mesh: {coverage_pc2mesh['beyond_50mm_pct']}%")
print(f"  Points BEYOND 100mm of mesh: {coverage_pc2mesh['beyond_100mm_pct']}%")
print(f"  P99 distance: {coverage_pc2mesh['p99_mm']} mm")
print(f"  Max distance: {coverage_pc2mesh['max_mm']} mm")

# ─── 5. Mesh-to-point-cloud distance analysis ────────────────────────────────
print("\n=== Coverage Analysis: Mesh -> Point Cloud ===")

pc_tree = o3d.geometry.KDTreeFlann(pcd)
mesh2pc_dists = []
for i in range(len(mesh_verts)):
    [_, _, dist_sq] = pc_tree.search_knn_vector_3d(mesh_verts[i], 1)
    mesh2pc_dists.append(np.sqrt(dist_sq[0]))

mesh2pc_dists = np.array(mesh2pc_dists)

coverage_mesh2pc = {
    "sample_size": len(mesh2pc_dists),
    "median_mm": round(float(np.median(mesh2pc_dists)) * 1000, 3),
    "mean_mm": round(float(np.mean(mesh2pc_dists)) * 1000, 3),
    "p90_mm": round(float(np.percentile(mesh2pc_dists, 90)) * 1000, 3),
    "p95_mm": round(float(np.percentile(mesh2pc_dists, 95)) * 1000, 3),
    "p99_mm": round(float(np.percentile(mesh2pc_dists, 99)) * 1000, 3),
    "max_mm": round(float(np.max(mesh2pc_dists)) * 1000, 3),
    "within_5mm_pct": round(float(np.mean(mesh2pc_dists <= 0.005)) * 100, 2),
    "within_10mm_pct": round(float(np.mean(mesh2pc_dists <= 0.010)) * 100, 2),
    "beyond_20mm_pct": round(float(np.mean(mesh2pc_dists > 0.020)) * 100, 2),
    "beyond_50mm_pct": round(float(np.mean(mesh2pc_dists > 0.050)) * 100, 2),
}

print(f"  Median Mesh->PC distance: {coverage_mesh2pc['median_mm']} mm")
print(f"  Mesh vertices within 5mm of PC: {coverage_mesh2pc['within_5mm_pct']}%")
print(f"  Mesh vertices within 10mm of PC: {coverage_mesh2pc['within_10mm_pct']}%")
print(f"  Mesh vertices BEYOND 20mm of PC: {coverage_mesh2pc['beyond_20mm_pct']}%")
print(f"  Mesh vertices BEYOND 50mm of PC: {coverage_mesh2pc['beyond_50mm_pct']}%")

# ─── 6. Spatial distribution of missing geometry ─────────────────────────────
print("\n=== Spatial Distribution of Missing Geometry ===")
# Points beyond 20mm from mesh (i.e., missing from mesh but present in point cloud)
far_mask = pc2mesh_dists > 0.020  # 20mm threshold
far_indices = idx_coverage[far_mask]
far_points = pc_points[far_indices]

if len(far_points) > 0:
    missing_regions = {
        "count": int(len(far_points)),
        "percentage_of_sample": round(float(np.mean(far_mask)) * 100, 2),
        "bbox_min": far_points.min(axis=0).tolist(),
        "bbox_max": far_points.max(axis=0).tolist(),
        "centroid": far_points.mean(axis=0).tolist(),
    }
    
    # Bin the missing points spatially to find where they cluster
    # Divide bounding box into a 10x10x10 grid
    pc_min = pc_points.min(axis=0)
    pc_max = pc_points.max(axis=0)
    grid_size = 10
    bin_sizes = (pc_max - pc_min) / grid_size
    bin_sizes = np.maximum(bin_sizes, 1e-6)
    
    far_bins = ((far_points - pc_min) / bin_sizes).astype(int)
    far_bins = np.clip(far_bins, 0, grid_size - 1)
    
    from collections import Counter
    bin_counts = Counter(tuple(b) for b in far_bins)
    top_missing_bins = bin_counts.most_common(10)
    
    missing_hotspots = []
    for bin_idx, count in top_missing_bins:
        center = pc_min + (np.array(bin_idx) + 0.5) * bin_sizes
        missing_hotspots.append({
            "grid_cell": list(bin_idx),
            "center_xyz": center.tolist(),
            "missing_points": count,
        })
    
    missing_regions["hotspots"] = missing_hotspots
    print(f"  Missing points (>20mm from mesh): {missing_regions['count']} ({missing_regions['percentage_of_sample']}%)")
    print(f"  Top missing hotspot centers:")
    for hs in missing_hotspots[:5]:
        print(f"    Grid {hs['grid_cell']}: {hs['missing_points']} pts at XYZ={[round(c,3) for c in hs['center_xyz']]}")
else:
    missing_regions = {"count": 0, "percentage_of_sample": 0.0}
    print("  No significant missing geometry detected.")

# ─── 7. Compile and save ─────────────────────────────────────────────────────
results.update({
    "point_cloud": pc_info,
    "point_cloud_density": pc_density,
    "mesh": mesh_info,
    "mesh_topology": mesh_topology,
    "coverage_pc_to_mesh": coverage_pc2mesh,
    "coverage_mesh_to_pc": coverage_mesh2pc,
    "missing_geometry": missing_regions,
})

out_path = os.path.join(DIAG_DIR, "stage1_ground_truth_audit.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== AUDIT COMPLETE ===")
print(f"Results saved to: {out_path}")

# Summary
print(f"\n{'='*70}")
print(f"GROUND TRUTH SUMMARY")
print(f"{'='*70}")
print(f"  Point cloud: {pc_info['point_count']:,} points")
print(f"  Mesh: {mesh_info['vertex_count']:,} vertices, {mesh_info['triangle_count']:,} triangles")
print(f"  Vertex retention: {mesh_info['vertex_count']/pc_info['point_count']*100:.1f}% of PC points")
print(f"  PC coverage by mesh (within 20mm): {coverage_pc2mesh['within_20mm_pct']}%")
print(f"  Missing from mesh (>20mm): {coverage_pc2mesh['beyond_50mm_pct']}% beyond 50mm")
print(f"  Connected components: {num_components}")
print(f"  Boundary edges: {boundary_edge_count}")
print(f"  DJI 4K master exists: {video_info['dji_4k_master']['exists']}")
