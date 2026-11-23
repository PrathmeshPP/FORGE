#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Surface Reconstruction Testing (Screened Poisson vs Delaunay / BPA)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
PILOT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm"
DENSE_PLY = PILOT_DIR / "dense_mvs/fused_dense_point_cloud.ply"
OUT_DIR = PILOT_DIR / "mesh_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — SURFACE MESH RECONSTRUCTION TESTING")
    print("=" * 80)

    pcd = o3d.io.read_point_cloud(str(DENSE_PLY))
    pts = np.asarray(pcd.points)
    print(f"Loaded dense point cloud: {len(pts):,} points")

    t0 = time.time()
    print("\nRunning Screened Poisson Surface Reconstruction (depth=9)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=True
    )
    dt_poisson = time.time() - t0
    densities_np = np.asarray(densities)
    print(f"Poisson meshing completed in {dt_poisson:.2f}s:")
    print(f"  Raw vertices: {len(mesh.vertices):,}")
    print(f"  Raw triangles: {len(mesh.triangles):,}")

    # Prune low density vertices (extrapolated bubbles in unobserved space)
    q_low = np.percentile(densities_np, 8.0) # bottom 8%
    print(f"Pruning vertices with density < {q_low:.2f} (quantile 0.08)...")
    vertices_to_remove = densities_np < q_low
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  Vertices after density prune: {len(mesh.vertices):,}")
    print(f"  Triangles after density prune: {len(mesh.triangles):,}")

    # Clean topology
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Distance-based support pruning: remove triangles far from any real point observation
    print("\nPruning unsupported extrapolated triangles (distance > 0.30m from dense points)...")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    mesh_verts = np.asarray(mesh.vertices)
    unsupported_verts = np.zeros(len(mesh_verts), dtype=bool)

    for i, v in enumerate(mesh_verts):
        [_, _, d2] = kdtree.search_knn_vector_3d(v, 1)
        if np.sqrt(d2[0]) > 0.35: # > 35 cm from nearest point
            unsupported_verts[i] = True

    mesh.remove_vertices_by_mask(unsupported_verts)
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # Filter connected components: keep components with > 200 triangles
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    valid_clusters = np.where(cluster_n_triangles >= 200)[0]
    triangles_to_remove = ~np.isin(triangle_clusters, valid_clusters)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    print(f"\nFinal Cleaned Surface Mesh:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")
    print(f"  Is edge manifold: {mesh.is_edge_manifold()}")
    print(f"  Is vertex manifold: {mesh.is_vertex_manifold()}")
    print(f"  Is watertight: {mesh.is_watertight()}")

    # Transfer vertex colors from nearest point cloud points
    mesh.compute_vertex_normals()
    mesh_verts_clean = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    vert_colors = np.zeros((len(mesh_verts_clean), 3))
    for i, v in enumerate(mesh_verts_clean):
        [_, idxs, _] = kdtree.search_knn_vector_3d(v, 1)
        vert_colors[i] = pcd_colors[idxs[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)

    out_mesh_path = OUT_DIR / "candidate_a_screened_poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_mesh_path), mesh)
    print(f"Saved candidate mesh: {out_mesh_path.name} ({out_mesh_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
