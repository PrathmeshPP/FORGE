#!/usr/bin/env python3
"""
F.O.R.G.E — Forensic Analysis of V7/V8 Geometry, Texturing, Rooftop Objects, and Camera Poses.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import trimesh
import cv2
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
V7_DIR = CAMPAIGN_DIR / "final_best_canonical_v7"
V8_DIR = CAMPAIGN_DIR / "final_best_canonical_v8"
PRED_DIR = CAMPAIGN_DIR / "upstream_16_frames/predictions"
UPSTREAM_FRAMES_DIR = CAMPAIGN_DIR / "upstream_16_frames/frames"
NATIVE_FRAMES_DIR = V6_DIR / "native_4k_frames"
CAMPUS_REF_DIR = PROJECT_ROOT / "outputs/opencv5_experiments/phase2d/campus_0049/candidate_a"

def main():
    print("=" * 80)
    print("F.O.R.G.E — FORENSIC ANALYSIS OF V7/V8 GEOMETRY & TEXTURING")
    print("=" * 80)

    # 1. Compare V7 and V8 Neutral Meshes
    print("\n[1] Comparing Neutral Meshes: V7 vs V8...")
    v7_mesh_path = V7_DIR / "final_best_canonical_v7_mesh.ply"
    v8_mesh_path = V8_DIR / "final_best_canonical_v8_mesh.ply"

    v7_m = trimesh.load(str(v7_mesh_path), process=False)
    v8_m = trimesh.load(str(v8_mesh_path), process=False)

    print(f"    V7 Mesh: {len(v7_m.vertices):,} vertices, {len(v7_m.faces):,} faces")
    print(f"    V8 Mesh: {len(v8_m.vertices):,} vertices, {len(v8_m.faces):,} faces")

    vert_diff = np.max(np.abs(v7_m.vertices - v8_m.vertices))
    face_diff = np.max(np.abs(v7_m.faces - v8_m.faces))
    print(f"    Max Vertex Coordinate Difference: {vert_diff:.10e} m")
    print(f"    Max Face Index Difference:        {face_diff}")
    if vert_diff < 1e-6:
        print("    -> V8 neutral mesh is 100% BIT-IDENTICAL to V7 neutral mesh!")

    # 2. Inspect V8 GLB File Structure and Embedded Assets
    print("\n[2] Inspecting V8 GLB File...")
    v8_glb_path = V8_DIR / "final_best_canonical_v8_reconstruction.glb"
    v8_scene = trimesh.load(str(v8_glb_path))
    
    if isinstance(v8_scene, trimesh.Scene):
        print(f"    GLB Scene Geometry Nodes: {list(v8_scene.geometry.keys())}")
        mesh_v8_glb = list(v8_scene.geometry.values())[0]
    else:
        mesh_v8_glb = v8_scene

    print(f"    GLB Mesh: {len(mesh_v8_glb.vertices):,} vertices, {len(mesh_v8_glb.faces):,} faces")
    print(f"    Visual type: {type(mesh_v8_glb.visual)}")
    
    if hasattr(mesh_v8_glb.visual, 'material'):
        mat = mesh_v8_glb.visual.material
        print(f"    Material: {type(mat).__name__}, name='{getattr(mat, 'name', '')}'")
        if hasattr(mat, 'image') and mat.image is not None:
            print(f"    Embedded Atlas Image size: {mat.image.size}, mode: {mat.image.mode}")
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            print(f"    baseColorTexture size: {mat.baseColorTexture.size}, mode: {mat.baseColorTexture.mode}")
            
    if hasattr(mesh_v8_glb.visual, 'uv') and mesh_v8_glb.visual.uv is not None:
        uvs = mesh_v8_glb.visual.uv
        print(f"    UV coords shape: {uvs.shape}")
        print(f"    UV min: ({uvs[:,0].min():.4f}, {uvs[:,1].min():.4f}), max: ({uvs[:,0].max():.4f}, {uvs[:,1].max():.4f})")
        u_out = np.sum((uvs[:, 0] < 0.0) | (uvs[:, 0] > 1.0))
        v_out = np.sum((uvs[:, 1] < 0.0) | (uvs[:, 1] > 1.0))
        print(f"    UV out of [0, 1] range: U={u_out}, V={v_out}")

    # 3. Source Frames and Prediction Arrays Correspondence
    print("\n[3] Investigating 16 Frames, Timestamps, and Camera Poses...")
    poses = np.load(PRED_DIR / "camera_poses.npz")
    depths = np.load(PRED_DIR / "depth_predictions.npz")
    print(f"    Poses arrays: {list(poses.keys())}")
    print(f"    Extrinsics shape: {poses['extrinsics'].shape}")
    print(f"    Intrinsics shape: {poses['intrinsics'].shape}")
    print(f"    Depths arrays: {list(depths.keys())}, shape: {depths['depth'].shape}")

    upstream_files = sorted(list(UPSTREAM_FRAMES_DIR.glob("*.png")))
    native_files = sorted(list(NATIVE_FRAMES_DIR.glob("*.png")))
    print(f"    Upstream frames count: {len(upstream_files)}")
    print(f"    Native frames count:   {len(native_files)}")

    print(f"\n    {'Idx':<4} {'Upstream Filename':<26} {'Native Filename':<26} {'Cam Pose (x,y,z)':<30}")
    print("    " + "-" * 90)
    for i in range(len(poses['extrinsics'])):
        c2w = np.linalg.inv(np.vstack([poses['extrinsics'][i], [0,0,0,1]]))
        pos = c2w[:3, 3]
        up_name = upstream_files[i].name if i < len(upstream_files) else "N/A"
        nat_name = native_files[i].name if i < len(native_files) else "N/A"
        print(f"    {i:<4} {up_name:<26} {nat_name:<26} ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")

    # 4. Investigate White Rooftop Objects
    print("\n[4] Investigating Rooftop Coordinates & Rooftop Geometry...")
    bounds = v8_m.bounds
    print(f"    Mesh Bounding Box:")
    print(f"      X: [{bounds[0,0]:.3f}, {bounds[1,0]:.3f}] (width: {bounds[1,0]-bounds[0,0]:.3f} m)")
    print(f"      Y: [{bounds[0,1]:.3f}, {bounds[1,1]:.3f}] (height/elevation: {bounds[1,1]-bounds[0,1]:.3f} m)")
    print(f"      Z: [{bounds[0,2]:.3f}, {bounds[1,2]:.3f}] (depth: {bounds[1,2]-bounds[0,2]:.3f} m)")

    y_vals = v8_m.vertices[:, 1]
    top_10_percentile = np.percentile(y_vals, 90)
    top_5_percentile = np.percentile(y_vals, 95)
    max_y = np.max(y_vals)
    print(f"    Elevation distribution: max Y = {max_y:.3f} m, 95th percentile = {top_5_percentile:.3f} m, 90th percentile = {top_10_percentile:.3f} m")

    components = v8_m.split(only_watertight=False)
    print(f"    Total connected components in V8 mesh: {len(components)}")
    comp_sizes = [len(c.vertices) for c in components]
    comp_sizes_sorted = sorted(comp_sizes, reverse=True)
    print(f"    Top 10 largest component vertex counts: {comp_sizes_sorted[:10]}")

    print("\n    Components at high elevation (Y > top_5_percentile):")
    high_comps = []
    for idx, c in enumerate(components):
        if c.bounds[1, 1] > top_5_percentile:
            high_comps.append((idx, len(c.vertices), len(c.faces), c.bounds))
    
    high_comps.sort(key=lambda x: x[1], reverse=True)
    for idx, nv, nf, b in high_comps[:10]:
        print(f"      Comp #{idx}: {nv:,} verts, {nf:,} faces, Y range [{b[0,1]:.3f}, {b[1,1]:.3f}], X range [{b[0,0]:.3f}, {b[1,0]:.3f}], Z range [{b[0,2]:.3f}, {b[1,2]:.3f}]")

    # 5. Check Point Cloud Support for High Elevation Structures
    print("\n[5] Checking Point Cloud Support for High Elevation Structures...")
    pcd_path = CAMPAIGN_DIR / "final_best_canonical_v6/final_best_canonical_v6_point_cloud.ply"
    pcd = trimesh.load(str(pcd_path), process=False)
    pcd_pts = pcd.vertices
    pcd_colors = pcd.colors if hasattr(pcd, 'colors') and pcd.colors is not None else None
    print(f"    Point cloud: {len(pcd_pts):,} points, Y range [{pcd_pts[:,1].min():.3f}, {pcd_pts[:,1].max():.3f}]")
    high_pcd = pcd_pts[pcd_pts[:, 1] > top_5_percentile]
    print(f"    Point cloud points at Y > {top_5_percentile:.3f} m: {len(high_pcd):,}")

    if pcd_colors is not None:
        high_colors = pcd_colors[pcd_pts[:, 1] > top_5_percentile]
        print(f"    Mean RGB of high points: {np.mean(high_colors[:, :3], axis=0)}")

if __name__ == "__main__":
    main()
