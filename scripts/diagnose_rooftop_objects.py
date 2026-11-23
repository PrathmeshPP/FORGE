#!/usr/bin/env python3
"""
F.O.R.G.E — Rooftop Objects & Merged Tree Canopy Forensic Investigation
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
V8_DIR = CAMPAIGN_DIR / "final_best_canonical_v8"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
PRED_DIR = CAMPAIGN_DIR / "upstream_16_frames/predictions"
NATIVE_FRAMES_DIR = V6_DIR / "native_4k_frames"

def main():
    print("=" * 80)
    print("F.O.R.G.E — ROOFTOP OBJECTS & TREE CANOPY FORENSIC INVESTIGATION")
    print("=" * 80)

    # 1. Load V8 Mesh and Point Cloud
    mesh_path = V8_DIR / "final_best_canonical_v8_mesh.ply"
    pcd_path = V6_DIR / "final_best_canonical_v6_point_cloud.ply"
    glb_path = V8_DIR / "final_best_canonical_v8_reconstruction.glb"

    mesh = trimesh.load(str(mesh_path), process=False)
    pcd = trimesh.load(str(pcd_path), process=False)
    glb_scene = trimesh.load(str(glb_path))
    glb_mesh = list(glb_scene.geometry.values())[0] if isinstance(glb_scene, trimesh.Scene) else glb_scene

    print(f"Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    print(f"Point Cloud: {len(pcd.vertices):,} points")

    # 2. Inspect Connected Components
    components = mesh.split(only_watertight=False)
    print(f"Total Components: {len(components)}")

    # Find rooftop objects: X in [-0.4, 0.4], Z in [-1.0, -0.4], Y > 0.15
    print("\n[1] Examining Components above the Roof (Y > 0.15):")
    rooftop_comps = []
    for idx, c in enumerate(components):
        b = c.bounds
        if b[1, 1] > 0.15:
            rooftop_comps.append((idx, c))
            print(f"  Comp #{idx}: {len(c.vertices):,} verts, {len(c.faces):,} faces")
            print(f"    Bounds X: [{b[0,0]:.3f}, {b[1,0]:.3f}], Y: [{b[0,1]:.3f}, {b[1,1]:.3f}], Z: [{b[0,2]:.3f}, {b[1,2]:.3f}]")

    # 3. For Component #42 (and any other high component), analyze point cloud support
    for idx, comp in rooftop_comps:
        comp_b = comp.bounds
        # Find points in pcd near this component
        mask = (
            (pcd.vertices[:, 0] >= comp_b[0, 0] - 0.05) & (pcd.vertices[:, 0] <= comp_b[1, 0] + 0.05) &
            (pcd.vertices[:, 1] >= comp_b[0, 1] - 0.05) & (pcd.vertices[:, 1] <= comp_b[1, 1] + 0.05) &
            (pcd.vertices[:, 2] >= comp_b[0, 2] - 0.05) & (pcd.vertices[:, 2] <= comp_b[1, 2] + 0.05)
        )
        nearby_pts = pcd.vertices[mask]
        print(f"\n  Comp #{idx} has {len(nearby_pts):,} nearby point cloud points (within 5cm).")
        if len(nearby_pts) > 0 and hasattr(pcd, 'colors') and pcd.colors is not None:
            nearby_colors = pcd.colors[mask]
            print(f"    Point cloud mean color (RGBA): {np.mean(nearby_colors, axis=0)}")

    # 4. Investigate Texture Projection on Component #42 in the GLB
    # Where does Component #42 map onto the texture atlas?
    print("\n[2] Examining Texture Atlas Mapping for Rooftop Components:")
    mat = glb_mesh.visual.material
    atlas_img = mat.baseColorTexture if hasattr(mat, 'baseColorTexture') else (mat.image if hasattr(mat, 'image') else None)
    if atlas_img is not None:
        atlas_np = np.array(atlas_img)
        print(f"  Atlas Image shape: {atlas_np.shape}")
        
        # Check UVs of faces belonging to Component #42
        # Since GLB mesh has unmerged vertices for UVs, let's locate faces matching Comp #42 vertices
        c42 = components[42]
        c42_center = np.mean(c42.vertices, axis=0)
        print(f"  Comp #42 center: {c42_center}")

        # In GLB mesh, find vertices close to c42_center
        dists = np.linalg.norm(glb_mesh.vertices - c42_center, axis=1)
        close_verts = np.where(dists < 0.2)[0]
        print(f"  GLB vertices within 20cm of Comp #42 center: {len(close_verts)}")
        if len(close_verts) > 0 and hasattr(glb_mesh.visual, 'uv'):
            c42_uvs = glb_mesh.visual.uv[close_verts]
            print(f"    UV range for Comp #42: U in [{c42_uvs[:,0].min():.4f}, {c42_uvs[:,0].max():.4f}], V in [{c42_uvs[:,1].min():.4f}, {c42_uvs[:,1].max():.4f}]")
            
            # Sample atlas pixels at these UVs
            # Note: in glTF UVs, V=0 is top or bottom? Trimesh handles UV origin: usually V=0 is bottom, 1-V is top
            px_x = np.clip((c42_uvs[:, 0] * (atlas_np.shape[1] - 1)).astype(int), 0, atlas_np.shape[1] - 1)
            px_y = np.clip(((1.0 - c42_uvs[:, 1]) * (atlas_np.shape[0] - 1)).astype(int), 0, atlas_np.shape[0] - 1)
            sampled_colors = atlas_np[px_y, px_x]
            print(f"    Sampled Atlas RGB for Comp #42: mean={np.mean(sampled_colors[:, :3], axis=0)}, min={np.min(sampled_colors[:, :3], axis=0)}, max={np.max(sampled_colors[:, :3], axis=0)}")
            # Is it white? (e.g. RGB > 200?)
            white_ratio = np.mean(np.all(sampled_colors[:, :3] > 180, axis=1))
            print(f"    Ratio of sampled pixels that are bright white (>180): {white_ratio:.2%}")

            # Check which camera tile in the atlas these UVs fall into!
            # Atlas is 4x4 grid: row = int(px_y / tile_h), col = int(px_x / tile_w)
            tile_w = atlas_np.shape[1] // 4
            tile_h = atlas_np.shape[0] // 4
            cam_indices = []
            for y, x in zip(px_y, px_x):
                r = min(3, y // tile_h)
                c = min(3, x // tile_w)
                cam_indices.append(r * 4 + c)
            unique_cams, counts = np.unique(cam_indices, return_counts=True)
            print(f"    Camera Tile distribution for Comp #42: {dict(zip(unique_cams, counts))}")

    # 5. Check what is physically in the source frames at that location!
    # Load camera poses and project Comp #42 center into frames
    print("\n[3] Projecting Comp #42 Center into Source Camera Views:")
    poses = np.load(PRED_DIR / "camera_poses.npz")
    ext_cv = poses["extrinsics"]   # (16, 3, 4)
    intr = poses["intrinsics"]     # (16, 3, 3) at 688x384
    # Intrinsics scaled to 4K (3840x2160)
    sx = 3840.0 / 688.0
    sy = 2160.0 / 384.0
    intr_4k = intr[0].copy()
    intr_4k[0, :] *= sx
    intr_4k[1, :] *= sy

    T_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])
    c42_pt_canonical = c42_center

    for i in range(16):
        # Transform canonical point (+Y Up) back to OpenCV camera space
        # Recall canonical mesh was transformed by T_CV_TO_GLTF from initial frame 0 space
        # In frame 0 OpenCV space: P_cv0 = T_CV_TO_GLTF @ P_canonical
        p_cv0 = T_CV_TO_GLTF @ c42_pt_canonical
        # Camera extrinsics transform from frame 0 space to frame i camera space:
        # P_cam = R_i @ P_cv0 + t_i
        R = ext_cv[i, :3, :3]
        t = ext_cv[i, :3, 3]
        p_cam = R @ p_cv0 + t

        if p_cam[2] > 0.1: # In front of camera
            u = (intr_4k[0, 0] * p_cam[0] / p_cam[2]) + intr_4k[0, 2]
            v = (intr_4k[1, 1] * p_cam[1] / p_cam[2]) + intr_4k[1, 2]
            in_view = (0 <= u < 3840) and (0 <= v < 2160)
            if in_view:
                print(f"    Cam {i:02d}: projects to pixel ({u:.1f}, {v:.1f}), depth {p_cam[2]:.3f} m [IN FRUSTUM]")
            else:
                print(f"    Cam {i:02d}: projects to pixel ({u:.1f}, {v:.1f}), depth {p_cam[2]:.3f} m [OUT OF FRUSTUM]")
        else:
            print(f"    Cam {i:02d}: BEHIND CAMERA (z={p_cam[2]:.3f})")

if __name__ == "__main__":
    main()
