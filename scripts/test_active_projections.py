#!/usr/bin/env python3
"""
F.O.R.G.E — Test Surface-Patch Charting and Multi-Material Texture Baking
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.photographic_texture import (
    scale_camera_intrinsics,
    MultiViewTextureEngine
)

CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V9_DIR = CAMPAIGN_DIR / "final_best_canonical_v9"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
PRED_DIR = CAMPAIGN_DIR / "upstream_16_frames/predictions"
NATIVE_4K_DIR = V6_DIR / "native_4k_frames"

T_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])

def main():
    print("=" * 80)
    print("F.O.R.G.E — TESTING SURFACE-PATCH & MULTI-VIEW TEXTURING FOR V9")
    print("=" * 80)

    # 1. Load V9 Verified Mesh
    v9_mesh_path = V9_DIR / "final_best_canonical_v9_mesh.ply"
    mesh_v9 = o3d.io.read_triangle_mesh(str(v9_mesh_path))
    print(f"Loaded V9 Mesh: {len(mesh_v9.vertices):,} vertices, {len(mesh_v9.triangles):,} triangles")

    # 2. Load Camera Poses & Depth
    poses = np.load(PRED_DIR / "camera_poses.npz")
    depths = np.load(PRED_DIR / "depth_predictions.npz")
    ext_cv = poses["extrinsics"]   # (16, 3, 4)
    intr = poses["intrinsics"]     # (16, 3, 3) at 688x384
    d_maps = np.squeeze(depths["depth"]) # (16, 384, 688)

    frame_files = sorted(list(NATIVE_4K_DIR.glob("*.png")))
    num_cams = len(frame_files)

    ext_canon = np.zeros_like(ext_cv)
    for i in range(num_cams):
        R_cv_i = ext_cv[i, :3, :3]
        T_cv_i = ext_cv[i, :3, 3]
        ext_canon[i, :3, :3] = R_cv_i @ T_CV_TO_GLTF
        ext_canon[i, :3, 3] = T_cv_i

    scaled_intr = scale_camera_intrinsics(intr, (688, 384), (3840, 2160))

    # 3. Initialize MultiViewTextureEngine
    print("[*] Running MultiViewTextureEngine on V9 Mesh...")
    engine = MultiViewTextureEngine(
        mesh=mesh_v9,
        extrinsics=ext_canon,
        intrinsics_native=scaled_intr,
        frame_paths=frame_files,
        depth_maps=d_maps,
        moving_object_strategy=None
    )

    best_cam, scores, stats = engine.compute_face_visibility_and_scores()
    print(f"Direct Photographic Coverage: {stats['photographic_coverage_pct']}% of faces")
    print(f"Per-camera face counts: {stats['per_camera_face_counts']}")

    # 4. Check active projection bounds for each camera
    print("\n[*] Analyzing Active Mesh Projection Bounds per Camera:")
    W_native, H_native = 3840, 2160
    for cam_idx in range(num_cams):
        cam_faces = np.where(best_cam == cam_idx)[0]
        if len(cam_faces) == 0:
            print(f"  Cam {cam_idx:02d}: 0 faces assigned.")
            continue
        
        # Get vertices of these faces
        cam_tris = engine.tris[cam_faces]
        cam_vert_indices = np.unique(cam_tris)
        cam_pts = engine.verts[cam_vert_indices]

        # Project to camera
        R = ext_canon[cam_idx, :3, :3]
        t = ext_canon[cam_idx, :3, 3]
        K = scaled_intr[cam_idx]

        P_c = (R @ cam_pts.T + t[:, None]).T
        Z_c = np.maximum(P_c[:, 2], 1e-4)
        u = K[0, 0] * (P_c[:, 0] / Z_c) + K[0, 2]
        v = K[1, 1] * (P_c[:, 1] / Z_c) + K[1, 2]

        u_min, u_max = max(0, int(np.min(u))), min(W_native, int(np.max(u)))
        v_min, v_max = max(0, int(np.min(v))), min(H_native, int(np.max(v)))
        pw = u_max - u_min
        ph = v_max - v_min
        active_area_pct = (pw * ph) / (W_native * H_native) * 100.0
        print(f"  Cam {cam_idx:02d}: {len(cam_faces):,} faces | Active crop: [{u_min}:{u_max}, {v_min}:{v_max}] ({pw}x{ph}) = {active_area_pct:.1f}% of frame")

if __name__ == "__main__":
    main()
