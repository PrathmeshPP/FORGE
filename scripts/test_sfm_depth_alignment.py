#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Test SfM-Guided Depth Alignment & Multi-View Fusion
Evaluates whether PyCOLMAP bundle-adjusted camera poses and sparse 3D points
can calibrate depth maps to eliminate scale drift, distortion, and patchwork artifacts.
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
PILOT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm"
REC_DIR = PILOT_DIR / "sparse/0"
PRED_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/upstream_16_frames/predictions"

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — SFM-GUIDED DEPTH CALIBRATION EVALUATION")
    print("=" * 80)

    # 1. Load PyCOLMAP Reconstruction
    rec = pycolmap.Reconstruction(str(REC_DIR))
    print(f"Loaded SfM Reconstruction: {rec.num_reg_images()} images, {rec.num_points3D():,} 3D points")

    # 2. Load existing 16-frame depth predictions to check depth-to-SfM correlation
    depths = np.load(PRED_DIR / "depth_predictions.npz")
    d_maps = np.squeeze(depths["depth"]) # (16, 384, 688)
    print(f"Loaded 16 Depth Maps: shape={d_maps.shape}")

    # Check the camera intrinsics from SfM
    cam = list(rec.cameras.values())[0]
    f_sfm, cx_sfm, cy_sfm, k1_sfm = cam.params
    print(f"SfM Camera: f={f_sfm:.2f}, c=({cx_sfm:.1f}, {cy_sfm:.1f}), k1={k1_sfm:.5f}")

    # For each registered image in SfM, inspect point observations
    print("\nPer-Camera 3D Point Observations & Depth Distribution:")
    print(f"{'Image Name':<28} {'Num 3D Points':<15} {'Depth Range (m)':<25} {'Mean Depth (m)'}")
    print("-" * 80)

    all_correlations = []
    for img_id, img in rec.images.items():
        points2D = img.points2D
        cfw = img.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation

        z_vals = []
        for p2d in points2D:
            if p2d.has_point3D():
                pt3d = rec.points3D[p2d.point3D_id]
                # Transform point to camera coordinates: P_c = R @ P_w + t
                p_c = R @ pt3d.xyz + t
                z_vals.append(p_c[2])

        z_vals = np.array(z_vals)
        if len(z_vals) > 0:
            print(f"{img.name:<28} {len(z_vals):<15} [{z_vals.min():.2f}, {z_vals.max():.2f}] m{' '*10} {z_vals.mean():.2f} m")

if __name__ == "__main__":
    main()
