#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Test Dense Multi-View Stereo & Depth Alignment
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pycolmap
import open3d as o3d

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
PILOT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm"
REC_DIR = PILOT_DIR / "sparse/0"
IMAGES_DIR = PILOT_DIR / "images"

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — DENSE STEREO & RECONSTRUCTION PIPELINE TESTING")
    print("=" * 80)

    rec = pycolmap.Reconstruction(str(REC_DIR))
    images = sorted(list(rec.images.values()), key=lambda x: x.name)
    cam = list(rec.cameras.values())[0]

    f_sfm, cx_sfm, cy_sfm, k1 = cam.params
    K = np.array([
        [f_sfm, 0.0, cx_sfm],
        [0.0, f_sfm, cy_sfm],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0]) # k1, k2, p1, p2

    print(f"Cameras count: {len(images)}")
    print(f"Intrinsics K:\n{K}")
    print(f"Distortion k1: {k1}")

    # Test StereoSGBM on consecutive pair: image 10 and 11 (directly over the building/roof)
    img1_info = images[10]
    img2_info = images[11]

    im1_bgr = cv2.imread(str(IMAGES_DIR / img1_info.name))
    im2_bgr = cv2.imread(str(IMAGES_DIR / img2_info.name))
    h, w, _ = im1_bgr.shape

    # Relative camera pose from image 1 to image 2:
    cfw1 = img1_info.cam_from_world()
    cfw2 = img2_info.cam_from_world()
    R1, t1 = cfw1.rotation.matrix(), cfw1.translation
    R2, t2 = cfw2.rotation.matrix(), cfw2.translation

    # Camera centers in world
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    base_in_1 = R1 @ (C2 - C1)
    print(f"C2 - C1 in cam1: {base_in_1}")

    # For standard stereo, camera L should have camera R to its right (+X)
    # If base_in_1[0] < 0, camera 2 is to the left of camera 1, so img2 is left, img1 is right!
    if base_in_1[0] < 0:
        print("Camera 2 is to the left of Camera 1 -> img2 is Left, img1 is Right")
        im_left_bgr, im_right_bgr = im2_bgr, im1_bgr
        RL, tL = R2, t2
        RR, tR = R1, t1
        left_name, right_name = img2_info.name, img1_info.name
    else:
        print("Camera 1 is to the left of Camera 2 -> img1 is Left, img2 is Right")
        im_left_bgr, im_right_bgr = im1_bgr, im2_bgr
        RL, tL = R1, t1
        RR, tR = R2, t2
        left_name, right_name = img1_info.name, img2_info.name

    # Transform from Left to Right: P_R = R_rel @ P_L + t_rel
    R_rel = RR @ RL.T
    t_rel = (tR - RR @ RL.T @ tL).reshape(3, 1)
    baseline = np.linalg.norm(t_rel)
    print(f"Rectified Pair: Left {left_name} <-> Right {right_name} | Baseline = {baseline:.3f} m")

    # Stereo rectification
    RL_rect, RR_rect, PL_rect, PR_rect, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist_coeffs, K, dist_coeffs, (w, h), R_rel, t_rel, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(K, dist_coeffs, RL_rect, PL_rect, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K, dist_coeffs, RR_rect, PR_rect, (w, h), cv2.CV_32FC1)

    im_left_rect = cv2.remap(im_left_bgr, mapLx, mapLy, cv2.INTER_LINEAR)
    im_right_rect = cv2.remap(im_right_bgr, mapRx, mapRy, cv2.INTER_LINEAR)

    # StereoSGBM matching
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8, # 128 disparities
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = sgbm.compute(im_left_rect, im_right_rect).astype(np.float32) / 16.0
    valid_disp = disp > 2.0
    print(f"Valid disparity pixels: {valid_disp.sum():,} ({valid_disp.mean()*100:.1f}%)")

    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    pts_rect = points_3d[valid_disp]
    cols_rect = cv2.cvtColor(im_left_rect, cv2.COLOR_BGR2RGB)[valid_disp] / 255.0

    print(f"pts_rect shape: {pts_rect.shape}")
    print(f"pts_rect Z min: {pts_rect[:, 2].min():.3f}, max: {pts_rect[:, 2].max():.3f}, median: {np.median(pts_rect[:, 2]):.3f}")
    mask_z = (pts_rect[:, 2] > 0.0) & (pts_rect[:, 2] < 100.0) & np.isfinite(pts_rect[:, 2])
    pts_clean = pts_rect[mask_z]
    cols_clean = cols_rect[mask_z]
    print(f"Clean 3D points from single stereo pair: {len(pts_clean):,}")

    # Transform back to world coordinates from Left camera
    # P_cL = RL_rect^T @ P_rect; P_w = RL^T @ (P_cL - tL)
    P_cL = (RL_rect.T @ pts_clean.T).T
    P_w = (RL.T @ (P_cL - tL).T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_w)
    pcd.colors = o3d.utility.Vector3dVector(cols_clean)

    out_pcd_path = PILOT_DIR / "test_stereo_pair_pts.ply"
    o3d.io.write_point_cloud(str(out_pcd_path), pcd)
    print(f"Saved test dense stereo point cloud: {out_pcd_path.name}")

if __name__ == "__main__":
    main()
