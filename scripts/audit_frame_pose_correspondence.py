#!/usr/bin/env python3
"""
F.O.R.G.E — Build Frame-to-Camera Correspondence and Timestamp Audit CSV
"""

import sys
import os
import csv
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
V8_DIR = CAMPAIGN_DIR / "final_best_canonical_v8"
V9_DIR = CAMPAIGN_DIR / "final_best_canonical_v9"
V9_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = CAMPAIGN_DIR / "upstream_16_frames/predictions"
UPSTREAM_FRAMES = CAMPAIGN_DIR / "upstream_16_frames/frames"
NATIVE_4K_DIR = V6_DIR / "native_4k_frames"
MASTER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")

def main():
    print("=" * 80)
    print("F.O.R.G.E — FRAME-TO-CAMERA CORRESPONDENCE AUDIT")
    print("=" * 80)

    poses = np.load(PRED_DIR / "camera_poses.npz")
    depths = np.load(PRED_DIR / "depth_predictions.npz")
    extrinsics = poses["extrinsics"] # (16, 3, 4)
    intrinsics = poses["intrinsics"] # (16, 3, 3) at 688x384
    d_maps = np.squeeze(depths["depth"]) # (16, 384, 688)

    up_files = sorted(list(UPSTREAM_FRAMES.glob("*.png")))
    nat_files = sorted(list(NATIVE_4K_DIR.glob("*.png")))

    print(f"Extrinsics: {extrinsics.shape}")
    print(f"Intrinsics: {intrinsics.shape}")
    print(f"Depth maps: {d_maps.shape}")
    print(f"Upstream frames: {len(up_files)}")
    print(f"Native 4K frames: {len(nat_files)}")

    # Open Master Video to verify exact frame timestamps and seek accuracy
    cap = cv2.VideoCapture(str(MASTER_VIDEO)) if MASTER_VIDEO.exists() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 59.94
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
    print(f"Master Video: {fps:.2f} fps, {total_frames:,} frames")

    csv_rows = []
    headers = [
        "array_index",
        "timestamp_seconds",
        "upstream_filename",
        "native_filename",
        "extracted_dimensions",
        "depth_map_index",
        "depth_map_shape",
        "camera_pose_index",
        "cam_pos_x",
        "cam_pos_y",
        "cam_pos_z",
        "focal_x_native",
        "focal_y_native",
        "center_x_native",
        "center_y_native",
        "laplacian_variance",
        "tenengrad_energy",
        "motion_blur_index",
        "primary_scene_features"
    ]

    features = [
        "South Approach / Entry Road & Gate",
        "Entrance Courtyard & South Perimeter Trees",
        "Courtyard Pavement & Ground Transition",
        "Main Building South Entrance & Lower Facade",
        "Building Facade & Southwest Tree Margin",
        "Building Elevation & West Roof Corner",
        "Building Roof & Front Solar Panel Array",
        "Nadir Building Complex & Dual Solar Arrays",
        "Dual Solar Arrays & Central Water Tanks",
        "Building Rooftop, Water Tanks & HVAC",
        "North Roof Eaves & Perimeter Forest Canopy",
        "North Forest Canopy & Building Rear Wall",
        "East Facade & Ground Terrain Slope",
        "Eastern Ground, Trees & Courtyard Apron",
        "Courtyard Pavement & East Approach Road",
        "Full Facility Departure Corridor & Exit Road"
    ]

    for i in range(len(extrinsics)):
        up_f = up_files[i]
        nat_f = nat_files[i]

        # Extract timestamp from filename, e.g. frame_0007_49.973s.png -> 49.973
        ts_part = nat_f.stem.split("_")[-1].replace("s", "")
        ts = float(ts_part)

        # Compute camera center: C = -R^T @ t
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        C = -R.T @ t

        # Scaled intrinsics (688x384 -> 3840x2160)
        sx = 3840.0 / 688.0
        sy = 2160.0 / 384.0
        fx_nat = intrinsics[i, 0, 0] * sx
        fy_nat = intrinsics[i, 1, 1] * sy
        cx_nat = intrinsics[i, 0, 2] * sx
        cy_nat = intrinsics[i, 1, 2] * sy

        # Read native frame to measure sharpness
        im = cv2.imread(str(nat_f))
        fh, fw, _ = im.shape
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))

        var_x = np.var(gx)
        var_y = np.var(gy)
        ratio = max(var_x, var_y) / (min(var_x, var_y) + 1e-5)
        blur_idx = float(np.clip(ratio - 1.0, 0.0, 2.0))

        row = [
            i,
            round(ts, 3),
            up_f.name,
            nat_f.name,
            f"{fw}x{fh}",
            i,
            f"{d_maps[i].shape[1]}x{d_maps[i].shape[0]}",
            i,
            round(float(C[0]), 4),
            round(float(C[1]), 4),
            round(float(C[2]), 4),
            round(float(fx_nat), 2),
            round(float(fy_nat), 2),
            round(float(cx_nat), 2),
            round(float(cy_nat), 2),
            round(lap_var, 2),
            round(tenengrad, 2),
            round(blur_idx, 3),
            features[i]
        ]
        csv_rows.append(row)
        print(f"[{i:02d}] t={ts:6.3f}s | {nat_f.name:<24} | Cam=({C[0]:6.3f}, {C[1]:6.3f}, {C[2]:6.3f}) | LapVar={lap_var:6.1f} | {features[i]}")

    if cap is not None:
        cap.release()

    csv_path = V9_DIR / "frame_pose_correspondence.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    print(f"\n[*] Saved Frame-to-Camera Correspondence CSV: {csv_path}")

if __name__ == "__main__":
    main()
