#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Pilot Structure-from-Motion Evaluation using PyCOLMAP
Tests independent photogrammetric camera tracking and bundle adjustment
on a representative sequence of 24 native drone frames.
"""

import sys
import os
import time
import json
from pathlib import Path
import cv2
import numpy as np
import pycolmap
import open3d as o3d

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
MASTER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V10_DIR = CAMPAIGN_DIR / "final_best_canonical_v10"
PILOT_DIR = V10_DIR / "pilot_sfm"
PILOT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = PILOT_DIR / "images"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PILOT_DIR / "database.db"
REC_DIR = PILOT_DIR / "sparse"
REC_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — PILOT SFM PIPELINE (PYCOLMAP 4.2.1)")
    print("=" * 80)
    t0 = time.perf_counter()

    # 1. Select 24 overlapping frames spanning 32.0s to 68.0s
    timestamps = np.linspace(32.0, 68.0, 24)
    print(f"[*] Extracting 24 Pilot Frames from {MASTER_VIDEO.name} (32.0s - 68.0s)...")

    cap = cv2.VideoCapture(str(MASTER_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    # Use 1920x1080 for fast and highly accurate SIFT detection
    target_w, target_h = 1920, 1080

    for i, t in enumerate(timestamps):
        fn = FRAMES_DIR / f"frame_{i:04d}_{t:06.3f}s.png"
        frame_idx = min(int(round(t * fps)), total_f - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if ret:
            # Resize with area interpolation
            frame_resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(fn), frame_resized)
            frame_paths.append(fn)
            print(f"    [{i:02d}] Extracted t={t:5.2f}s -> {fn.name}")

    cap.release()
    print(f"    Saved {len(frame_paths)} images to {FRAMES_DIR.name}")

    # Remove existing DB if any
    if DB_PATH.exists():
        DB_PATH.unlink()

    # 2. Extract SIFT Features
    print(f"\n[*] Extracting SIFT Features via PyCOLMAP...")
    t_feat = time.perf_counter()

    # Using SIMPLE_RADIAL camera model (shared single camera for the entire flight)
    # SIMPLE_RADIAL estimates: focal_length, principal_point_x, principal_point_y, radial_distortion_k1
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.sift.max_num_features = 4096
    extract_opts.max_image_size = 2000
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_RADIAL"

    pycolmap.extract_features(
        database_path=DB_PATH,
        image_path=FRAMES_DIR,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extract_opts
    )
    print(f"    Feature extraction completed in {time.perf_counter() - t_feat:.2f}s")

    # 3. Match Features (Sequential matching with overlap=10 and loop detection)
    print(f"\n[*] Matching Features (Sequential with overlap=10)...")
    t_match = time.perf_counter()

    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = 10
    pairing_opts.quadratic_overlap = True
    pairing_opts.loop_detection = True

    pycolmap.match_sequential(
        database_path=DB_PATH,
        pairing_options=pairing_opts
    )
    print(f"    Feature matching completed in {time.perf_counter() - t_match:.2f}s")

    # 4. Incremental Mapping / Reconstruction
    print(f"\n[*] Running Incremental Structure-from-Motion (Bundle Adjustment)...")
    t_sfm = time.perf_counter()

    pipe_opts = pycolmap.IncrementalPipelineOptions()
    pipe_opts.min_num_matches = 15

    maps = pycolmap.incremental_mapping(
        database_path=DB_PATH,
        image_path=FRAMES_DIR,
        output_path=REC_DIR,
        options=pipe_opts
    )
    print(f"    Incremental SfM completed in {time.perf_counter() - t_sfm:.2f}s")

    if not maps:
        print("[!] Error: Incremental mapping did not produce any reconstructions.")
        sys.exit(1)

    rec = maps[0]
    num_reg = rec.num_reg_images()
    num_pts = rec.num_points3D()
    mean_err = rec.compute_mean_reprojection_error()
    track_len = rec.compute_mean_track_length()

    print("\n" + "=" * 60)
    print("PYCOLMAP PILOT SFM RESULTS:")
    print("=" * 60)
    print(f"  Registered Images:      {num_reg} / {len(frame_paths)} ({num_reg/len(frame_paths)*100:.1f}%)")
    print(f"  Sparse 3D Points:       {num_pts:,}")
    print(f"  Mean Reprojection Error: {mean_err:.4f} pixels")
    print(f"  Mean Track Length:      {track_len:.2f} observations / point")

    # Export sparse point cloud to PLY
    sparse_ply_path = PILOT_DIR / "pilot_sparse_points.ply"
    pts_3d = []
    colors_3d = []
    for pt_id, pt in rec.points3D.items():
        pts_3d.append(pt.xyz)
        colors_3d.append(pt.color / 255.0)

    pts_3d = np.array(pts_3d)
    colors_3d = np.array(colors_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d)
    o3d.io.write_point_cloud(str(sparse_ply_path), pcd)
    print(f"  Saved Sparse PLY:       {sparse_ply_path.name} ({len(pts_3d):,} points)")

    # Save summary JSON
    summary = {
        "status": "SUCCESS",
        "num_input_frames": len(frame_paths),
        "num_registered_frames": num_reg,
        "registration_rate_pct": round(num_reg / len(frame_paths) * 100.0, 2),
        "num_3d_points": num_pts,
        "mean_reprojection_error_px": round(mean_err, 4),
        "mean_track_length": round(track_len, 2),
        "runtime_seconds": round(time.perf_counter() - t0, 2)
    }
    with open(PILOT_DIR / "pilot_sfm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Total Runtime:          {time.perf_counter() - t0:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
