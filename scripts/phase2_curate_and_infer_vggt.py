r"""
F.O.R.G.E. — Phase 2: Keyframe Curation & Multi-Window VGGT Inference (VRAM-Optimized)
Video: WhatsApp Video 2026-11-01 at 14.28.11.mp4
Output: outputs/indoor/digital_twin_reconstruction_video_20261101_142811/
"""

import os
import sys
import gc
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vggt.inference import VGGTInferencePipeline

EXP_DIR = PROJECT_ROOT / "outputs/indoor/digital_twin_reconstruction_video_20261101_142811"
FRAMES_DIR = EXP_DIR / "frames"
GEOM_DIR = EXP_DIR / "camera_geometry"
WIN_CACHE_DIR = GEOM_DIR / "window_cache"

for d in [FRAMES_DIR, GEOM_DIR, WIN_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76, flush=True)
print("  F.O.R.G.E. PHASE 2: VGGT INFERENCE & GEOMETRIC CHAINING (VRAM OPTIMIZED)", flush=True)
print(f"  Target Dir: {EXP_DIR}", flush=True)
print("=" * 76, flush=True)

# Frames are already curated and saved in FRAMES_DIR
frame_files = sorted(list(FRAMES_DIR.glob("frame_*.png")))
NUM_KEYFRAMES = len(frame_files)
print(f"[*] Found {NUM_KEYFRAMES} curated keyframes in {FRAMES_DIR}", flush=True)

saved_frame_paths = [str(f) for f in frame_files]

# Window design: Window size = 5, step = 3. Overlap = 2 frames per pair.
# This keeps peak VRAM < 3.8 GB on the 8 GB GPU.
WINDOW_SIZE = 5
WINDOW_STEP = 3
windows = []
for start in range(0, NUM_KEYFRAMES - WINDOW_SIZE + 1, WINDOW_STEP):
    end = start + WINDOW_SIZE
    windows.append((start, end))

if windows[-1][1] < NUM_KEYFRAMES:
    windows.append((NUM_KEYFRAMES - WINDOW_SIZE, NUM_KEYFRAMES))

print(f"[*] Configured {len(windows)} overlapping windows (size {WINDOW_SIZE}, overlap {WINDOW_SIZE - WINDOW_STEP}):", flush=True)
for w_idx, (w_start, w_end) in enumerate(windows):
    cache_file = WIN_CACHE_DIR / f"window_{w_idx:02d}.npz"
    status = "CACHED" if cache_file.exists() else "PENDING"
    print(f"    Window {w_idx:02d}: Frames [{w_start:02d} .. {w_end-1:02d}] -> {status}", flush=True)

# Initialize pipeline only if there are pending windows
pending_windows = [w for w_idx, w in enumerate(windows) if not (WIN_CACHE_DIR / f"window_{w_idx:02d}.npz").exists()]

if pending_windows:
    print(f"\n[*] Initializing VGGT-Omega Inference Engine on cuda:0 for {len(pending_windows)} pending windows...", flush=True)
    pipeline = VGGTInferencePipeline(device="cuda:0", image_resolution=512)
else:
    print("\n[*] All windows already computed in cache!", flush=True)
    pipeline = None

for w_idx, (w_start, w_end) in enumerate(windows):
    cache_file = WIN_CACHE_DIR / f"window_{w_idx:02d}.npz"
    if cache_file.exists():
        print(f"  [+] Window {w_idx:02d} loaded from cache: {cache_file.name}", flush=True)
        continue

    win_frame_paths = saved_frame_paths[w_start:w_end]
    print(f"\n[*] Running VGGT on Window {w_idx:02d} (frames {w_start}..{w_end-1})...", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        res = pipeline.run_inference(win_frame_paths)
    dt = time.perf_counter() - t0

    # Move strictly to CPU numpy immediately
    depths = res["depth"].squeeze(0).squeeze(-1).cpu().numpy().astype(np.float32)
    confs = res["depth_conf"].squeeze(0).cpu().numpy().astype(np.float32)
    exts = res["extrinsics"].squeeze(0).cpu().numpy().astype(np.float32)
    intrs = res["intrinsics"].squeeze(0).cpu().numpy().astype(np.float32)

    # Save to disk immediately
    np.savez_compressed(
        cache_file,
        start=w_start,
        end=w_end,
        depths=depths,
        confs=confs,
        extrinsics=exts,
        intrinsics=intrs
    )
    print(f"    [+] Window {w_idx:02d} finished in {dt:.2f}s and saved to {cache_file.name}", flush=True)

    # Aggressive memory cleanup
    del res
    del depths, confs, exts, intrs
    gc.collect()
    torch.cuda.empty_cache()

# Free model from GPU memory completely before geometry chaining
if pipeline is not None:
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print("[*] Pipeline unloaded from GPU memory.", flush=True)

# 3. Geometric Alignment and Chaining
print("\n" + "=" * 76, flush=True)
print("  STEP 3: MULTI-WINDOW GEOMETRIC ALIGNMENT & SCALE PROPAGATION", flush=True)
print("=" * 76, flush=True)

# Load all cached windows
window_results = []
for w_idx in range(len(windows)):
    cache_file = WIN_CACHE_DIR / f"window_{w_idx:02d}.npz"
    data = np.load(cache_file)
    window_results.append({
        "window_idx": w_idx,
        "start": int(data["start"]),
        "end": int(data["end"]),
        "depths": data["depths"],
        "confs": data["confs"],
        "extrinsics": data["extrinsics"],
        "intrinsics": data["intrinsics"]
    })

# Global arrays
global_extrinsics = [None] * NUM_KEYFRAMES
global_intrinsics = [None] * NUM_KEYFRAMES
global_depths = [None] * NUM_KEYFRAMES
global_confs = [None] * NUM_KEYFRAMES

# Window 0 sets reference world frame
w0 = window_results[0]
for local_i in range(w0["end"] - w0["start"]):
    k = w0["start"] + local_i
    global_extrinsics[k] = w0["extrinsics"][local_i].copy()
    global_intrinsics[k] = w0["intrinsics"][local_i].copy()
    global_depths[k] = w0["depths"][local_i].copy()
    global_confs[k] = w0["confs"][local_i].copy()

# Chain subsequent windows
for w_idx in range(1, len(windows)):
    w_curr = window_results[w_idx]
    w_prev = window_results[w_idx - 1]

    curr_start = w_curr["start"]
    curr_end = w_curr["end"]
    prev_start = w_prev["start"]
    prev_end = w_prev["end"]

    overlap_indices = list(range(curr_start, prev_end))
    print(f"\n[*] Aligning Window {w_idx:02d} (frames {curr_start}..{curr_end-1}) to world...", flush=True)
    print(f"    Overlapping frames with prior: {overlap_indices}", flush=True)

    # 1. Scale propagation via median depth ratio
    scale_ratios = []
    for f_k in overlap_indices:
        idx_in_curr = f_k - curr_start
        d_curr = w_curr["depths"][idx_in_curr]
        c_curr = w_curr["confs"][idx_in_curr]
        d_prev = global_depths[f_k]
        c_prev = global_confs[f_k]

        valid = (c_curr > 0.5) & (c_prev > 0.5) & (d_curr > 0.2) & (d_prev > 0.2)
        if np.sum(valid) > 500:
            ratio = np.median(d_prev[valid] / d_curr[valid])
            scale_ratios.append(ratio)

    scale_prop = float(np.median(scale_ratios)) if len(scale_ratios) > 0 else 1.0
    print(f"    Scale factor s = {scale_prop:.4f}", flush=True)

    # 2. Rotation and translation alignment
    # In VGGT extrinsics: X_cam = R @ X_world + t
    # Camera center C = - R.T @ t
    curr_centers = []
    world_centers = []
    R_align_candidates = []

    for f_k in overlap_indices:
        idx_in_curr = f_k - curr_start
        R_c = w_curr["extrinsics"][idx_in_curr, :3, :3]
        t_c = w_curr["extrinsics"][idx_in_curr, :3, 3]
        C_c = - R_c.T @ t_c * scale_prop

        R_w = global_extrinsics[f_k][:3, :3]
        t_w = global_extrinsics[f_k][:3, 3]
        C_w = - R_w.T @ t_w

        curr_centers.append(C_c)
        world_centers.append(C_w)

        # X_cam = R_c @ X_c + t_c = R_w @ X_w + t_w
        # X_w = R_curr_to_world @ X_c + t_curr_to_world
        # R_c = R_w @ R_curr_to_world => R_curr_to_world = R_w.T @ R_c... wait!
        # Let's verify:
        # X_cam = R_w @ (R_curr_to_world @ X_c + t_curr_to_world) + t_w
        #       = (R_w @ R_curr_to_world) @ X_c + (R_w @ t_curr_to_world + t_w)
        # For this to equal R_c @ X_c + t_c:
        # R_w @ R_curr_to_world = R_c  =>  R_curr_to_world = R_w^T @ R_c
        # And R_w @ t_curr_to_world + t_w = t_c  =>  t_curr_to_world = R_w^T @ (t_c - t_w)
        # In terms of camera centers:
        # C_w = - R_w^T @ t_w
        # C_w = R_curr_to_world @ C_c + t_curr_to_world
        # Let's check:
        # R_curr_to_world @ (- R_c^T @ t_c) + R_w^T @ (t_c - t_w)
        # = (R_w^T @ R_c) @ (- R_c^T @ t_c) + R_w^T @ t_c - R_w^T @ t_w
        # = - R_w^T @ t_c + R_w^T @ t_c - R_w^T @ t_w = - R_w^T @ t_w = C_w! Exactly consistent!
        R_cw = R_w.T @ R_c
        R_align_candidates.append(R_cw)

    curr_centers = np.array(curr_centers)
    world_centers = np.array(world_centers)

    # Average rotation via SVD
    R_mean = np.mean(R_align_candidates, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_curr_to_world = U @ Vt
    if np.linalg.det(R_curr_to_world) < 0:
        U[:, -1] *= -1
        R_curr_to_world = U @ Vt

    # Centroid translation
    c_curr_mean = curr_centers.mean(axis=0)
    c_world_mean = world_centers.mean(axis=0)
    t_curr_to_world = c_world_mean - R_curr_to_world @ c_curr_mean

    pred_world_centers = (R_curr_to_world @ curr_centers.T).T + t_curr_to_world
    res_err = np.linalg.norm(pred_world_centers - world_centers, axis=1).mean()
    print(f"    Camera center alignment residual: {res_err*100:.2f} cm", flush=True)

    # Transform current window to world
    for local_i in range(w_curr["end"] - w_curr["start"]):
        f_k = curr_start + local_i
        if global_extrinsics[f_k] is not None and f_k in overlap_indices:
            continue

        R_c = w_curr["extrinsics"][local_i, :3, :3]
        t_c = w_curr["extrinsics"][local_i, :3, 3]

        # Camera center in world
        C_c = - R_c.T @ t_c * scale_prop
        C_w = R_curr_to_world @ C_c + t_curr_to_world

        # R_w: Since R_c = R_w @ R_curr_to_world, R_w = R_c @ R_curr_to_world^T
        R_w = R_c @ R_curr_to_world.T
        t_w = - R_w @ C_w

        ext_w = np.zeros((3, 4), dtype=np.float32)
        ext_w[:3, :3] = R_w
        ext_w[:3, 3] = t_w

        global_extrinsics[f_k] = ext_w
        global_intrinsics[f_k] = w_curr["intrinsics"][local_i].copy()
        global_depths[f_k] = w_curr["depths"][local_i].copy() * scale_prop
        global_confs[f_k] = w_curr["confs"][local_i].copy()

# Save final world camera geometry
global_exts_arr = np.stack(global_extrinsics, axis=0)
global_intrs_arr = np.stack(global_intrinsics, axis=0)
global_depths_arr = np.stack(global_depths, axis=0)
global_confs_arr = np.stack(global_confs, axis=0)

np.savez_compressed(
    GEOM_DIR / "global_camera_poses_raw.npz",
    extrinsics=global_exts_arr,
    intrinsics=global_intrs_arr,
    depths=global_depths_arr,
    confs=global_confs_arr
)
print(f"\n[+] Successfully saved chained global camera poses for all {NUM_KEYFRAMES} frames!", flush=True)
print(f"    File: {GEOM_DIR / 'global_camera_poses_raw.npz'}", flush=True)
print("=" * 76, flush=True)
