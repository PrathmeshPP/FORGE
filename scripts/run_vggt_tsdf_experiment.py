r"""
F.O.R.G.E. — Major Indoor Reconstruction Experiment:
VGGT + Confidence-Guided Point Cloud + Open3D Scalable TSDF Fusion + Multi-View Photographic Texture Baking.

Target Video: C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-31 at 14.23.03.mp4
Output Root: outputs/indoor/vggt_tsdf_video_20261031_142303/
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import psutil
import trimesh
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _candidate_vggt in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega",
]:
    if os.path.isdir(_candidate_vggt) and _candidate_vggt not in sys.path:
        sys.path.insert(0, _candidate_vggt)

from src.vggt.inference import load_vggt_model, DEFAULT_CHECKPOINT_PATH
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from src.video.adaptive_frame_curator import AdaptiveIndoorFrameCurator
from src.reconstruction.point_cloud import depth_to_points, voxel_grid_fusion, save_ply
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    solve_umeyama_sim3,
    compute_floor_leveling_rotation
)
from src.reconstruction.stage2_texture_engine import (
    Stage2TextureEngine,
    compute_crease_aware_normals,
    save_mesh_ply_float32
)

VIDEO_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-31 at 14.23.03.mp4"
EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"


def get_nvidia_smi_vram_mb() -> float:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def get_system_ram_mb() -> float:
    return float(psutil.virtual_memory().used / (1024 ** 2))


def get_default_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ==============================================================================
# STAGE 1: ADAPTIVE 1.0 FPS FRAME CURATION
# ==============================================================================

def curate_indoor_video(
    video_path: str,
    out_dir: Path,
    target_base_fps: float = 1.0
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 1: ADAPTIVE KEYFRAME CURATION (1.0 FPS BASELINE + MOTION DISPARITY)")
    print("=" * 76)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "frame_curation_report.json"
    if report_path.exists():
        print(f"[*] Found existing curation report: {report_path.name}. Loading...")
        return json.loads(report_path.read_text())

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    curator = AdaptiveIndoorFrameCurator(
        video_path,
        min_sharpness=50.0,
        min_luminance=25.0,
        max_luminance=235.0,
        min_contrast=18.0
    )

    print(f"[*] Scanning video candidates: {curator.total_frames} frames ({curator.duration_sec:.2f}s, {curator.fps:.2f} FPS)...")
    all_candidates = curator.scan_all_candidates()

    # Base ~1.0 FPS selection
    base_interval_sec = 1.0 / target_base_fps
    num_intervals = int(math.ceil(curator.duration_sec * target_base_fps))

    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(nfeatures=400)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    selected_indices = set()
    base_indices = []

    for inv_idx in range(num_intervals):
        t_start = inv_idx * base_interval_sec
        t_end = min((inv_idx + 1) * base_interval_sec, curator.duration_sec)
        f_start = int(t_start * curator.fps)
        f_end = min(int(t_end * curator.fps), curator.total_frames)
        if f_start >= curator.total_frames:
            break
        window_cands = all_candidates[f_start:f_end]
        if not window_cands:
            continue
        valid_cands = [c for c in window_cands if c.rejection_reason is None]
        best_cand = max(valid_cands or window_cands, key=lambda c: c.sharpness * (c.contrast / 50.0))
        selected_indices.add(best_cand.frame_idx)
        base_indices.append(best_cand.frame_idx)

    # Adaptive keyframes: add intermediate frames where camera rotates or translates rapidly
    sorted_base = sorted(list(selected_indices))
    adaptive_added = []

    for i in range(len(sorted_base) - 1):
        idx_a = sorted_base[i]
        idx_b = sorted_base[i + 1]
        mid = (idx_a + idx_b) // 2

        # Check motion between base frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
        ret_a, frame_a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
        ret_b, frame_b = cap.read()

        if ret_a and ret_b:
            ga = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), (160, 160))
            gb = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), (160, 160))
            disp = float(np.mean(cv2.absdiff(ga, gb)))

            # If substantial camera displacement or rotation occurred, insert best intermediate frame
            if disp > 38.0 and (idx_b - idx_a) >= 12:
                mid_cands = all_candidates[max(idx_a + 2, mid - 4): min(idx_b - 2, mid + 4)]
                if mid_cands:
                    best_mid = max(mid_cands, key=lambda c: c.sharpness)
                    if best_mid.frame_idx not in selected_indices:
                        selected_indices.add(best_mid.frame_idx)
                        adaptive_added.append(best_mid.frame_idx)

    final_indices = sorted(list(selected_indices))
    curated_records = []

    prev_gray = None
    prev_des = None
    prev_kp = None

    print(f"[*] Extracting and saving {len(final_indices)} curated frames ({len(base_indices)} base + {len(adaptive_added)} adaptive)...")

    for rank, f_idx in enumerate(final_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        cand = all_candidates[f_idx]
        file_name = f"frame_{rank:03d}_f{f_idx:05d}_{cand.timestamp_sec:05.2f}s.png"
        file_path = frames_dir / file_name
        cv2.imwrite(str(file_path), frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        disp = 0.0
        ovlp = 100.0
        rot = 0.0

        if prev_gray is not None:
            g1 = cv2.resize(prev_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            disp = float(np.mean(cv2.absdiff(g1, g2)))

            if prev_des is not None and des is not None and len(prev_des) > 10 and len(des) > 10:
                matches = bf.match(prev_des, des)
                ovlp = float(len(matches) / min(len(prev_des), len(des)) * 100.0)
                if len(matches) >= 8:
                    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M is not None:
                        rot = abs(float(np.degrees(math.atan2(M[1, 0], M[0, 0]))))

        prev_gray = gray
        prev_des = des
        prev_kp = kp

        curated_records.append({
            "rank": rank,
            "frame_idx": f_idx,
            "timestamp_sec": cand.timestamp_sec,
            "file_path": str(file_path.resolve()),
            "sharpness": cand.sharpness,
            "luminance": cand.luminance,
            "contrast": cand.contrast,
            "disp": round(disp, 2),
            "rotation_deg": round(rot, 2),
            "overlap_pct": round(ovlp, 2),
            "is_adaptive": f_idx in adaptive_added
        })

    cap.release()
    cur_time = time.perf_counter() - t0

    report = {
        "video_path": str(video_path),
        "total_candidate_frames": curator.total_frames,
        "duration_sec": round(curator.duration_sec, 2),
        "fps": round(curator.fps, 2),
        "resolution": f"{curator.width}x{curator.height}",
        "selected_total_frames": len(curated_records),
        "base_keyframes": len(base_indices),
        "adaptive_keyframes": len(adaptive_added),
        "effective_sampling_fps": round(len(curated_records) / curator.duration_sec, 3),
        "curation_time_sec": round(cur_time, 2),
        "frames": curated_records
    }

    report_path = out_dir / "frame_curation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[+] Curation complete in {cur_time:.2f}s: {len(curated_records)} frames saved to {frames_dir.name}")
    return report


# ==============================================================================
# STAGE 2: VGGT INFERENCE & PARALLAX-GATED WINDOW CHAINING
# ==============================================================================

def run_vggt_with_robust_chaining(
    curation_report: Dict[str, Any],
    out_dir: Path,
    image_resolution: int = 320,
    window_size: int = 6,
    overlap: int = 3
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 2: VGGT INFERENCE WITH ROBUST SIM(3) SCALE-BOUNDED WINDOW CHAINING")
    print("=" * 76)
    cam_dir = out_dir / "camera_geometry"
    cam_dir.mkdir(parents=True, exist_ok=True)

    poses_file = cam_dir / "camera_poses.npz"
    depth_file = cam_dir / "depth_predictions.npz"
    perf_file = cam_dir / "vggt_inference_performance.json"
    if poses_file.exists() and depth_file.exists():
        print(f"[*] Found existing VGGT poses and depth: {poses_file.name}. Loading...")
        poses = np.load(poses_file)
        depths = np.load(depth_file)
        perf = json.loads(perf_file.read_text()) if perf_file.exists() else {}
        return {
            "extrinsics": poses["extrinsics"],
            "intrinsics": poses["intrinsics"],
            "depth": depths["depth"],
            "confidence": depths["confidence"],
            "performance": perf
        }

    frame_records = curation_report["frames"]
    image_paths = [r["file_path"] for r in frame_records]
    N = len(image_paths)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading VGGT-Omega checkpoint into {device}...")
    t_vggt_load = time.perf_counter()
    model = load_vggt_model(DEFAULT_CHECKPOINT_PATH, device)
    print(f"    Loaded in {time.perf_counter() - t_vggt_load:.2f}s")

    # Generate sliding windows
    step = max(1, window_size - overlap)
    windows = []
    start = 0
    while start + window_size <= N:
        windows.append(list(range(start, start + window_size)))
        start += step
    if start < N:
        windows.append(list(range(max(0, N - window_size), N)))

    print(f"[*] Processing {len(windows)} overlapping windows across {N} frames...")

    global_frames = {}
    alignment_logs = []
    total_infer_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    for w_idx, win in enumerate(windows):
        t_w0 = time.perf_counter()
        win_paths = [image_paths[i] for i in win]
        inp = load_and_preprocess_images(win_paths, image_resolution=image_resolution).to(device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = model(inp)
            ext, intr = encoding_to_camera(out.get("pose_enc"), image_size_hw=inp.shape[-2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        w_time = time.perf_counter() - t_w0
        total_infer_time += w_time

        ext_np = ext.squeeze(0).cpu().numpy()       # (W, 3, 4)
        intr_np = intr.squeeze(0).cpu().numpy()     # (W, 3, 3)
        depth_np = out["depth"].squeeze(0).cpu().numpy()  # (W, H, W, 1)
        conf_np = out["depth_conf"].squeeze(0).cpu().numpy()  # (W, H, W)

        if w_idx == 0:
            for lp, gp in enumerate(win):
                R = ext_np[lp, :3, :3]
                t = ext_np[lp, :3, 3]
                c = -R.T @ t
                global_frames[gp] = {
                    "ext": ext_np[lp].copy(),
                    "intr": intr_np[lp].copy(),
                    "depth": depth_np[lp].copy(),
                    "conf": conf_np[lp].copy(),
                    "center": c,
                    "scale": 1.0
                }
            alignment_logs.append({
                "window": 0, "scale_sw": 1.0, "residual_cm": 0.0, "mode": "init"
            })
            print(f"  [Window 00/{len(windows):02d}] (frames {win[0]:03d}:{win[-1]:03d}): {w_time:.2f}s | Reference initialized")
        else:
            shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
            shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]

            pts_loc = []
            pts_glob = []
            H_d, W_d = depth_np.shape[1], depth_np.shape[2]
            sub_y = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 18, dtype=int)
            sub_x = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 18, dtype=int)

            for lp, gp in zip(shared_local, shared_global):
                ge = global_frames[gp]
                pg = depth_to_points(ge["depth"], ge["ext"], ge["intr"])
                pl = depth_to_points(depth_np[lp], ext_np[lp], intr_np[lp])

                c_loc = conf_np[lp].squeeze()
                c_glob = ge["conf"].squeeze()
                mask = (c_loc[sub_y][:, sub_x] >= 0.70) & (c_glob[sub_y][:, sub_x] >= 0.70)
                pl_sub = pl[sub_y][:, sub_x][mask]
                pg_sub = pg[sub_y][:, sub_x][mask]
                if len(pl_sub) > 10:
                    pts_loc.append(pl_sub)
                    pts_glob.append(pg_sub)

            if pts_loc and pts_glob:
                X = np.vstack(pts_loc)
                Y = np.vstack(pts_glob)

                # Robust Sim(3) with Planar Degeneracy Protection
                s_w, R_w, t_w, med_err = solve_umeyama_sim3(X, Y)

                # Check covariance singular value condition ratio
                X_c = X - X.mean(axis=0)
                cov = (X_c.T @ X_c) / len(X)
                eigvals = np.linalg.eigvalsh(cov)
                cond_ratio = math.sqrt(max(1e-8, eigvals[0])) / math.sqrt(max(1e-8, eigvals[-1]))

                mode = "Sim(3) full"
                # If points are coplanar or scale is out of reasonable physical bounds [0.75, 1.35], regularize!
                if cond_ratio < 0.08 or s_w < 0.75 or s_w > 1.35:
                    s_w_clamped = np.clip(s_w, 0.85, 1.20)
                    mode = f"Regularized (orig s={s_w:.3f}, cond={cond_ratio:.3f})"
                    s_w = float(s_w_clamped)
            else:
                s_w, R_w, t_w, med_err = 1.0, np.eye(3), np.zeros(3), 0.0
                mode = "Fallback identity"

            alignment_logs.append({
                "window": w_idx,
                "scale_sw": round(float(s_w), 4),
                "residual_cm": round(float(med_err * 100.0), 2),
                "mode": mode
            })
            print(f"  [Window {w_idx:02d}/{len(windows):02d}] (frames {win[0]:03d}:{win[-1]:03d}): {w_time:.2f}s | s_w={s_w:.4f}, res={med_err*100:.1f}cm | {mode}")

            for lp, gp in enumerate(win):
                if gp not in global_frames:
                    R_local = ext_np[lp, :3, :3]
                    t_local = ext_np[lp, :3, 3]
                    c_local = -R_local.T @ t_local

                    c_global = s_w * (R_w @ c_local) + t_w
                    R_global = R_local @ R_w.T
                    t_global = -R_global @ c_global

                    ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
                    depth_global = s_w * depth_np[lp]

                    global_frames[gp] = {
                        "ext": ext_global.astype(np.float32),
                        "intr": intr_np[lp].astype(np.float32),
                        "depth": depth_global.astype(np.float32),
                        "conf": conf_np[lp].copy(),
                        "center": c_global,
                        "scale": s_w
                    }

    # Assemble global arrays
    all_ext = np.zeros((N, 3, 4), dtype=np.float32)
    all_intr = np.zeros((N, 3, 3), dtype=np.float32)
    H_d, W_d = global_frames[0]["depth"].shape[:2]
    all_depth = np.zeros((N, H_d, W_d), dtype=np.float32)
    all_conf = np.zeros((N, H_d, W_d), dtype=np.float32)

    for i in range(N):
        all_ext[i] = global_frames[i]["ext"]
        all_intr[i] = global_frames[i]["intr"]
        d = global_frames[i]["depth"]
        if d.ndim == 3:
            d = d.squeeze(-1)
        all_depth[i] = d
        c = global_frames[i]["conf"]
        if c.ndim == 3:
            c = c.squeeze(-1)
        all_conf[i] = c

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    smi_vram = get_nvidia_smi_vram_mb()

    # Save outputs
    poses_file = cam_dir / "camera_poses.npz"
    depth_file = cam_dir / "depth_predictions.npz"
    np.savez_compressed(poses_file, extrinsics=all_ext, intrinsics=all_intr)
    np.savez_compressed(depth_file, depth=all_depth, confidence=all_conf)

    # Compute trajectory diagnostics
    centers = np.array([-all_ext[i, :3, :3].T @ all_ext[i, :3, 3] for i in range(N)])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    total_path = float(np.sum(steps))
    bbox_extents = np.ptp(centers, axis=0).round(4).tolist()

    mean_res = float(np.mean([a["residual_cm"] for a in alignment_logs[1:]])) if len(alignment_logs) > 1 else 0.0

    perf_report = {
        "num_frames": N,
        "num_windows": len(windows),
        "total_inference_sec": round(total_infer_time, 2),
        "sustained_fps": round(N / total_infer_time, 2),
        "mean_sim3_residual_cm": round(mean_res, 2),
        "total_path_length_m": round(total_path, 3),
        "trajectory_bbox_extent_m": bbox_extents,
        "peak_pytorch_allocated_mb": round(peak_alloc, 1),
        "peak_pytorch_reserved_mb": round(peak_res, 1),
        "peak_nvidia_smi_mb": round(smi_vram, 1),
        "alignment_logs": alignment_logs
    }
    (cam_dir / "vggt_inference_performance.json").write_text(json.dumps(perf_report, indent=2))
    print(f"[+] VGGT complete: {N} cameras chained | Path: {total_path:.2f}m | Extent: {bbox_extents} | SMI: {smi_vram:.1f} MB")

    return {
        "extrinsics": all_ext,
        "intrinsics": all_intr,
        "depth": all_depth,
        "confidence": all_conf,
        "performance": perf_report
    }


# ==============================================================================
# STAGE 3: CONFIDENCE-GUIDED POINT CLOUD GENERATION & FLOOR LEVELING
# ==============================================================================

def generate_confidence_guided_point_cloud(
    curation_report: Dict[str, Any],
    vggt_data: Dict[str, Any],
    out_dir: Path,
    min_confidence: float = 0.75,
    min_depth: float = 0.20,
    max_depth: float = 5.5,
    voxel_size: float = 0.012
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 3: CONFIDENCE-GUIDED POINT CLOUD GENERATION & FLOOR LEVELING")
    print("=" * 76)
    pcd_dir = out_dir / "point_clouds"
    pcd_dir.mkdir(parents=True, exist_ok=True)

    leveled_ply = pcd_dir / "filtered_point_cloud_leveled.ply"
    plane_json = pcd_dir / "plane_detection.json"
    stats_json = pcd_dir / "point_cloud_statistics.json"
    if leveled_ply.exists() and plane_json.exists():
        print(f"[*] Found existing filtered leveled point cloud: {leveled_ply.name}. Loading...")
        pcd_o3d = o3d.io.read_point_cloud(str(leveled_ply))
        clean_pts_leveled = np.asarray(pcd_o3d.points, dtype=np.float32)
        clean_cols = (np.asarray(pcd_o3d.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        plane_res = json.loads(plane_json.read_text())
        R_floor = np.eye(3, dtype=np.float32)
        horiz_planes = [p for p in plane_res["detected_planes"] if "horizontal" in p["type"]]
        if horiz_planes:
            R_floor = compute_floor_leveling_rotation(np.array(horiz_planes[0]["normal"])).astype(np.float32)
        stats = json.loads(stats_json.read_text()) if stats_json.exists() else {}
        return {
            "clean_pts_leveled": clean_pts_leveled,
            "clean_cols": clean_cols,
            "R_floor": R_floor,
            "stats": stats
        }

    frame_records = curation_report["frames"]
    extrinsics = vggt_data["extrinsics"]
    intrinsics = vggt_data["intrinsics"]
    depth_maps = vggt_data["depth"]
    confidence_maps = vggt_data["confidence"]
    N = len(frame_records)

    all_raw_pts = []
    all_raw_cols = []
    all_valid_pts = []
    all_valid_cols = []
    total_pixels = 0
    valid_pixels = 0

    print(f"[*] Backprojecting {N} frames with confidence gating (conf >= {min_confidence}, depth in [{min_depth}, {max_depth}]m)...")

    for i in range(N):
        d = depth_maps[i]
        c = confidence_maps[i]
        im_bgr = cv2.imread(frame_records[i]["file_path"])
        H_d, W_d = d.shape[:2]
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)

        total_pixels += (H_d * W_d)

        pts_3d = depth_to_points(d, extrinsics[i], intrinsics[i])
        mask = (d >= min_depth) & (d <= max_depth) & (c >= min_confidence)
        valid_pixels += int(np.sum(mask))

        # Sample raw points for baseline comparison
        sub_step = 8
        all_raw_pts.append(pts_3d[::sub_step, ::sub_step].reshape(-1, 3))
        all_raw_cols.append(im_rgb[::sub_step, ::sub_step].reshape(-1, 3))

        if np.any(mask):
            all_valid_pts.append(pts_3d[mask])
            all_valid_cols.append(im_rgb[mask])

    raw_pts = np.vstack(all_raw_pts)
    raw_cols = np.vstack(all_raw_cols)
    valid_pts = np.vstack(all_valid_pts)
    valid_cols = np.vstack(all_valid_cols)

    print(f"    Raw backprojected: {len(raw_pts):,} pts | Valid masked: {len(valid_pts):,} pts ({valid_pixels/total_pixels*100:.1f}%)")

    # Scale-calibrated voxel grid fusion
    fused_pts, fused_cols = voxel_grid_fusion(valid_pts, valid_cols, voxel_size=voxel_size)
    print(f"    Voxel fused ({voxel_size*100:.1f}cm): {len(fused_pts):,} points")

    # Statistical outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(fused_cols.astype(np.float64) / 255.0)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    clean_pcd = pcd.select_by_index(ind)
    clean_pts = np.asarray(clean_pcd.points, dtype=np.float32)
    clean_cols = (np.asarray(clean_pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    print(f"    Statistical outlier removal: {len(clean_pts):,} clean points ({len(clean_pts)/len(fused_pts)*100:.1f}% retained)")

    # Save raw & clean un-leveled point clouds
    save_ply(raw_pts, raw_cols, str(pcd_dir / "raw_point_cloud.ply"))
    save_ply(clean_pts, clean_cols, str(pcd_dir / "filtered_point_cloud_raw_coords.ply"))

    # Dominant Plane Detection (Floor Leveling)
    config = IndoorPipelineConfig()
    engine = IndoorReconstructionEngine(config)
    plane_res = engine.detect_room_planes(clean_pts)
    (pcd_dir / "plane_detection.json").write_text(json.dumps(plane_res, indent=2))

    R_floor = np.eye(3, dtype=np.float32)
    horiz_planes = [p for p in plane_res["detected_planes"] if "horizontal" in p["type"]]
    if horiz_planes:
        best_floor = horiz_planes[0]
        R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"])).astype(np.float32)
        print(f"    Floor plane detected: inliers={best_floor['inlier_count']:,} ({best_floor['inlier_percentage']}%) | Normal={best_floor['normal']}")
        print(f"    Applying floor leveling rotation...")

    clean_pts_leveled = (R_floor @ clean_pts.T).T.astype(np.float32)
    save_ply(clean_pts_leveled, clean_cols, str(pcd_dir / "filtered_point_cloud_leveled.ply"))

    pcd_stats = {
        "total_pixels_evaluated": total_pixels,
        "valid_pixels_passed": valid_pixels,
        "valid_pixel_pct": round(valid_pixels / total_pixels * 100.0, 2),
        "raw_points_saved": len(raw_pts),
        "valid_points_masked": len(valid_pts),
        "voxel_fused_points": len(fused_pts),
        "clean_points": len(clean_pts),
        "clean_retention_rate_pct": round(len(clean_pts) / max(1, len(valid_pts)) * 100.0, 2),
        "pcd_extents_m": np.ptp(clean_pts_leveled, axis=0).round(4).tolist(),
        "floor_rotation_applied": horiz_planes is not None and len(horiz_planes) > 0
    }
    (pcd_dir / "point_cloud_statistics.json").write_text(json.dumps(pcd_stats, indent=2))
    print(f"[+] Point cloud complete: Extents {pcd_stats['pcd_extents_m']} m | Saved to {pcd_dir.name}")

    return {
        "raw_pts": raw_pts,
        "raw_cols": raw_cols,
        "clean_pts": clean_pts,
        "clean_cols": clean_cols,
        "clean_pts_leveled": clean_pts_leveled,
        "R_floor": R_floor,
        "stats": pcd_stats
    }


# ==============================================================================
# STAGE 4: OPEN3D SCALABLE TSDF VOLUME CONTROLLED RESOLUTION BENCHMARK
# ==============================================================================

def run_open3d_scalable_tsdf_benchmark(
    curation_report: Dict[str, Any],
    vggt_data: Dict[str, Any],
    pcd_data: Dict[str, Any],
    out_dir: Path
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 4: OPEN3D SCALABLE TSDF VOLUME BENCHMARK (10mm vs 15mm vs 20mm)")
    print("=" * 76)
    tsdf_dir = out_dir / "tsdf_fusion"
    tsdf_dir.mkdir(parents=True, exist_ok=True)

    summary_file = tsdf_dir / "tsdf_benchmark_summary.json"
    mesh_15mm_file = tsdf_dir / "voxel_15mm" / "tsdf_mesh.ply"
    if summary_file.exists() and mesh_15mm_file.exists():
        print(f"[*] Found existing TSDF benchmark summary and meshes: {summary_file.name}. Loading...")
        bench = json.loads(summary_file.read_text())
        best_mesh = o3d.io.read_triangle_mesh(str(mesh_15mm_file))
        best_mesh.compute_vertex_normals()
        return {
            "benchmark": bench,
            "best_id": "voxel_15mm",
            "best_mesh": best_mesh
        }

    frame_records = curation_report["frames"]
    extrinsics = vggt_data["extrinsics"]
    intrinsics = vggt_data["intrinsics"]
    depth_maps = vggt_data["depth"]
    confidence_maps = vggt_data["confidence"]
    R_floor = pcd_data["R_floor"]
    N = len(frame_records)

    voxel_configs = [
        {"id": "voxel_10mm", "name": "TSDF 10mm (High Detail)", "voxel_m": 0.010, "trunc_m": 0.030},
        {"id": "voxel_15mm", "name": "TSDF 15mm (Balanced)", "voxel_m": 0.015, "trunc_m": 0.045},
        {"id": "voxel_20mm", "name": "TSDF 20mm (Fast Smooth)", "voxel_m": 0.020, "trunc_m": 0.060}
    ]

    benchmark_results = {}
    extracted_meshes = {}

    # Pre-load aligned RGB-D pairs
    print(f"[*] Preparing {N} aligned RGB-D observation pairs (depth scale 1.0m, confidence filtered)...")
    H_d, W_d = depth_maps[0].shape[:2]

    rgbd_pairs = []
    intr_objects = []
    ext_4x4_list = []

    for i in range(N):
        d = depth_maps[i].copy()
        c = confidence_maps[i]
        # Invalidate low-confidence and out-of-bounds pixels
        invalid_mask = (d < 0.20) | (d > 5.5) | (c < 0.75)
        d[invalid_mask] = 0.0  # Open3D treats <= 0 as invalid

        im_bgr = cv2.imread(frame_records[i]["file_path"])
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)

        o3d_color = o3d.geometry.Image(im_rgb)
        o3d_depth = o3d.geometry.Image(d)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=5.5, convert_rgb_to_intensity=False
        )
        rgbd_pairs.append(rgbd)

        K = intrinsics[i]
        intr_obj = o3d.camera.PinholeCameraIntrinsic(
            width=W_d, height=H_d, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2])
        )
        intr_objects.append(intr_obj)

        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :4] = extrinsics[i]
        ext_4x4_list.append(ext_4x4)

    # Execute controlled TSDF benchmark across voxel sizes
    for cfg in voxel_configs:
        v_id = cfg["id"]
        v_name = cfg["name"]
        voxel_len = cfg["voxel_m"]
        sdf_trunc = cfg["trunc_m"]

        print(f"\n[*] Running Open3D ScalableTSDFVolume: {v_name} (voxel={voxel_len*1000:.0f}mm, trunc={sdf_trunc*1000:.0f}mm)...")
        v_out_dir = tsdf_dir / v_id
        v_out_dir.mkdir(parents=True, exist_ok=True)

        ram_before = get_system_ram_mb()
        t0_tsdf = time.perf_counter()

        # Instantiate CPU ScalableTSDFVolume with RGB8 color integration
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_len,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i in range(N):
            volume.integrate(rgbd_pairs[i], intr_objects[i], ext_4x4_list[i])

        integration_sec = time.perf_counter() - t0_tsdf
        print(f"    Integration of {N} frames completed in {integration_sec:.2f}s ({N/integration_sec:.1f} FPS)")

        # Extract colored triangle mesh
        t0_extract = time.perf_counter()
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        extract_mesh_sec = time.perf_counter() - t0_extract

        # Extract colored point cloud
        t0_pcd = time.perf_counter()
        pcd = volume.extract_point_cloud()
        extract_pcd_sec = time.perf_counter() - t0_pcd

        ram_after = get_system_ram_mb()

        # Clean disconnected floating fragments & non-manifold edges
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()

        # Apply floor leveling rotation for upright viewing
        mesh.rotate(R_floor, center=(0, 0, 0))
        pcd.rotate(R_floor, center=(0, 0, 0))

        # Save deliverables
        mesh_ply_path = v_out_dir / "tsdf_mesh.ply"
        pcd_ply_path = v_out_dir / "tsdf_point_cloud.ply"
        o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
        o3d.io.write_point_cloud(str(pcd_ply_path), pcd)

        extents = (mesh.get_max_bound() - mesh.get_min_bound()).round(4).tolist()
        num_v = len(mesh.vertices)
        num_t = len(mesh.triangles)
        num_p = len(pcd.points)

        res_info = {
            "config_id": v_id,
            "voxel_size_mm": voxel_len * 1000,
            "sdf_trunc_mm": sdf_trunc * 1000,
            "integration_time_sec": round(integration_sec, 2),
            "extract_mesh_time_sec": round(extract_mesh_sec, 2),
            "total_tsdf_time_sec": round(integration_sec + extract_mesh_sec + extract_pcd_sec, 2),
            "peak_system_ram_mb": round(max(ram_before, ram_after), 1),
            "mesh_vertices": num_v,
            "mesh_triangles": num_t,
            "pcd_points": num_p,
            "mesh_extents_m": extents,
            "is_watertight": mesh.is_watertight(),
            "mesh_ply": str(mesh_ply_path.relative_to(EXP_DIR)),
            "pcd_ply": str(pcd_ply_path.relative_to(EXP_DIR))
        }
        benchmark_results[v_id] = res_info
        extracted_meshes[v_id] = mesh
        print(f"    Mesh: {num_v:,} V | {num_t:,} F | PCD: {num_p:,} pts | Extent: {extents} m")

    (tsdf_dir / "tsdf_benchmark_summary.json").write_text(json.dumps(benchmark_results, indent=2))
    print(f"[+] TSDF Benchmark Complete! Results saved to {tsdf_dir.name}")

    # Select winning TSDF resolution:
    # 15mm offers ideal balance of noise suppression and sharp geometry
    best_id = "voxel_15mm"
    best_mesh = extracted_meshes[best_id]

    return {
        "benchmark": benchmark_results,
        "best_id": best_id,
        "best_mesh": best_mesh
    }


# ==============================================================================
# STAGE 5: SCREENED POISSON COMPARISON BASELINE
# ==============================================================================

def run_poisson_baseline(
    pcd_data: Dict[str, Any],
    vggt_data: Dict[str, Any],
    out_dir: Path,
    poisson_depth: int = 9
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 5: SCREENED POISSON SURFACE RECONSTRUCTION (BASELINE COMPARISON)")
    print("=" * 76)
    mesh_dir = out_dir / "poisson_baseline"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    poisson_ply = mesh_dir / "poisson_mesh.ply"
    stats_file = mesh_dir / "poisson_statistics.json"
    if poisson_ply.exists():
        print(f"[*] Found existing Poisson baseline mesh: {poisson_ply.name}. Loading...")
        mesh = o3d.io.read_triangle_mesh(str(poisson_ply))
        mesh.compute_vertex_normals()
        stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
        return {
            "mesh": mesh,
            "stats": stats
        }

    clean_pts = pcd_data["clean_pts_leveled"]
    clean_cols = pcd_data["clean_cols"]
    extrinsics = vggt_data["extrinsics"]
    R_floor = pcd_data["R_floor"]
    N = len(extrinsics)

    cam_centers = []
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        c = -R.T @ t
        cam_centers.append(R_floor @ c)

    config = IndoorPipelineConfig(poisson_depth=poisson_depth, voxel_size=0.012)
    engine = IndoorReconstructionEngine(config)

    t0 = time.perf_counter()
    raw_mesh, post_mesh = engine.reconstruct_poisson_mesh(
        points=clean_pts,
        colors=clean_cols,
        cam_centers=cam_centers,
        depth=poisson_depth
    )
    poisson_sec = time.perf_counter() - t0

    # Save mesh
    poisson_ply = mesh_dir / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(poisson_ply), post_mesh)

    extents = (post_mesh.get_max_bound() - post_mesh.get_min_bound()).round(4).tolist()
    poisson_stats = {
        "method": "Screened Poisson",
        "poisson_depth": poisson_depth,
        "generation_time_sec": round(poisson_sec, 2),
        "vertices": len(post_mesh.vertices),
        "triangles": len(post_mesh.triangles),
        "mesh_extents_m": extents,
        "ply_path": str(poisson_ply.relative_to(EXP_DIR))
    }
    (mesh_dir / "poisson_statistics.json").write_text(json.dumps(poisson_stats, indent=2))
    print(f"[+] Poisson baseline complete in {poisson_sec:.2f}s: {len(post_mesh.vertices):,} V | {len(post_mesh.triangles):,} F | Extent: {extents} m")

    return {
        "mesh": post_mesh,
        "stats": poisson_stats
    }


# ==============================================================================
# STAGE 6: MULTI-VIEW PHOTOGRAPHIC TEXTURE BAKING ON TSDF MESH
# ==============================================================================

def bake_photographic_texture_on_tsdf(
    tsdf_mesh: o3d.geometry.TriangleMesh,
    curation_report: Dict[str, Any],
    vggt_data: Dict[str, Any],
    pcd_data: Dict[str, Any],
    out_dir: Path
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 6: MULTI-VIEW PHOTOGRAPHIC TEXTURE BAKING ON TSDF MESH")
    print("=" * 76)
    tex_dir = out_dir / "textured_tsdf_model"
    tex_dir.mkdir(parents=True, exist_ok=True)

    t0_tex = time.perf_counter()

    # Edge-preserving quadric decimation to 250,000 triangles for optimal real-time viewer performance
    target_triangles = 250000
    mesh_in = o3d.geometry.TriangleMesh(tsdf_mesh)
    if len(mesh_in.triangles) > target_triangles:
        print(f"[*] Decimating TSDF mesh from {len(mesh_in.triangles):,} to {target_triangles:,} triangles...")
        mesh_in = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh_in.remove_degenerate_triangles()
        mesh_in.remove_duplicated_vertices()
        mesh_in.remove_duplicated_triangles()
        mesh_in.remove_non_manifold_edges()

    mesh_in.compute_vertex_normals()
    mesh_in.compute_triangle_normals()

    frame_records = curation_report["frames"]
    frame_files = [Path(r["file_path"]) for r in frame_records]
    extrinsics = vggt_data["extrinsics"]
    intrinsics = vggt_data["intrinsics"]
    depth_maps = vggt_data["depth"]
    R_floor = pcd_data["R_floor"]
    sharpness_list = [r["sharpness"] for r in frame_records]

    # Initialize Stage 2 Texture Engine
    engine = Stage2TextureEngine(
        mesh=mesh_in,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_files,
        depth_maps=depth_maps,
        floor_rotation=R_floor,
        min_cos_theta=0.08
    )

    # 1. Compute Quality Matrix & Line-of-Sight Visibility
    print("[*] Computing photographic quality matrix & line-of-sight visibility...")
    scores, vis_mask, occ_mask = engine.compute_photographic_quality_matrix(frame_sharpness=sharpness_list)

    # 2. Select Master Keyframes
    master_indices = engine.select_master_keyframes(scores, target_cameras=16)

    # 3. Optimize Camera Chart Boundaries via Graph MRF
    print("[*] Optimizing camera chart assignments with seam energy minimization...")
    tri_cam_opt, seam_stats = engine.optimize_camera_charts_mrf(scores, master_indices, seam_penalty=0.45)

    # 4. Exposure Normalization across Master Keyframes
    print("[*] Computing exposure compensation gains across master keyframes...")
    exposure_gains = engine.compute_exposure_compensation_gains(
        master_cameras=master_indices,
        best_cam_per_face=tri_cam_opt,
        vis_mask=vis_mask
    )

    # 5. Synthesize Aspect-Preserving Native 4K Atlas
    atlas_path = tex_dir / "texture_atlas.png"
    print(f"[*] Synthesizing aspect-preserving native 4K atlas: {atlas_path.name}...")
    tile_rects, atlas_stats = engine.synthesize_native_aspect_atlas(
        master_cameras=master_indices,
        exposure_gains=exposure_gains,
        out_atlas_path=atlas_path,
        atlas_dim=4096
    )

    # 6. Generate 4-Class Surface Coverage Map
    print("[*] Generating 4-class diagnostic surface coverage heatmap...")
    cov_ply_path = tex_dir / "surface_coverage_heatmap.ply"
    cov_stats = engine.generate_four_class_coverage_mesh(
        output_ply_path=cov_ply_path,
        vis_mask=vis_mask,
        occ_mask=occ_mask,
        best_cam_per_face=tri_cam_opt
    )

    # 7. Export Canonical Textured Deliverables (GLB, OBJ, MTL, PLY)
    print("[*] Exporting canonical textured deliverables (GLB, OBJ, MTL, PLY)...")
    deliverables = engine.export_stage2_deliverables(
        output_dir=tex_dir,
        base_name="model_tsdf_textured",
        best_cam_per_face=tri_cam_opt,
        tile_rects=tile_rects,
        atlas_path=atlas_path,
        crease_angle_deg=38.0,
        cull_exterior_hull=False
    )

    total_tex_sec = time.perf_counter() - t0_tex

    glb_file = tex_dir / "model_tsdf_textured.glb"
    obj_file = tex_dir / "model_tsdf_textured_canonical.obj"
    ply_file = tex_dir / "model_tsdf_textured.ply"

    # Compute photographic vertex colors on leveled mesh for identical-viewpoint rendering
    photo_v_cols = engine.sample_photographic_vertex_colors(tri_cam_opt)
    leveled_photo_mesh = o3d.geometry.TriangleMesh(mesh_in)
    leveled_photo_mesh.vertex_colors = o3d.utility.Vector3dVector(photo_v_cols)

    tex_report = {
        "texture_baking_time_sec": round(total_tex_sec, 2),
        "target_mesh_vertices": len(mesh_in.vertices),
        "target_mesh_triangles": len(mesh_in.triangles),
        "master_cameras_selected": len(master_indices),
        "seam_statistics": seam_stats,
        "surface_coverage": cov_stats,
        "atlas_statistics": atlas_stats,
        "deliverables": {
            "glb": str(glb_file.relative_to(EXP_DIR)),
            "obj": str(obj_file.relative_to(EXP_DIR)),
            "ply": str(ply_file.relative_to(EXP_DIR)),
            "atlas": str(atlas_path.relative_to(EXP_DIR)),
            "coverage_heatmap": str(cov_ply_path.relative_to(EXP_DIR))
        }
    }
    (tex_dir / "texture_baking_report.json").write_text(json.dumps(tex_report, indent=2))
    print(f"[+] Texture baking complete in {total_tex_sec:.2f}s: GLB saved to {glb_file.name}")

    return {
        "report": tex_report,
        "glb_path": glb_file,
        "obj_path": obj_file,
        "ply_path": ply_file,
        "atlas_path": atlas_path,
        "leveled_photo_mesh": leveled_photo_mesh
    }


# ==============================================================================
# STAGE 7: IDENTICAL-CAMERA VIEWPOINT COMPARISON PANELS
# ==============================================================================

def generate_comparison_review_panels(
    pcd_data: Dict[str, Any],
    tsdf_mesh: o3d.geometry.TriangleMesh,
    poisson_mesh: o3d.geometry.TriangleMesh,
    textured_data: Dict[str, Any],
    vggt_data: Dict[str, Any],
    out_dir: Path
):
    print("\n" + "=" * 76)
    print("  STAGE 7: IDENTICAL-CAMERA VIEWPOINT DIAGNOSTIC RENDERS")
    print("=" * 76)
    comp_dir = out_dir / "visual_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Target models to compare side-by-side:
    # 1. Confidence-Filtered Point Cloud
    # 2. Open3D Scalable TSDF Mesh
    # 3. Screened Poisson Mesh
    # 4. Final Textured TSDF Mesh
    clean_pts = pcd_data["clean_pts_leveled"]
    clean_cols = pcd_data["clean_cols"]
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(clean_pts.astype(np.float64))
    pcd_obj.colors = o3d.utility.Vector3dVector(clean_cols.astype(np.float64) / 255.0)

    models = [
        {"id": "filtered_pcd", "label": "Confidence-Filtered Point Cloud", "data": pcd_obj},
        {"id": "tsdf_mesh", "label": "Open3D Scalable TSDF Mesh (15mm)", "data": tsdf_mesh},
        {"id": "poisson_mesh", "label": "Screened Poisson Mesh (Baseline)", "data": poisson_mesh},
        {"id": "tsdf_textured", "label": "Photographic-Textured TSDF Model", "data": textured_data["leveled_photo_mesh"]}
    ]

    # Define 4 canonical evaluation perspectives:
    # 1. Full Scene Oblique Overview
    # 2. Top-Down Plan View
    # 3. Furniture Primary Close-Up
    # 4. Wall & Floor Surface Alignment
    bbox = tsdf_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    ext = bbox.get_extent()

    views = [
        {
            "id": "overview",
            "name": "Full Scene Oblique Overview",
            "cam_pos": [center[0] - ext[0] * 0.9, center[1] + ext[1] * 1.1, center[2] + ext[2] * 1.2],
            "cam_target": center
        },
        {
            "id": "top_down",
            "name": "Top-Down Plan Flight View",
            "cam_pos": [center[0], center[1] + ext[1] * 2.2, center[2]],
            "cam_target": center
        },
        {
            "id": "furniture",
            "name": "Primary Detail & Furniture Close-Up",
            "cam_pos": [center[0] - ext[0] * 0.3, center[1] + ext[1] * 0.2, center[2] + ext[2] * 0.5],
            "cam_target": [center[0], center[1] - ext[1] * 0.1, center[2]]
        },
        {
            "id": "wall_floor",
            "name": "Wall & Floor Orthogonal Alignment",
            "cam_pos": [center[0] + ext[0] * 0.6, center[1] - ext[1] * 0.1, center[2] + ext[2] * 0.4],
            "cam_target": [center[0] - ext[0] * 0.2, center[1] - ext[1] * 0.2, center[2]]
        }
    ]

    card_w, card_h = 440, 280
    header_h = 60
    panel_w = card_w * len(models)
    panel_h = card_h * len(views) + header_h

    panel = Image.new("RGB", (panel_w, panel_h), color=(15, 18, 24))
    draw = ImageDraw.Draw(panel)

    font_head = get_default_font(18)
    font_sub = get_default_font(12)
    font_lbl = get_default_font(11)

    draw.rectangle([0, 0, panel_w, header_h], fill=(22, 27, 34))
    draw.text((25, 18), "F.O.R.G.E. ARCHITECTURAL COMPARISON — VGGT + OPEN3D SCALABLE TSDF BENCHMARK", fill=(56, 189, 248), font=font_head)
    draw.text((panel_w - 560, 22), "Identical Perspective Comparison: Point Cloud vs TSDF vs Poisson vs Textured TSDF", fill=(148, 163, 184), font=font_sub)

    # Render each cell
    for col_idx, mod in enumerate(models):
        m_id = mod["id"]
        m_lbl = mod["label"]
        x = col_idx * card_w

        for row_idx, vw in enumerate(views):
            v_id = vw["id"]
            v_name = vw["name"]
            y = header_h + row_idx * card_h

            thumb_file = comp_dir / f"render_{m_id}_{v_id}.png"

            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=640, height=400)
                vis.add_geometry(mod["data"])

                opt = vis.get_render_option()
                opt.background_color = np.asarray([0.10, 0.11, 0.14])
                opt.light_on = True

                ctr = vis.get_view_control()
                p = np.array(vw["cam_pos"], dtype=float)
                t = np.array(vw["cam_target"], dtype=float)
                front = t - p
                front /= (np.linalg.norm(front) + 1e-8)
                ctr.set_front(front)
                ctr.set_up([0.0, 1.0, 0.0] if v_id != "top_down" else [0.0, 0.0, -1.0])
                ctr.set_lookat(t)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(str(thumb_file), do_render=True)
                vis.destroy_window()

                im_cell = Image.open(thumb_file).resize((card_w, card_h), Image.Resampling.LANCZOS)
                panel.paste(im_cell, (x, y))
            except Exception as e:
                # Black placeholder with error
                draw.rectangle([x, y, x + card_w, y + card_h], fill=(25, 30, 40))
                draw.text((x + 20, y + card_h // 2), f"Render: {m_id}_{v_id}", fill=(100, 116, 139), font=font_lbl)

            # Label overlay
            draw.rectangle([x, y, x + card_w, y + 26], fill=(18, 24, 34))
            draw.text((x + 8, y + 6), f"{m_lbl} — {v_name}", fill=(241, 245, 249), font=font_lbl)
            draw.rectangle([x, y, x + card_w, y + card_h], outline=(45, 55, 72), width=1)

    panel_path = comp_dir / "master_architectural_comparison_panel.png"
    panel.save(panel_path, quality=95)
    print(f"[+] Saved Master Comparison Panel: {panel_path.name}")


# ==============================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# ==============================================================================

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    print("\n" + "#" * 80)
    print("  F.O.R.G.E. INDOOR RECONSTRUCTION EXPERIMENT (VGGT + TSDF + TEXTURE BAKING)")
    print(f"  Input Video: {VIDEO_PATH}")
    print(f"  Output Directory: {EXP_DIR}")
    print("#" * 80)

    # 1. Curate Frames (~1.0 FPS baseline + adaptive motion disparity keyframes)
    curation_report = curate_indoor_video(VIDEO_PATH, EXP_DIR, target_base_fps=1.0)

    # 2. VGGT Overlapping Window Inference with Robust Sim(3) Alignment
    vggt_data = run_vggt_with_robust_chaining(curation_report, EXP_DIR, image_resolution=320, window_size=6, overlap=3)

    # 3. Confidence-Guided Point Cloud Generation & Floor Leveling
    pcd_data = generate_confidence_guided_point_cloud(curation_report, vggt_data, EXP_DIR, min_confidence=0.75, voxel_size=0.012)

    # 4. Open3D Scalable TSDF Volume Benchmark (10mm, 15mm, 20mm)
    tsdf_results = run_open3d_scalable_tsdf_benchmark(curation_report, vggt_data, pcd_data, EXP_DIR)

    # 5. Screened Poisson Surface Reconstruction (Baseline Comparison)
    poisson_results = run_poisson_baseline(pcd_data, vggt_data, EXP_DIR, poisson_depth=9)

    # 6. Multi-View Photographic Texture Baking on Best TSDF Mesh
    textured_data = bake_photographic_texture_on_tsdf(tsdf_results["best_mesh"], curation_report, vggt_data, pcd_data, EXP_DIR)

    # 7. Render Identical-Camera Diagnostic Viewpoint Comparison Panels
    generate_comparison_review_panels(pcd_data, tsdf_results["best_mesh"], poisson_results["mesh"], textured_data, vggt_data, EXP_DIR)

    total_wall_sec = time.perf_counter() - t_global_start
    print("\n" + "=" * 76)
    print(f"[+] FULL EXPERIMENTAL PIPELINE COMPLETED IN {total_wall_sec:.2f}s ({total_wall_sec/60:.2f} min)!")
    print(f"[+] All deliverables ready in: {EXP_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
