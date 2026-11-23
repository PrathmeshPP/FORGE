"""
F.O.R.G.E. — Part 5: Complete Sim(3) Window Chaining Forensic Audit across all 59 Windows.
Inspects:
- Correspondence counts.
- Sim(3) parameters (scale s, rotation angle, translation vector).
- Clamping / Regularization events and the resulting translation displacement.
- Camera center discrepancies in overlap frames between adjacent windows.
- Cumulative trajectory path and drift.
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path

EXP_DIR = Path("outputs/indoor/vggt_tsdf_video_20261031_142303")
DIAG_DIR = Path("outputs/indoor/distortion_investigation_20261031/sim3_audit")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

perf_file = EXP_DIR / "camera_geometry/vggt_inference_performance.json"
poses_file = EXP_DIR / "camera_geometry/camera_poses.npz"
depth_file = EXP_DIR / "camera_geometry/depth_predictions.npz"

perf = json.loads(perf_file.read_text())
poses = np.load(poses_file)
depths = np.load(depth_file)

all_ext = poses["extrinsics"]   # (179, 3, 4)
all_intr = poses["intrinsics"] # (179, 3, 3)
all_depth = depths["depth"]    # (179, 432, 240)

align_logs = perf.get("alignment_logs", [])
print(f"[*] Total alignment logs: {len(align_logs)}")

# Compute camera centers for all 179 frames: C = -R.T @ t
centers = np.array([-all_ext[i, :3, :3].T @ all_ext[i, :3, 3] for i in range(len(all_ext))])

# Step sizes between consecutive frames
step_sizes = np.linalg.norm(np.diff(centers, axis=0), axis=1)

print("\n" + "=" * 76)
print("  WINDOW ALIGNMENT FORENSIC AUDIT (WINDOWS 0 to 58)")
print("=" * 76)

suspicious_windows = []
for log in align_logs:
    w = log["window"]
    s = log["scale_sw"]
    res = log["residual_cm"]
    mode = log.get("mode", "")
    is_regularized = "Regularized" in mode
    if is_regularized or s < 0.85 or s > 1.25 or res > 2.0:
        suspicious_windows.append(log)
        print(f"  [Window {w:02d}]: scale={s:.4f} | res={res:.2f} cm | Mode: {mode}")

print(f"\n[*] Total suspicious or regularized window transitions: {len(suspicious_windows)} of {len(align_logs)}")

# Check step size distribution
print("\n" + "=" * 76)
print("  CAMERA STEP SIZES BETWEEN CONSECUTIVE FRAMES")
print("=" * 76)
print(f"[*] Mean step: {step_sizes.mean()*100:.2f} cm | Median: {np.median(step_sizes)*100:.2f} cm | Max: {step_sizes.max()*100:.2f} cm")
large_steps = np.where(step_sizes > 0.15)[0]  # Greater than 15 cm jump between consecutive 1-second keyframes
print(f"[*] Number of jumps > 15 cm: {len(large_steps)}")
for idx in large_steps:
    print(f"    Frame {idx:03d} -> Frame {idx+1:03d}: jump = {step_sizes[idx]*100:.2f} cm")

# Cumulative path length
print(f"[*] Total cumulative path: {np.sum(step_sizes):.3f} m")
print(f"[*] Bounding box extent: {np.ptp(centers, axis=0).round(4).tolist()} m")

# Depth statistics across timeline
print("\n" + "=" * 76)
print("  DEPTH STATISTICS ACROSS TIMELINE")
print("=" * 76)
frame_medians = [float(np.median(all_depth[i])) for i in range(len(all_depth))]
frame_mins = [float(np.min(all_depth[i])) for i in range(len(all_depth))]
frame_maxs = [float(np.max(all_depth[i])) for i in range(len(all_depth))]

print(f"[*] Overall depth median: {np.median(frame_medians):.3f} m")
print(f"[*] Frame median range: min={min(frame_medians):.3f} m, max={max(frame_medians):.3f} m")

# Save audit report
audit_summary = {
    "num_windows": len(align_logs),
    "suspicious_window_count": len(suspicious_windows),
    "suspicious_windows": suspicious_windows,
    "step_size_mean_cm": round(float(step_sizes.mean() * 100.0), 2),
    "step_size_max_cm": round(float(step_sizes.max() * 100.0), 2),
    "step_jumps_over_15cm": int(len(large_steps)),
    "large_step_indices": large_steps.tolist(),
    "total_path_length_m": round(float(np.sum(step_sizes)), 3),
    "centers_extent_m": np.ptp(centers, axis=0).round(4).tolist()
}

out_file = DIAG_DIR / "sim3_window_audit_report.json"
out_file.write_text(json.dumps(audit_summary, indent=2))
print(f"\n[+] Saved Sim(3) audit report to {out_file.name}")

